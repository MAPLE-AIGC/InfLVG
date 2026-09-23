import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from .wan.wan_base.modules.model import WAN_CROSSATTENTION_CLASSES, WanLayerNorm


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def wan_rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
    freqs = torch.outer(torch.arange(max_seq_len), freqs)
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

def init_rope_freqs(max_seq_len=1024, head_dim=128, theta=10000.0, dtype=torch.complex128):
    d = head_dim
    
    part1_dim = d - 4 * (d // 6) 
    part2_dim = 2 * (d // 6)    
    part3_dim = 2 * (d // 6)   
    
    freqs1 = wan_rope_params(max_seq_len, part1_dim, theta)
    freqs2 = wan_rope_params(max_seq_len, part2_dim, theta) 
    freqs3 = wan_rope_params(max_seq_len, part3_dim, theta)
    
    freqs_combined = torch.cat([freqs1, freqs2, freqs3], dim=1).to(dtype)
    
    return freqs_combined

class Permute(nn.Module):
    """
    A custom module to permute tensor dimensions.
    Useful for inserting permutation operations into nn.Sequential.
    For InstanceNorm1d with input B x N x C, use Permute((0, 2, 1)).
    """
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

class PMAttentionBlockv8(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))


    def cross_attn_ffn(self, x, context, context_lens):
        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        y = self.ffn(self.norm2(x))
        x = x + y
        return x
        
    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        # self-attention
        # cross-attention & ffn function
        x = self.cross_attn_ffn(x, context, context_lens)
        
        return x
    
class PolicyModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_dim,
        num_layers=2,
        num_linear=2,
        txt_len=512,
        height=30,
        width=52,
        num_heads=12,
        window_size=(-1,-1),
        qk_norm=True,
        ffn_dim=None,
        norm_type="layer",
        *args,
        **kwargs,
    ):
        nn.Module.__init__(self)
        logger.info(f"---using PM v6 ---")

        self.height = height
        self.width = width

        self.freqs = init_rope_freqs(max_seq_len=1024, head_dim=128, theta=10000.0)
        self.grid_sizes = torch.tensor([[3,30,52]])

        self.blocks = nn.ModuleList([
            PMAttentionBlockv8(
                cross_attn_type="t2v_cross_attn",
                dim=in_dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                window_size=window_size,
                qk_norm=qk_norm,
                cross_attn_norm=False,
            )
            
            for _ in range(num_layers)
        ])

        std = float(os.environ.get("std", 0.01))
        self.beta = torch.fill(torch.zeros((1, 48*1560, 1)), std)

        self.scaler = self.build_scaler(in_dim, num_linear, out_dim=1, norm_type=norm_type)

        self.act = nn.ReLU()
        self.sgm = nn.Sigmoid()
        self.gradient_checkpointing = True

        self.top_n = int(os.environ.get("top_n", 6*1560))
        logger.info(f"--- top: {self.top_n} ---")

    def build_scaler(
        self,
        input_dim: int,
        num_linear: int,
        out_dim: int = 2,
        norm_type: str = "instance"  # "instance" or "layer"
    ) -> nn.Sequential:
        """
        Build a sequential model with optional pre- and post-normalization,
        where normalization can be InstanceNorm1d or LayerNorm.

        Args:
            input_dim: Channel dimension of input (C).
            num_linear: Number of linear layers (including final).
            out_dim: Output channel dimension of final layer.
            norm_type: Type of normalization: "instance" or "layer".
        """
        assert norm_type in ("instance", "layer"), \
            f"norm_type must be 'instance' or 'layer', got {norm_type}"

        layers = []
        current_dim = input_dim
        permute_op = Permute((0, 2, 1))

        # Helper to add normalization layer
        def add_norm(dim: int):
            if norm_type == "instance":
                # For InstanceNorm1d, permute to BxCxN, norm over C, permute back
                layers.append(permute_op)
                layers.append(nn.InstanceNorm1d(num_features=dim, affine=True))
                layers.append(permute_op)
            else:
                # LayerNorm over last dimension (C)
                layers.append(nn.LayerNorm(normalized_shape=dim, elementwise_affine=True))

        for i in range(num_linear - 1):
            next_dim = max(current_dim // 4, 4)

            # Pre-Norm
            if int(os.environ.get("use_prenorm", 1)):
                add_norm(current_dim)

            # Linear
            layers.append(nn.Linear(current_dim, next_dim))

            # Post-Norm
            if int(os.environ.get("use_postnorm", 1)):
                add_norm(next_dim)

            # Activation
            layers.append(nn.GELU(approximate='tanh'))
            current_dim = next_dim

        # Final linear layer
        layers.append(nn.Linear(current_dim, out_dim))
        # add_norm(out_dim)
        return nn.Sequential(*layers)

    def find_top_n_mask(self, alpha):
        alpha_flat = alpha.view(-1)  # 展平为一维

        _, indices = torch.topk(alpha_flat, self.top_n, largest=False)
        mask = torch.ones_like(alpha_flat, dtype=torch.uint8)  # 或 torch.bool，如果你需要布尔 mask
        mask[indices] = 0

        mask = mask.view_as(alpha)

        return mask

    def weights_to_mask(self, weights: torch.Tensor, FF: int, H: int, W: int) -> torch.Tensor:
        """
        Convert (B, F*N) token weights to a mask of shape (B, F*H*W) using interpolation.
        """
        B = weights.shape[0]
        # k = int(sqrt(N))
        weights = weights.view(B, FF, 30, 52)  # (B, F, k, k)

        # Use bicubic or nearest interpolation to upscale to (H, W)
        mask = F.interpolate(weights, size=(H, W), mode='nearest')  # (B, F, H, W)
        return mask.view(B, 1, FF * H * W)  # Flatten to (B, F*H*W)

    def get_joint_prob(self, probs, indices, eps=1e-8):
        """
        批量计算不放回采样的 log 联合概率

        参数：
        probs: 形状 (b, m, vocab_size) 的张量，表示每个 batch 和样本的概率分布
        indices: 形状 (b, m, n) 的张量，表示每个 batch 采样的索引（按顺序）

        返回：
        log 联合概率，形状 (b, m)
        """
        b, m, vocab_size = probs.shape  # 获取 batch 维度
        n = indices.shape[-1]  # 采样数 n

        # 获取 indices 对应的概率值 -> (b, m, n)
        # torch.gather 的作用是根据 indices 从 probs 中取出相应位置的值。
        # 例如，如果 probs[0, 0] 是一个长度为 vocab_size 的概率分布，
        # indices[0, 0] 是一个长度为 n 的索引列表，那么
        # torch.gather(probs, dim=-1, index=indices)[0, 0]
        # 将会得到一个长度为 n 的张量，其中包含 probs[0, 0] 中
        # 索引指定位置的概率值。
        probs_gathered = torch.gather(probs, dim=-1, index=indices)

        # 计算剩余概率质量，使用 cumsum() 来批量更新
        # 对于不放回采样，每次采样后，剩余元素的概率质量会减少。
        # remaining_mass 记录了每次采样后的剩余概率质量。
        # torch.cumsum(probs, dim=-1) 计算了概率的累积和，
        # 1 - torch.cumsum(probs, dim=-1) + probs 得到了剩余概率质量。
        # torch.clamp 确保剩余概率质量不会小于 eps。
        remaining_mass = torch.clamp(1 - torch.cumsum(probs_gathered, dim=-1) + probs_gathered, min=eps)

        # 计算 log 概率
        #  计算每个采样的对数概率，并减去剩余质量的对数，以考虑不放回采样。
        log_probs = torch.log(probs_gathered + eps) - torch.log(remaining_mass + eps)

        # 计算联合 log 概率：沿 `n` 维度累加
        # 将每个采样步骤的对数概率相加，得到联合对数概率。
        joint_logprob = log_probs.sum(dim=-1)  # 形状 (b, m)

        return joint_logprob


    def sample_action(self, batch_prob, top_n, mode='train'):
        """
        从给定的概率分布中采样 top_n 个动作（token 索引）。

        参数：
        batch_prob: 形状 (b, m, vocab_size) 的张量，表示每个 batch 中每个样本的概率分布。
        top_n:  整数，表示要采样的动作数量。
        mode:  字符串，指定采样模式，可以是 'train' 或 'test'。
                'train' 模式下使用 torch.multinomial 进行采样，
                'test' 模式下使用 torch.topk 获取概率最高的 top_n 个动作。

        返回：
        形状 (b, m, top_n) 的张量，包含采样的动作索引。
        """
        b, m, vocab_size = batch_prob.shape
        if mode == "test":
            _, batch_topk_indices = torch.topk(batch_prob, k=top_n, dim=-1)
        else:
            # torch.multinomial 用于从多项式分布中采样。
            # batch_prob.view(-1, mem_num) 将 batch_prob 展平为 (b * m, vocab_size) 的形状。
            # num_samples=top_n 指定每个样本采样 top_n 个动作。
            # replacement=False 表示不放回采样。
            batch_topk_indices = torch.multinomial(
                batch_prob.view(-1, vocab_size),    # B 1 FHW -> B FHW
                num_samples=top_n,
                replacement=False
            ).view(b, m, top_n)  # 恢复原始形状 (B 1 FHW)
        return batch_topk_indices

    def action_01(self, alpha, beta, num_frames, samples=None, mode="train"):
        if mode == 'train':
            # dist = torch.distributions.Beta(alpha, beta)
            # dist = torch.distributions.Normal(alpha, beta)
            dist = torch.distributions.Bernoulli(logits=alpha)
            if samples is None:
                samples = dist.sample()                     # b fn
            log_probs = dist.log_prob(samples).sum()
        else:
            # TODO
            samples = alpha
            log_probs = 0.

        if self.top_n > 0:
            alpha_top_min = self.find_top_n_mask(alpha)
            samples_clamp = torch.clamp((alpha_top_min+samples), 0, 1)
        else:
            samples_clamp = samples
        softmask = self.weights_to_mask(samples_clamp*-1000, num_frames, self.height, self.width).repeat(1, 3*1560, 1)
        return softmask, log_probs, samples
    

    def action(self, alpha, beta, num_frames, samples=None, mode="train"):
        # probs = torch.softmax(torch.exp(alpha), dim=-1).unsqueeze(0)
        probs = torch.softmax(alpha, dim=-1).unsqueeze(1)   # B FHW -> B 1 FHW
        topn_indices = self.sample_action(probs, top_n=self.top_n, mode=mode)
        log_probs = self.get_joint_prob(probs, topn_indices).squeeze(0)
        topn_indices = topn_indices.squeeze(0)

        samples = torch.ones_like(alpha, dtype=torch.bfloat16)
        samples[:, topn_indices] = 0.
        hardmask = (samples*-1000).repeat(1, 3*1560, 1) # B Q K
        return hardmask, log_probs[0], samples
            
    
    def forward(self, x, y):
        """
        Input: 
            - x: (b, f*h*w, c)
            - y: (b, l, c)
        Output:
            - alpha: (b, m)
            - beta: (b, m)
        """
        # x = rearrange(x, "b (f h w) c -> b c f h w", h=self.height, w=self.width)

        kwargs = dict(
            seq_lens=torch.tensor([512]),
            grid_sizes=self.grid_sizes,
            freqs=self.freqs,
            context=y,
            context_lens = torch.stack([torch.tensor(y.shape[1])]*len(x))
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        # os.environ['debug']="1"
        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                # logger.info("gradient_checkpointing")
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)
        # os.environ['debug']="0"
        alpha   = self.scaler(x)
        # alpha   = -self.act(alpha)                  # logprobs <= 0, 0<prob<=0
        beta    = self.beta.to(alpha.device)        # 占位
        
        return alpha.flatten(1), beta.flatten(1)