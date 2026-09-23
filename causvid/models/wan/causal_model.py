from .wan_base.modules.attention import attention
from .wan_base.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    Head,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import os

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune")


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame +
                     f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.kv_cache_private = None
        
    def init_private_kv(self):
        self.kv_cache_private = {"k": {
            "1000": [],
            "757": [],
            "522": [],
        }, "v": {
            "1000": [],
            "757": [],
            "522": [],
        }}
        

    @torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=False, enable_mem_efficient=True)
    def sdpa(self, query, key, value):
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1))
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        # attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        # attn_weight += attn_bias
        attn_bias = torch.rand(L, S, dtype=query.dtype, device=query.device)
        attn_bias[:,-L:] = 1.
        attn_weight *= attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        x = attn_weight @ value
        del query, key, value
        return x
    
    @torch.no_grad()    # TODO
    def forward_bak(self, x, seq_lens, grid_sizes, freqs, block_mask, 
                kv_cache=None, current_start=0, current_end=0, softmask=None, kv_update=True, time_id=None, prev_len=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None:
            raise NotImplementedError
        
        roped_query = causal_rope_apply(
            q, grid_sizes, freqs, start_frame=current_start // math.prod(grid_sizes[0][1:]).item()).type_as(v)
        roped_key = causal_rope_apply(
            k, grid_sizes, freqs, start_frame=current_start // math.prod(grid_sizes[0][1:]).item()).type_as(v)

        device = roped_key.device
        if kv_update:
            kv_cache["k"][:, current_start:current_end] = roped_key
            kv_cache["v"][:, current_start:current_end] = v
            key_cur = kv_cache["k"][:, :current_end].cuda()
            value_cur = kv_cache["v"][:, :current_end].cuda()
        else:
            assert prev_len is not None
            # prev_num = int(os.environ.get("total_chunk", 21))*3//2*1560
            self.kv_cache_private["k"][time_id].append(roped_key)
            self.kv_cache_private["v"][time_id].append(v)
            key1 = kv_cache["k"][:, :prev_len].to(device)
            key2 = self.kv_cache_private["k"][time_id]
            key_cur = torch.cat([key1]+key2, dim=1)
            value_cur = torch.cat([kv_cache["v"][:, :prev_len].to(device)]+self.kv_cache_private["v"][time_id], dim=1)

        if softmask is not None:
            # attention_mask = torch.zeros((1, roped_query.shape[1], current_end), dtype=roped_query.dtype, device=roped_query.device)
            # attention_mask[:,:,:softmask.shape[-1]] = softmask
            attention_mask = torch.ones(
                (1, roped_query.shape[1], current_end), dtype=torch.bool, device=roped_query.device
            )
            attention_mask[:,:,:softmask.shape[-1]] = (softmask == 0)
            # print(f"attention_mask: {attention_mask.shape}, k: {key_cur.shape}, current_end:{current_end}, {len(self.kv_cache_private['k'])}")

        else:
            attention_mask = None
        with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=False, enable_mem_efficient=True):
            x = F.scaled_dot_product_attention(
                roped_query.transpose(1,2), 
                key_cur.transpose(1,2), 
                value_cur.transpose(1,2),
                attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            ).transpose(1,2)
        del attention_mask

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

    
    @torch.no_grad()    # TODO
    def forward(self, x, seq_lens, grid_sizes, freqs, block_mask, 
                kv_cache=None, current_start=0, current_end=0, softmask=None, kv_update=True, time_id=None, prev_len=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None:
            raise NotImplementedError
        
        roped_query = causal_rope_apply(
            q, grid_sizes, freqs, start_frame=current_start // math.prod(grid_sizes[0][1:]).item()).type_as(v)
        roped_key = causal_rope_apply(
            k, grid_sizes, freqs, start_frame=current_start // math.prod(grid_sizes[0][1:]).item()).type_as(v)

        device = roped_key.device
        if kv_update:
            kv_cache["k"][:, current_start:current_end] = roped_key
            kv_cache["v"][:, current_start:current_end] = v
            key_cur = kv_cache["k"][:, :current_end].cuda()
            value_cur = kv_cache["v"][:, :current_end].cuda()
        else:
            assert prev_len is not None
            # prev_num = int(os.environ.get("total_chunk", 21))*3//2*1560
            self.kv_cache_private["k"][time_id].append(roped_key)
            self.kv_cache_private["v"][time_id].append(v)
            key1 = kv_cache["k"][:, :prev_len].to(device)
            key2 = self.kv_cache_private["k"][time_id]
            key_cur = torch.cat([key1]+key2, dim=1)
            value_cur = torch.cat([kv_cache["v"][:, :prev_len].to(device)]+self.kv_cache_private["v"][time_id], dim=1)

        if softmask is not None:
            attention_mask = torch.ones(
                (1, roped_query.shape[1], current_end), dtype=torch.bool, device=roped_query.device
            )
            # attention_mask[:,:,:softmask.shape[-1]] = (softmask == 0)
            sel_from = 1560*1
            sel_end = softmask.shape[-1] + sel_from
            # assert sel_end + 1560*2 == current_end
            attention_mask[:,:,sel_from:sel_end,...] = (softmask == 0)

            #   roped_query: (B, N, H, C)
            #   key_cur:     (B, M, H, C)
            #   value_cur:   (B, M, H, C)
            #   attention_mask: (1, N, M), dtype=torch.bool

            # 1) 找出哪些 key/value positions 至少有一个 query 要“看”它
            #    valid_kv[j] = True 表示第 j 个 key/value 在某些 query 上是 unmasked
            valid_kv = attention_mask.any(dim=1).squeeze(0)    # → shape (M,)

            # 2) 用 boolean indexing 裁剪 key 和 value
            #    保留 batch 维不变，但 seq_kv 维只保留 valid_kv 为 True 的位置
            pruned_key   = key_cur  [:, valid_kv, :, :]       # → (B, M′, H, C)
            pruned_value = value_cur[:, valid_kv, :, :]       # → (B, M′, H, C)

            # 3) 构造新的 attention_mask′，只对剩下的 M′ 个 positions 生效
            #    注意仍保持 shape = (1, N, M′)
            pruned_mask = None # attention_mask[:, :, valid_kv]      # → (1, N, M′)
            del attention_mask

        else:
            pruned_key = key_cur
            pruned_value = value_cur
            pruned_mask = None

        # 4) 调用 scaled_dot_product_attention
        with torch.backends.cuda.sdp_kernel(
                enable_math=False,
                enable_flash=False,
                enable_mem_efficient=True
            ):
            x = F.scaled_dot_product_attention(
                roped_query.transpose(1,2),              # (B, H, N, C)
                pruned_key.transpose(1,2),                # (B, H, M′, C)
                pruned_value.transpose(1,2),              # (B, H, M′, C)
                attn_mask=pruned_mask,
                dropout_p=0.0,
                is_causal=False
            ).transpose(1,2)                            # → (B, N, H, C)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

    # def scaled_dot_product_attention(self, query, key, value, attn_mask=None, dropout_p=0.0,
    #         is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    #     L, S = query.size(-2), key.size(-2)
    #     scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    #     attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    #     if is_causal:
    #         assert attn_mask is None
    #         temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
    #         attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    #         attn_bias.to(query.dtype)
    #     attn_mask = None
    #     if attn_mask is not None:
    #         if attn_mask.dtype == torch.bool:
    #             attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
    #         else:
    #             attn_bias = attn_mask + attn_bias

    #     if enable_gqa:
    #         key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
    #         value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    #     attn_weight = query @ key.transpose(-2, -1) * scale_factor
    #     top_k = int(os.environ.get("top_n"))
    #     if attn_weight.shape[-1] > top_k:
    #         _, top_k_indices = torch.topk(attn_weight, k=top_k, dim=-1, largest=True)
    #         top_k_mask = torch.zeros_like(attn_weight, dtype=torch.bool)
    #         top_k_mask.scatter_(-1, top_k_indices, True)
    #         attn_bias = attn_bias.masked_fill(top_k_mask, float("-inf"))
    #         import pdb; pdb.set_trace()
    #     attn_weight += attn_bias
    #     attn_weight = torch.softmax(attn_weight, dim=-1)
    #     attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    #     return attn_weight @ value

    # @torch.no_grad()    # TODO
    # def forward(self, x, seq_lens, grid_sizes, freqs, block_mask, 
    #             kv_cache=None, current_start=0, current_end=0, softmask=None, kv_update=True, time_id=None, prev_len=None):
    #     r"""
    #     Args:
    #         x(Tensor): Shape [B, L, num_heads, C / num_heads]
    #         seq_lens(Tensor): Shape [B]
    #         grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
    #         freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
    #         block_mask (BlockMask)
    #     """
    #     b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

    #     # query, key, value function
    #     def qkv_fn(x):
    #         q = self.norm_q(self.q(x)).view(b, s, n, d)
    #         k = self.norm_k(self.k(x)).view(b, s, n, d)
    #         v = self.v(x).view(b, s, n, d)
    #         return q, k, v

    #     q, k, v = qkv_fn(x)

    #     if kv_cache is None:
    #         raise NotImplementedError
        
    #     roped_query = causal_rope_apply(
    #         q, grid_sizes, freqs, start_frame=current_start // math.prod(grid_sizes[0][1:]).item()).type_as(v).transpose(1,2)
    #     roped_key = causal_rope_apply(
    #         k, grid_sizes, freqs, start_frame=current_start // math.prod(grid_sizes[0][1:]).item()).type_as(v).transpose(1,2)

    #     value = v.transpose(1,2)
    #     device = roped_key.device

    #     if softmask is None:
    #         # 4) 调用 scaled_dot_product_attention
    #         with torch.backends.cuda.sdp_kernel(
    #                 enable_math=False,
    #                 enable_flash=False,
    #                 enable_mem_efficient=True
    #             ):
    #             x = F.scaled_dot_product_attention(
    #                 roped_query,              # (B, H, N, C)
    #                 roped_key,                # (B, H, M′, C)
    #                 value,              # (B, H, M′, C)
    #                 attn_mask=softmask,
    #                 dropout_p=0.0,
    #                 is_causal=False
    #             ).transpose(1,2)                            # → (B, N, H, C)
    #     else:
    #         x = self.scaled_dot_product_attention(
    #             roped_query, roped_key, value, attn_mask=softmask
    #         ).transpose(1,2)  
    #     # output
    #     x = x.flatten(2)
    #     x = self.o(x)
    #     return x
    
class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 last_block=False):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, window_size, qk_norm,
                                                eps)
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

        use_parall_ffn = bool(int(os.environ.get("use_parall_ffn", 0)))
        if last_block  and use_parall_ffn:
            self.ffn_new = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))
        else:
            self.ffn_new = None
        
        use_delta_ffn = bool(int(os.environ.get("use_delta_ffn", 0)))
        if last_block and use_delta_ffn:
            self.delta = nn.Parameter(torch.zeros(1, ffn_dim))
        else:
            self.delta = None

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        current_end=0,
        softmask=None,
        kv_update=True,
        prev_len=None,
        time_id=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
             * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, current_end, softmask=softmask,kv_update=kv_update,prev_len=prev_len,time_id=time_id)

        # with amp.autocast(dtype=torch.float32):
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen))
                 * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            with torch.no_grad():   # TODO
                x = x + self.cross_attn(self.norm3(x), context,
                                        context_lens, crossattn_cache=crossattn_cache)
            x_norm = (self.norm2(x).unflatten(dim=1, sizes=(num_frames,frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            if self.delta is None:
                y = self.ffn(x_norm)
            else:
                y = self.ffn[0](x_norm) # linear 1
                y = y + self.delta      # delta
                y = self.ffn[1](y)      # GeLU
                y = self.ffn[2](y)      # linear 2
            if self.ffn_new is not None:
                y_new = self.ffn_new(x_norm)
                y = y + y_new # TODO: gating

            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(
            self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) *
            (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    window_size, qk_norm, cross_attn_norm, eps, last_block=(layer_id+1) == num_layers)
            for layer_id in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6), use_RIFLEx=True), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))
            ],dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        return block_mask

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        current_end: int = 0,
        softmask=None,
        kv_update=True,
        prev_len=None,
        time_id=None,
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])

        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]).to(device))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                assert False
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "current_end": current_end,
                        "softmask": softmask,
                        "kv_update": kv_update,
                        "prev_len": prev_len,
                        "time_id": time_id,
                    }
                )
                x = block(x, **kwargs)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            raise NotImplementedError
            # return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
