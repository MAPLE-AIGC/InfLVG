from .. import (
    get_diffusion_wrapper,
    get_text_encoder_wrapper,
    get_vae_wrapper
)
from typing import List, Optional
from tqdm import tqdm
import torch
import copy
from diffusers.configuration_utils import ConfigMixin
from diffusers import ModelMixin
import os

class InferencePipeline(ModelMixin, ConfigMixin):
    def __init__(self, args, device):
        super().__init__()
        # Step 1: Initialize all models
        self.generator_model_name = getattr(
            args, "generator_name", args.model_name)
        self.generator = get_diffusion_wrapper(
            model_name=self.generator_model_name)()
        self.text_encoder = get_text_encoder_wrapper(
            model_name=args.model_name)()
        self.vae = get_vae_wrapper(model_name=args.model_name)()

        # Step 2: Initialize all causal hyperparmeters
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long, device=device)
        assert self.denoising_step_list[-1] == 0
        # remove the last timestep (which equals zero)
        self.denoising_step_list = self.denoising_step_list[:-1]

        self.scheduler = self.generator.get_scheduler()

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.kv_cache3 = None
        self.kv_cache4 = None
        self.args = args
        self.num_frame_per_block = getattr(
            args, "num_frame_per_block", 1)

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def _initialize_kv_cache(self, batch_size, dtype, device, seq_len=None):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        total_gpus = torch.cuda.device_count()
        # 32 个chunk * 3 可以放满一个GPU，即一个GPU可以放2个clip的所有timestep下的kv cache
        # so，一个GPU可以放96个chunk，96/16 = 6，刚好最多6个clip，每个timestep放在3张卡上
        device = f'cuda:{total_gpus-1}'
        kv_cache1 = []
        seq_len = self.frame_seq_length*3*int(os.environ.get("total_chunk", 7)) if seq_len is None else seq_len
        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, seq_len, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, seq_len, 12, 128], dtype=dtype, device=device)
            })
        device = f'cuda:{total_gpus-2}'
        kv_cache2 = []
        seq_len = self.frame_seq_length*3*int(os.environ.get("total_chunk", 7)) if seq_len is None else seq_len
        for _ in range(self.num_transformer_blocks):
            kv_cache2.append({
                "k": torch.zeros([batch_size, seq_len, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, seq_len, 12, 128], dtype=dtype, device=device)
            })
        device = f'cuda:{total_gpus-3}'
        kv_cache3 = []
        seq_len = self.frame_seq_length*3*int(os.environ.get("total_chunk", 7)) if seq_len is None else seq_len
        for _ in range(self.num_transformer_blocks):
            kv_cache3.append({
                "k": torch.zeros([batch_size, seq_len, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, seq_len, 12, 128], dtype=dtype, device=device)
            })


        self.kv_cache1 = kv_cache1
        self.kv_cache2 = kv_cache2
        self.kv_cache3 = kv_cache3

        print(f"\n[WARMING] Your have allocated 2 GPUs for denosing KV Cache.\n")

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })

        self.crossattn_cache = crossattn_cache  # always store the clean cache

    def inference(self, noise: torch.Tensor, text_prompts: List[str], start_latents: Optional[torch.Tensor] = None, return_latents: bool = False) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        output = torch.zeros(
            [batch_size, num_frames, num_channels, height, width],
            device=self.device,
            dtype=self.dtype
        )

        # Step 1: Initialize KV cache
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=self.dtype,
                device=self.device
            )

            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=self.dtype,
                device=self.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False

        num_input_blocks = start_latents.shape[1] // self.num_frame_per_block if start_latents is not None else 0

        # Step 2: Temporal denoising loop
        num_blocks = num_frames // self.num_frame_per_block
        print(f"self.kv_cache1: {self.kv_cache1[0]['k'].shape}")
        for block_index in tqdm(range(num_blocks)):
           
            noisy_input = noise[:, block_index *
                                self.num_frame_per_block:(block_index + 1) * self.num_frame_per_block]

            if start_latents is not None and block_index < num_input_blocks:
                import pdb; pdb.set_trace() # 上一个clip的后3帧
                timestep = torch.ones(
                    [batch_size, self.num_frame_per_block], device=self.device, dtype=torch.int64) * 0

                current_ref_latents = start_latents[:, block_index * self.num_frame_per_block:(
                    block_index + 1) * self.num_frame_per_block]
                output[:, block_index * self.num_frame_per_block:(
                    block_index + 1) * self.num_frame_per_block] = current_ref_latents  # 为什么要复写

                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,      # 模拟最后一步去噪，应该是想要把kv cache存下来，那为何要rollout，不直接在一遍的时候直接自回归生成很长的？
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=block_index * self.num_frame_per_block * self.frame_seq_length,
                    current_end=(block_index + 1) *
                    self.num_frame_per_block * self.frame_seq_length
                )
                continue

            # Step 2.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                # set current timestep
                timestep = torch.ones(
                    [batch_size, self.num_frame_per_block], device=self.device, dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=block_index * self.num_frame_per_block * self.frame_seq_length,
                        current_end=(
                            block_index + 1) * self.num_frame_per_block * self.frame_seq_length
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep *
                        torch.ones([batch_size], device="cuda",
                                   dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=block_index * self.num_frame_per_block * self.frame_seq_length,
                        current_end=(
                            block_index + 1) * self.num_frame_per_block * self.frame_seq_length
                    )
            # Step 2.2: rerun with timestep zero to update the cache
            output[:, block_index * self.num_frame_per_block:(
                block_index + 1) * self.num_frame_per_block] = denoised_pred

            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=block_index * self.num_frame_per_block * self.frame_seq_length,
                current_end=(block_index + 1) *
                self.num_frame_per_block * self.frame_seq_length
            )

        # Step 3: Decode the output
        video = self.vae.decode_to_pixel(output)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_latents:
            return video, output
        else:
            return video
