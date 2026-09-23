# Standard libraries
import os
import logging
import copy
from typing import List, Optional
from multiprocessing.pool import ThreadPool

# Third-party libraries
import asyncio
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F

# Diffusers utilities
from diffusers.utils import export_to_video

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)   

total_chunk = int(os.environ.get("total_chunk", 21))
logger.info(f"Total chunks={total_chunk}")

@torch.no_grad()
def decode_and_save(pipeline, latent, path='./tmp.mp4', save_video=True):
    pipeline.vae.to(latent.device)
    video = pipeline.vae.decode_to_pixel(latent)
    video = (video * 0.5 + 0.5).clamp(0, 1)[0].permute(0, 2, 3, 1).cpu().numpy()
    if save_video:
        export_to_video(video, path) 
    return video

@torch.no_grad()
def denoising_multi_steps(self, kv_cache_list, batch_size, noisy_input, conditional_dict, 
                          start_frame, end_frame, current_start, current_end, softmask=None, valid_steps=None):
    if valid_steps is None:
        valid_steps = self.denoising_step_list
    denoised_pred_inner = []
    for index, current_timestep in enumerate(self.denoising_step_list):
        # set current timestep
        if not current_timestep in valid_steps:
            continue
        timestep = torch.ones(
            [batch_size, end_frame-start_frame], device=noisy_input.device, dtype=torch.long) * current_timestep
        if index < len(self.denoising_step_list) - 1:
            denoised_pred = self.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=kv_cache_list[index],
                crossattn_cache=self.crossattn_cache,
                current_start=current_start,
                current_end=current_end,
                softmask=softmask
            )
            next_timestep = self.denoising_step_list[index + 1]
            noisy_input = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                next_timestep *
                torch.ones([batch_size], device=noisy_input.device,
                            dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])
        else:
            # for getting real output
            denoised_pred = self.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=kv_cache_list[index],
                crossattn_cache=self.crossattn_cache,
                current_start=current_start,
                current_end=current_end,
                softmask=softmask
            )
        denoised_pred_inner.append(denoised_pred)

    return denoised_pred_inner


@torch.no_grad()
def denoising_multi_steps_multi(self, kv_cache_list, batch_size, noisy_input, conditional_dict, 
                          start_frame, end_frame, current_start, current_end, softmask=None, kv_update=True,prev_len=None):
    denoised_pred_inner = []
    for index, current_timestep in enumerate(self.denoising_step_list):
        # set current timestep
        time_id=str(int(current_timestep))
        timestep = torch.ones(
            [batch_size, end_frame-start_frame], device=noisy_input.device, dtype=torch.long) * current_timestep
        if index < len(self.denoising_step_list) - 1:
            denoised_pred = self.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=kv_cache_list[index],
                crossattn_cache=self.crossattn_cache,
                current_start=current_start,
                current_end=current_end,
                softmask=softmask,
                kv_update=kv_update,
                prev_len=prev_len,
                time_id=time_id
            )
            next_timestep = self.denoising_step_list[index + 1]
            noisy_input = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                next_timestep *
                torch.ones([batch_size], device=noisy_input.device,
                            dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])
        else:
            # for getting real output
            denoised_pred = self.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=kv_cache_list[index],
                crossattn_cache=self.crossattn_cache,
                current_start=current_start,
                current_end=current_end,
                softmask=softmask,
                kv_update=kv_update,
                prev_len=prev_len,
                time_id=time_id
            )
        denoised_pred_inner.append(denoised_pred)

    return denoised_pred_inner[-1]


@torch.no_grad()
def inference_resample_multi_prev(
    self, noise: torch.Tensor, conditional_dict_list: List[dict], text_index_list: List[int], 
    return_latents: bool = False, kv_cache_list=None, runner=None, start_chunk=0, end_chunk=16, softmask=None, output=None, valid_steps=None, need_decode=True) -> torch.Tensor:
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
    num_blocks = num_frames // self.num_frame_per_block

    output = torch.zeros(
        [batch_size, num_frames, num_channels, height, width],
        device=noise.device,
        dtype=noise.dtype
    ) if output is None else output

    if self.kv_cache1 is None:
        self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device,
            seq_len=self.frame_seq_length*3*total_chunk
        )
    kv_cache_list = [self.kv_cache1, self.kv_cache2, self.kv_cache3, self.kv_cache4]

    for block_id  in tqdm(range(num_blocks)):
        if block_id <start_chunk: continue
        if block_id >= end_chunk: break
        # === 1. init crossattn cache ===
        if block_id == 0 or text_index_list[block_id-1] != text_index_list[block_id]:
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )   
        # === 2. Setting ===
        conditional_dict= conditional_dict_list[text_index_list[block_id]]
        start_frame     = block_id*self.num_frame_per_block
        end_frame       = (block_id+1)*self.num_frame_per_block
        start_token     = start_frame * self.frame_seq_length
        end_token       = end_frame * self.frame_seq_length
        noisy_input     = noise[:, start_frame:end_frame]

        # === 3. Inference ===
        output[:, start_frame:end_frame]    = denoising_multi_steps(self, kv_cache_list, batch_size, noisy_input, conditional_dict, start_frame, end_frame, start_token, end_token, softmask=softmask, valid_steps=valid_steps)[-1]

    # Step 3: Decode the output
    if need_decode:
        video = decode_and_save(self, output[:, :end_chunk * 3], save_video=False)  # 3 frames per block
    else:
        video = None
    return video, output

@torch.no_grad()
def inference_resample_multi_last(
    self, noise: torch.Tensor, conditional_dict_list: List[dict], text_index_list: List[int], 
    softmask=None, kv_cache_list=None, output=None) -> torch.Tensor:
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
    num_blocks = num_frames // self.num_frame_per_block

    kv_cache_list = [self.kv_cache1, self.kv_cache2, self.kv_cache3, self.kv_cache4]

    for block_id  in tqdm(range(num_blocks)):
        if block_id < num_blocks/2: continue
        # === 1. init crossattn cache ===
        if block_id == 0 or text_index_list[block_id-1] != text_index_list[block_id]:
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )   
        # === 2. Setting ===
        conditional_dict= conditional_dict_list[text_index_list[block_id]]
        start_frame     = block_id*self.num_frame_per_block
        end_frame       = (block_id+1)*self.num_frame_per_block
        start_token     = start_frame * self.frame_seq_length
        end_token       = end_frame * self.frame_seq_length
        noisy_input     = noise[:, start_frame:end_frame]

        # === 3. Inference ===
        output[:, start_frame:end_frame]    = denoising_multi_steps(self, kv_cache_list, 
                                                                    batch_size, noisy_input, conditional_dict, 
                                                                    start_frame, end_frame, start_token, end_token,
                                                                    softmask=softmask)[-1]

    # Step 3: Decode the output
    video = decode_and_save(self, output[:,:block_id*3], save_video=False)

    return video, output

@torch.no_grad()
def inference_resample_multi(
    self, noise: torch.Tensor, conditional_dict_list: List[dict], text_index_list: List[int], 
    return_latents: bool = False, kv_cache_list=None, runner=None, reward_model=None, outputs_prev=None,
    softmasks=None, tmp_path=None, idx=0, prompt="", start_chunk=16, end_chunk=19, prev_len=None) -> torch.Tensor:
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
    num_blocks = num_frames // self.num_frame_per_block

    kv_cache_list = [self.kv_cache1, self.kv_cache2, self.kv_cache3, self.kv_cache4]

    # === 3. Inference ===
    results = asyncio.run(runner.infer(reward_model, kv_cache_list, batch_size, num_blocks, conditional_dict_list, text_index_list, noise, 
                                       outputs_prev,softmasks,tmp_path,idx,prompt, start_chunk=start_chunk, end_chunk=end_chunk, prev_len=prev_len))
    return results
