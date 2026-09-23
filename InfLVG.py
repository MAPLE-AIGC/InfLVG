import asyncio
import copy
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import re
from pathlib import Path
import gc

# Standard libraries
import os
import argparse
from copy import deepcopy
import logging
from PIL import Image

# Third-party libraries
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from diffusers.utils import export_to_video

# Local modules
from causvid.models.wan.causal_inference import InferencePipeline
from causvid.models.policy_model import PolicyModel

from causvid.rewards.model_warp import VideoRewardWarp
from causvid.models.data import EventPromptSetDataset
from causvid.ttt import decode_and_save, inference_resample_multi_prev, inference_resample_multi, denoising_multi_steps_multi

import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)   

seed = 123456

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

generator = torch.Generator('cuda').manual_seed(seed)

# ---- Toy MLP Definition ----
def load_pipeline(args, config, device='cuda', dtype=torch.bfloat16):

    logger.info(f"--- loading pipeline ---")
    pipeline = InferencePipeline(config, device=device)
    pipeline.to(device, dtype)
    state_dict = torch.load(os.path.join(args.checkpoint_folder, "model.pt"), map_location=device)[
        'generator']
    pipeline.generator.load_state_dict(
        state_dict, strict=False
    )

    return pipeline

def load_policy_model(device='cuda', dtype=torch.bfloat16):
    logger.info(f"--- loading policy model ---")
    policy_model = PolicyModel(
        in_dim=1536,
        height=30,
        width=52,
        txt_len=512,
        num_layers=int(os.environ.get("pm_num_layers",2)),
        num_linear=int(os.environ.get("pm_num_linear", 2)),
        ffn_dim=1536*4,
        norm_type=os.environ.get("norm_type", "instance")
    ).to(device, dtype)

    logger.info(f"policy_model: {policy_model}\n Trainable Params: {sum(p.numel() for p in policy_model.parameters())/1e6:.2f}M")

    return policy_model


def load_reward_model(device='cuda', dtype=torch.bfloat16):
    logger.info(f"--- loading reward model ---")
    reward_model = VideoRewardWarp(
        use_video_reward_model=False,
        use_rm=False,
        use_hps=False,
        use_face=True,
        use_mps=False,
        use_vlm=True,
        use_clip=True,
        use_clipflan=False,
        face_mode="leftbottom",
        clip_path=os.environ["CLIP_PATH"],
        vlm_ins="/storage/qiguojunLab/fangxueji/Projects/sota_llm/Qwen-VL/vgen_bench_dict.json",
        vlm_path=os.environ["QWEN_PATH"],
        load_from_pretrained="/storage/qiguojunLab/qiguojun/home/Models/KwaiVGI/VideoReward",
        rm_config="/storage/qiguojunLab/qiguojun/home/Models/ImageReward/med_config.json",
        rm_path="/storage/qiguojunLab/qiguojun/home/Models/ImageReward/ImageReward.pt",
        enc_path="/storage/qiguojunLab/qiguojun/.cache/huggingface/hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/1c2b8495b28150b8a4922ee1c8edee224c284c0c/open_clip_pytorch_model.bin",
        hps_path="/storage/qiguojunLab/qiguojun/home/Models/HPSv2/HPS_v2.1_compressed.pt",
        face_path="/storage/qiguojunLab/qiguojun/home/Models/arcface/backbone_ir50_ms1m_epoch120.pth",
        mps_path="/storage/qiguojunLab/fangxueji/Models/MPS/MPS_overall_checkpoint_new.pth",
        mps_processor_path="/storage/qiguojunLab/fangxueji/Models/laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        device=device,
        dtype=dtype,
    )

    return reward_model


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--checkpoint_folder", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--prompt_file_path", type=str)
    args = parser.parse_args()
    return args

# ---- Async Runner for multi-GPU ----
class MultiGPUAsyncRunner:
    def __init__(self, pipeline, device_list):
        self.device_list = device_list
        # replicate and place models on each GPU
        self.pipelines = []
        for i, dev in enumerate(device_list):
            if i == 0:
                self.pipelines.append(pipeline)
                continue
            m = copy.deepcopy(pipeline).to(dev)
            m.denoising_step_list = m.denoising_step_list.to(dev)
            self.pipelines.append(m)

    def decode_and_score(self, pipeline, latents, reward_model, save_path, prompt):
        video = decode_and_save(pipeline, latents, save_video=True, path=f"{save_path}_full.mp4")
        export_to_video(video[len(video)//2+4:], f"{save_path}_half.mp4")
        return [0, 0, 0]
    
    @torch.no_grad()
    async def infer(self, reward_model, kv_cache_list, batch_size, num_blocks, conditional_dict_list, text_index_list, noise, outputs_prev, 
                    softmasks=[], tmp_path=None, idx=0,prompt=None, start_chunk=0, end_chunk=16, prev_len=None ):
        outputs = {dev: deepcopy(outputs_prev) for dev in self.device_list}

        # initialize cache on each pipeline
        for pipe, device in zip(self.pipelines, self.device_list):
            logger.info(f"--- init cross kv cache for device {device} ---")
            pipe._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=device
            )
            logger.info(f"--- init priavte kv cache ---")
            for block in pipe.generator.model.blocks:
                block.self_attn.init_private_kv()

        for block_id in tqdm(range(num_blocks)):
            if block_id < start_chunk: continue
            if block_id >= end_chunk: break

            conditional_dict = conditional_dict_list[text_index_list[block_id]]
            start_frame = block_id * 3
            end_frame = (block_id + 1) * 3
            start_token = start_frame * 1560
            end_token = end_frame * 1560
            noisy_input = noise[:, start_frame:end_frame]
            # --------- denoising on all devices (in parallel) ---------
            denoise_tasks = []
            for i, (device, model) in enumerate(zip(self.device_list, self.pipelines)):
                cond = {k: v.to(device) for k, v in conditional_dict.items()}
                kwargs = dict(
                    kv_cache_list=kv_cache_list,
                    batch_size=batch_size,
                    noisy_input=noisy_input.to(device),
                    conditional_dict=cond,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    current_start=start_token,
                    current_end=end_token,
                    softmask=softmasks[i],
                    kv_update=False,
                    prev_len=prev_len
                )
                task = asyncio.to_thread(denoising_multi_steps_multi, model, **kwargs)
                denoise_tasks.append((device, task))

            # wait for all GPUs to finish this block
            denoise_results = await asyncio.gather(*[t[1] for t in denoise_tasks])
            for (device, _), result in zip(denoise_tasks, denoise_results):
                outputs[device][:, start_frame:end_frame, ...] = result

        # --------- decode and score (in parallel) ---------
        score_tasks = []
        for device, pipelin in zip(self.device_list, self.pipelines):
            latent = outputs[device][:, end_frame - 18:end_frame, ...]
            save_path = tmp_path+f"{1+idx+device.index}"
            task = asyncio.to_thread(self.decode_and_score, pipeline, latent, None, save_path, prompt)
            score_tasks.append(task)

        scores = await asyncio.gather(*score_tasks)
        return scores

def vis_mask(mask, save_path='./tmp.mp4'):
    mask = (1-mask.view(1, -1, 30, 52)/(-1000))
    mask = (mask*255).to(torch.uint8).cpu().numpy()
    mask_vis = [Image.fromarray(it).resize((52*8, 30*8), resample=Image.NEAREST).convert("RGB") for it in mask[0]]
    export_to_video(mask_vis, save_path)


def rename_video_path(path_str: str) -> str:
    pattern = r'(.*?/)(epoch_(\d+))_prompt_(\d+)_clip_(\d+)_.*\.mp4'
    match = re.search(pattern, path_str)
    if not match:
        raise ValueError(f"无法解析路径: {path_str}")
    
    folder, _, epoch_id, prompt_id, clip_id = match.groups()
    new_clip_id = int(clip_id) - 1
    new_name = f"aa_epoch_{epoch_id}_{prompt_id}_clean_clip{new_clip_id}.mp4"
    return str(Path(folder) / new_name), clip_id

def grpo_train(policy_model, group_size, optimizer, parameters_to_optimize,
                sampled_noise, output, 
               conditional_dict_list, text_index_list, prompt, pre_num_frames, 
               inner_loop, exp_dir, epoch, prompt_index, kv_layer_idx,prev_len,start_chunk=16, end_chunk=19):
    def get_group_score(tmp_path):
        group_score = []
        for gid in range(int(os.environ.get("group_size", 16))):
            save_path = tmp_path+f"{1+gid}"
            reward_score_0  = reward_model.vlm_reward_model.reward([f"{save_path}_half.mp4"], [prompt])[0]
            reward_score_1  = reward_model.image_reward_model.reward_each([f"{save_path}_half.mp4"], [prompt])[0] # [". Then ".join(prompts[text_index_list[block_id]-1:1+text_index_list[block_id]])]) #  [prompts[text_index_list[block_id]]])
            hist_clip_path, hist_num = rename_video_path(f"{save_path}_half.mp4")
            reward_score_2  = reward_model.arcface.multiclip(hist_clip_path, hist_num, f"{save_path}_half.mp4")
            reward_score_2  = reward_score_2 if not torch.isnan(reward_score_2).any() else 0.
            score = reward_score_0 + reward_score_1 + reward_score_2
            logger.info(f"score {gid}: {score, reward_score_0, reward_score_1, reward_score_2}")
            group_score.append([reward_score_0, reward_score_1, reward_score_2])
        return torch.tensor(group_score)

    logger.info(f"--- current train prompt: {prompt} ---")
    for j in range(inner_loop):
        sel_from = 1560*1
        sel_end = prev_len - 1560*2
        alpha, beta = policy_model(
            pipeline.kv_cache3[kv_layer_idx]['k'][:,sel_from:sel_end,...].flatten(2).cuda(),
            pipeline.generator.model.text_embedding(conditional_dict_list[1]['sub_prompt_embeds']).cuda() # sub_prompt_embeds prompt_embeds
        )
                    
        group_log_probs = []
        group_masks = []
        group_score = []
        for i in range(group_size):
            softmask, log_probs, _ = policy_model.action(alpha, beta, num_frames=pre_num_frames)

            tmp_path        = f"{exp_dir}/epoch_{epoch}_prompt_{prompt_index+1}_clip_{start_chunk//16}_inner_{j+1}_group_{i+1}"
            vis_mask(softmask[:,:1,:], f"{tmp_path}_mask.mp4")

            group_log_probs.append(log_probs)
            group_masks.append(softmask)
        for i in range(0, group_size, int(os.environ.get("num_gpus", 7))):
            tmp_path        = f"{exp_dir}/epoch_{epoch}_prompt_{prompt_index+1}_clip_{start_chunk//16}_inner_{j+1}_group_"
            group_score_inner = inference_resample_multi(
                self=pipeline,
                noise=sampled_noise,
                conditional_dict_list=conditional_dict_list,
                text_index_list=text_index_list,
                runner=runner,
                outputs_prev=output.clone().detach(),
                softmasks=group_masks[i:i+int(os.environ.get("num_gpus", 7))],
                tmp_path=tmp_path,
                idx=i,
                prompt=prompt,
                start_chunk=start_chunk,
                end_chunk=end_chunk,
                prev_len=prev_len
            )
            group_score.append(group_score_inner)
        
        group_log_probs = torch.stack(group_log_probs).cuda()   # [B]
        group_score   = get_group_score(tmp_path).cuda().view(group_size, -1)   
        
        w1 = int(os.environ.get("w1", 1))
        w2 = int(os.environ.get("w2", 1))
        w3 = int(os.environ.get("w3", 1))
        group_score = group_score[:,0] * w1 + group_score[:, 1] * w2 + group_score[:, 2] * w3
        advantage = (group_score - group_score.mean())/(group_score.std())  # [B]
        logger.info(f"advantage: {advantage.shape, advantage}")
        
        # loss
        loss_policy     = -(torch.exp(group_log_probs - group_log_probs.detach())*advantage)
        # backward
        loss            = loss_policy.mean()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(parameters_to_optimize, max_norm=10.0)
        
        if (j+1) % int(os.environ.get("grad_acc", 4)) == 0:
            logger.info(f"--- Optimizer Step ---")
            optimizer.step()
            optimizer.zero_grad()

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(
            f"[{current_time}] "
            f"[{prompt_index+1}/{len(dataset)}] [Inner {j+1}/{inner_loop}] \n"
            f"Group: {group_score} \n"
            f"Loss = {loss.item()}, \n"
            f"Reward = ({group_score.mean().item()}, {group_score.std().item()})"
        )

    return softmask

def main(pipeline, prompt_index):

    inner_loop_list = [int(num) for num in (os.environ.get("inner_loop", "1")).split(',')]

    group_size = int(os.environ.get("group_size", 16))
    num_clips = int(os.environ.get("num_clips", 4))

    case_id = f"dcolt_vlm_eps_v4_sub_case{prompt_index}"
    exp_dir = f"./logging/{os.environ.get('exp_name', case_id)}"
    os.makedirs(exp_dir, exist_ok=True)  


    total_chunk = int(os.environ.get("total_chunk", 48))
    num_frames = total_chunk*3
    num_blocks = total_chunk
    suffix=""

    sampled_noise = torch.randn(
        [1, num_frames, 16, 60, 104], device=devices[0], dtype=torch.bfloat16
    )

    # for epoch in range(int(os.environ.get("epochs", 3))):
    #     for prompt_index in range(len(dataset)):
    prompts = [dataset[prompt_index][k]+suffix for k in sorted(dataset[prompt_index].keys())]
    # TODO
    prompts = prompts[:num_clips]
    text_index_list = torch.arange(len(prompts)).repeat_interleave(num_blocks // len(prompts)).tolist()

    conditional_dict_list = [
            pipeline.text_encoder(text_prompts=text_prompt) for text_prompt in prompts
    ]
    logger.info(f"Prompts: {prompts}")

    assert len(prompts) == num_clips
    # assert len(prompts) == len(inner_loop_list) + 1

    # ---- START polciy embedding ----
    # sub_prefix = os.path.commonprefix([prompts[0], prompts[1]]).rstrip(" ")
    from prompts.eps_ele_20_v4 import eps_human
    _ = 0
    while (eps_human[_] not in prompts[0]) and (eps_human[_] not in prompts[1]):
        _ += 1
    sub_prefix = eps_human[_]
    _, mask = pipeline.text_encoder.tokenizer(sub_prefix, return_mask=True, add_special_tokens=True)
    sub_prefix_len = mask.sum()
    logger.info(f"sub_prefix: {sub_prefix}, len={sub_prefix_len}")

    conditional_dict_list[0]['sub_prompt_embeds'] = conditional_dict_list[0]['prompt_embeds'][:, :sub_prefix_len, :]
    for i in range(1, len(conditional_dict_list)):
        conditional_dict_list[i]['sub_prompt_embeds'] = conditional_dict_list[0]['sub_prompt_embeds']   # 每个clip的主语都一样
    # ---- END policy embedding ---

    # --- 永远先生成第一个clip ---
    video, output = inference_resample_multi_prev(
        self=pipeline,
        noise=sampled_noise,
        conditional_dict_list=conditional_dict_list,
        text_index_list=text_index_list,
        start_chunk=0,
        end_chunk=16
    )
    export_to_video(
        video, f"{exp_dir}/aa_epoch_{epoch}_{prompt_index+1}_clean_clip0.mp4", fps=16)
    
    # policy model
    kv_layer_idx = 19 # randint(0, len(pipeline.kv_cache3)-1)
    policy_model = None
    for i, (each_clip_loop, prompt) in enumerate(zip(inner_loop_list, prompts[1:])):
        clip_id = i + 1
        # if policy_model is None:
        # ---- 每个clip 重新初始化policy model和optimizer
        policy_model = load_policy_model(device=devices[0])
        opt_steps = int(os.environ.get("opt_steps", 5))
        lr        = float(os.environ.get("lr", 0.0001))
        group_size = int(os.environ.get("group_size", 2))
        policy_model.train()
        policy_model.gradient_checkpointing=True
        parameters_to_optimize = list(policy_model.parameters())

        opt_cls = os.environ.get("opt_cls", "AdamW")
        if opt_cls == "AdamW":
            optimizer              = torch.optim.AdamW(
                                        parameters_to_optimize, 
                                        lr=lr, 
                                        betas=(0.9, 0.999)
                                        )
        else:
            raise NotImplementedError

        optimizer.zero_grad()
    
        # ---- 训练当前clip ---
        gc.collect()
        torch.cuda.empty_cache()

        pre_num_frames = num_frames//num_clips*clip_id
        prev_len = 1560*pre_num_frames
        softmask = grpo_train(
            policy_model, group_size, optimizer, parameters_to_optimize,
            sampled_noise, output, 
            conditional_dict_list, text_index_list, prompt, pre_num_frames, 
            each_clip_loop, exp_dir, epoch, prompt_index, kv_layer_idx, prev_len,
            start_chunk=16*clip_id,
            end_chunk=16*clip_id+3,       # 每次只往后生成3个chunk 用来算reward
        )
        # --- 训练完推理当前clip ---
        video, output = inference_resample_multi_prev(
            self=pipeline,
            noise=sampled_noise,
            conditional_dict_list=conditional_dict_list,
            text_index_list=text_index_list,
            start_chunk=16*clip_id,
            end_chunk=16*(clip_id+1),
            softmask=softmask,
            output=output
        )

        export_to_video(
            video, f"{exp_dir}/aa_epoch_{epoch}_{prompt_index+1}_clean_clip{clip_id}.mp4", fps=16)


# ---- Main Execution ----
if __name__ == "__main__":
    # Settings
    args = init_args()
    devices = [torch.device(f"cuda:{i}") for i in range(int(os.environ.get("num_gpus", 7)))]

    # instantiate base toy model
    pipeline = load_pipeline(args, OmegaConf.load(args.config_path), device=devices[0])

    # prepare runner
    runner = MultiGPUAsyncRunner(pipeline, devices)

    # prepare dataset
    dataset = EventPromptSetDataset(args.prompt_file_path)

    reward_model = load_reward_model(device=torch.device('cuda:1'))

    for epoch in range(int(os.environ.get("epochs", 3))):
        for prompt_index in range(int(os.environ.get("train_start", 0)), int(os.environ.get("train_end", 100))):
            main(pipeline, prompt_index)
