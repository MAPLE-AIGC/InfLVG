import json
import os
from collections.abc import Mapping

import torch
import numpy as np
from .vision_process import process_vision_info

from .data import DataConfig
from .utils import ModelConfig, PEFTLoraConfig, TrainingConfig
from .utils import load_model_from_checkpoint
from .train_reward import create_model_and_processor
from .prompt_template import build_prompt


def load_configs_from_json(config_path):
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # del config_dict["training_args"]["_n_gpu"]
    del config_dict["data_config"]["meta_data"]
    del config_dict["data_config"]["data_dir"]

    return config_dict["data_config"], None, config_dict["model_config"], config_dict["peft_lora_config"], \
           config_dict["inference_config"] if "inference_config" in config_dict else None

class VideoVLMRewardInference():
    def __init__(self, load_from_pretrained, load_from_pretrained_step=-1, device='cuda', dtype=torch.bfloat16):
        config_path = os.path.join(load_from_pretrained, "model_config.json")
        data_config, _, model_config, peft_lora_config, inference_config = load_configs_from_json(config_path)
        data_config = DataConfig(**data_config)
        model_config = ModelConfig(**model_config)
        peft_lora_config = PEFTLoraConfig(**peft_lora_config)

        # training_args = TrainingConfig(
        #     load_from_pretrained=load_from_pretrained,
        #     load_from_pretrained_step=load_from_pretrained_step,
        #     gradient_checkpointing=False,
        #     disable_flash_attn2=False,
        #     bf16=True if dtype == torch.bfloat16 else False,
        #     fp16=True if dtype == torch.float16 else False,
        #     output_dir="",
        # )
        
        model, processor, peft_config = create_model_and_processor(
            model_config=model_config,
            peft_lora_config=peft_lora_config,
            training_args=None,
        )

        self.device = device

        model, checkpoint_step = load_model_from_checkpoint(model, load_from_pretrained, load_from_pretrained_step)
        model.eval()

        self.model = model
        self.processor = processor

        self.model.to(self.device)

        self.data_config = data_config

        self.inference_config = inference_config

    def _norm(self, reward):
        if self.inference_config is None:
            return reward
        else:
            reward['VQ'] = (reward['VQ'] - self.inference_config['VQ_mean']) / self.inference_config['VQ_std']
            reward['MQ'] = (reward['MQ'] - self.inference_config['MQ_mean']) / self.inference_config['MQ_std']
            reward['TA'] = (reward['TA'] - self.inference_config['TA_mean']) / self.inference_config['TA_std']
            return reward

    def _pad_sequence(self, sequences, attention_mask, max_len, padding_side='right'):
        """
        Pad the sequences to the maximum length.
        """
        assert padding_side in ['right', 'left']
        if sequences.shape[1] >= max_len:
            return sequences, attention_mask
        
        pad_len = max_len - sequences.shape[1]
        padding = (0, pad_len) if padding_side == 'right' else (pad_len, 0)

        sequences_padded = torch.nn.functional.pad(sequences, padding, 'constant', self.processor.tokenizer.pad_token_id)
        attention_mask_padded = torch.nn.functional.pad(attention_mask, padding, 'constant', 0)

        return sequences_padded, attention_mask_padded
    
    def _prepare_input(self, data):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.device}
            ## TODO: Maybe need to add dtype
            # if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
            #     # NLP models inputs are int/uint and those get adjusted to the right dtype of the
            #     # embedding. Other models such as wav2vec2's inputs are already float and thus
            #     # may need special handling to match the dtypes of the model
            #     kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
            return data.to(**kwargs)
        return data
    
    def _prepare_inputs(self, inputs):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError
        return inputs
    

    
    def prepare_batch(self, video_paths, prompts, fps=None, num_frames=None, max_pixels=None,):
        fps = self.data_config.fps if fps is None else fps
        num_frames = self.data_config.num_frames if num_frames is None else num_frames
        max_pixels = self.data_config.max_frame_pixels if max_pixels is None else max_pixels

        if num_frames is None:
            chat_data = [
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video", 
                                "video": f"{video_path}", 
                                "max_pixels": max_pixels, 
                                "fps": fps,
                                "sample_type": self.data_config.sample_type,
                            },
                            {"type": "text", "text": build_prompt(prompt, self.data_config.eval_dim, self.data_config.prompt_template_type)},
                        ],
                    },
                ] for video_path, prompt in zip(video_paths, prompts)
            ]
        else:
            chat_data = [
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": f"{video_path}", 
                                "max_pixels": max_pixels, 
                                "nframes": num_frames,
                                "sample_type": self.data_config.sample_type,
                            },
                            {"type": "text", "text": build_prompt(prompt, self.data_config.eval_dim, self.data_config.prompt_template_type)},
                        ],
                    },
                ] for video_path, prompt in zip(video_paths, prompts)
            ]
        image_inputs, video_inputs = process_vision_info(chat_data)

        batch = self.processor(
            text=self.processor.apply_chat_template(chat_data, tokenize=False, add_generation_prompt=True),
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        batch = self._prepare_inputs(batch)
        return batch
    
    def prepare_batch_from_frames(self, video_frames_list, prompts, fps=None, num_frames=None, max_pixels=None):
        """
        处理 PIL 视频帧列表，生成模型输入批次
        Args:
            video_frames_list: List[List[PIL.Image]]，每个元素是单视频的帧列表
            prompts: List[str]，每个视频对应的提示文本
        Returns:
            batch: 包含处理后的张量输入
        """
        fps = self.data_config.fps if fps is None else fps
        num_frames = self.data_config.num_frames if num_frames is None else num_frames
        max_pixels = self.data_config.max_frame_pixels if max_pixels is None else max_pixels

        # 构建虚拟的 chat_data 结构（模拟视频路径）
        chat_data = []
        for frames, prompt in zip(video_frames_list, prompts):
            # 假设 process_vision_info 能处理 PIL 帧，需要自定义处理逻辑
            chat_entry = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": frames,  # 传递 PIL 帧列表
                            "max_pixels": max_pixels,
                            "nframes": num_frames,
                            "sample_type": self.data_config.sample_type,
                        },
                        {"type": "text", "text": build_prompt(prompt, self.data_config.eval_dim, self.data_config.prompt_template_type)},
                    ],
                }
            ]
            chat_data.append(chat_entry)

        # 自定义处理 PIL 帧的逻辑（需修改 process_vision_info 或新建函数）
        image_inputs, video_inputs = process_vision_frames(chat_data)  # 新函数或修改现有函数

        # 生成模型输入批次
        batch = self.processor(
            text=self.processor.apply_chat_template(chat_data, tokenize=False, add_generation_prompt=True),
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        batch = self._prepare_inputs(batch)
        return batch

    @torch.no_grad()
    def reward(self, video_paths, prompts, fps=None, num_frames=None, max_pixels=None, use_norm=True):
        """
        Inputs:
            video_paths: List[str], B paths of the videos.
            prompts: List[str], B prompts for the videos.
            eval_dims: List[str], N evaluation dimensions.
            fps: float, sample rate of the videos. If None, use the default value in the config.
            num_frames: int, number of frames of the videos. If None, use the default value in the config.
            max_pixels: int, maximum pixels of the videos. If None, use the default value in the config.
            use_norm: bool, whether to rescale the output rewards
        Outputs:
            Rewards: List[dict], N + 1 rewards of the B videos.
        """
        assert fps is None or num_frames is None, "fps and num_frames cannot be set at the same time."
        
        batch = self.prepare_batch(video_paths, prompts, fps, num_frames, max_pixels)

        rewards = self.model(
            return_dict=True,
            **batch
        )["logits"]

        rewards = [{'VQ': reward[0].item(), 'MQ': reward[1].item(), 'TA': reward[2].item()} for reward in rewards]
        all_rewards = []
        for i in range(len(rewards)):
            if use_norm:
                rewards[i] = self._norm(rewards[i])
            rewards[i]['Overall'] = rewards[i]['VQ'] + rewards[i]['MQ'] + rewards[i]['TA']
            # print("overall:", rewards[i]['Overall'], "reward vq:", rewards[i]['VQ'], "mq:", rewards[i]['MQ'], "ta:", rewards[i]['TA'])
            # print(f'overall:{rewards[i]['Overall']}, reward vq:{rewards[i]['VQ']}, mq:{rewards[i]['MQ']}, ta:{rewards[i]['TA']}, ')
            all_rewards.append(rewards[i]['Overall'])
        return torch.Tensor(all_rewards) # rewards
    
    # @torch.no_grad()
    # def reward_latents(self, video_frames_list, prompts, fps=None, num_frames=None, max_pixels=None, use_norm=True):
    #     """
    #     直接接收 PIL 视频帧列表计算奖励
    #     Args:
    #         video_frames_list: List[List[PIL.Image]]，每个元素是单视频的帧列表
    #         prompts: List[str]，每个视频对应的提示文本
    #     Returns:
    #         rewards: torch.Tensor，奖励值
    #     """
    #     # 生成输入批次
    #     batch = self.prepare_batch_from_frames(video_frames_list, prompts, fps, num_frames, max_pixels)
        
    #     # 执行模型推理
    #     rewards = self.model(
    #         return_dict=True,
    #         **batch
    #     )["logits"]

    #     # 解析奖励值
    #     reward_list = [{'VQ': reward[0].item(), 'MQ': reward[1].item(), 'TA': reward[2].item()} 
    #                 for reward in rewards]
    #     all_rewards = []
    #     for reward in reward_list:
    #         if use_norm:
    #             reward = self._norm(reward)
    #         reward['Overall'] = reward['VQ'] + reward['MQ'] + reward['TA']
    #         all_rewards.append(reward['Overall'])
        
    #     return torch.tensor(all_rewards, device=self.device)

def process_vision_frames(chat_data):
    """
    处理包含 PIL 帧的 chat_data，提取视频特征
    """
    image_inputs = []
    video_inputs = []

    for chat in chat_data:
        for content in chat[0]["content"]:
            if content["type"] == "video":
                frames = content["video"]  # 获取 PIL 帧列表
                # 将 PIL 帧转换为张量（示例代码，需根据实际预处理逻辑调整）
                video_tensor = torch.stack([torch.from_numpy(np.array(frame)) for frame in frames])
                video_tensor = video_tensor.permute(0, 3, 1, 2)  # (T, C, H, W)
                # 添加至视频输入列表
                video_inputs.append(video_tensor)
            elif content["type"] == "image":
                # 处理图像（如果有）
                pass
    return image_inputs, video_inputs

if __name__ == "__main__":
    load_from_pretrained = "/storage/qiguojunLab/qiguojun/home/Models/KwaiVGI/VideoReward"
    device = "cuda:1"
    dtype = torch.bfloat16

    inferencer = VideoVLMRewardInference(load_from_pretrained, device=device, dtype=dtype)


    video_paths = [
        "/storage/qiguojunLab/qiguojun/maliyuan/trl/tmp_outputs/vid_proc0002_gstep0008_bsid00.mp4"
    ]

    prompts = [
        "In the video, a small owl with brown and white feathers is perched on a branch of a tree.  The owl has a round face with large, expressive eyes and a small beak.  The tree has green leaves and some yellow flowers.  The background is blurred, focusing attention on the owl.  The owl appears to be looking around, possibly observing its surroundings or searching for prey.  The lighting is natural, suggesting that the video was shot during the day.  The overall scene is peaceful and serene, with the owl being the main subject of the video."
    ]


    with torch.no_grad():
        rewards = inferencer.reward(video_paths, prompts, use_norm=True)
        print(rewards)