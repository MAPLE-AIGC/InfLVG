from .moe_reward import InternVLChatRewardModeling, InternVLChatRewardModelingConfig
from .internvl2 import prepare_chat_input
from .data_processor import load_video
import torch
from transformers import AutoTokenizer

class MJVideo():
    def __init__(
        self,
        model_name,
        device,
        dtype=torch.bfloat16,
        **kwargs,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.config = InternVLChatRewardModelingConfig.from_pretrained(model_name, pad_token_id=self.tokenizer.pad_token_id, num_objectives=10, num_aspects=3, aspect2criteria={
            0: [0, 1, 2],
            1: [3, 4, 5],
            2: [6, 7, 8, 9]
        }, gating_temperature=1.0, gating_hidden_dim=1024, gating_n_hidden=3)

        self.generation_config = dict(max_new_tokens=1024, do_sample=True)

        self.model = InternVLChatRewardModeling(name=model_name, config=self.config).cuda()
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = self.model.to(device, dtype)
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        self.model.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.eval()

        self.device = device

    @torch.no_grad()
    def reward(self, video_paths, prompts, **kwargs):
        scores = []
        for video_path, caption in zip(video_paths, prompts):
            pixel_values, num_patches_list = load_video(video_path, num_segments=8, max_num=1)
            video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
            pixel_values = pixel_values.to(torch.bfloat16).to(self.device)
            prompt = video_prefix + caption
            input_ids, attention_mask = prepare_chat_input(self.config, self.tokenizer, pixel_values, prompt, self.generation_config, device=self.device)

            output = self.model.forward(pixel_values, input_ids, attention_mask)
            scores.append(output.score)
            # scores.append(output.rewards.min(dim=1)[0])

        scores = torch.cat(scores, dim=0)
        return scores