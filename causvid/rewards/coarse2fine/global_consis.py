# LLM
import transformers
import torch
from qwen_vl_utils import process_vision_info
import torch
import numpy as np
import json
import gc
from decord import VideoReader, cpu

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import torch
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np

""" load instruction and prompts """
# instruc_meta = []
# with open("/storage/qiguojunLab/fangxueji/Projects/sota_llm/Qwen-VL/vgen_bench.jsonl", "r", encoding="utf-8") as f:
#     for line in f:
#         instruc_meta.append(json.loads(line.strip()))

""" VLM Warper """
class GlobalConsisWarper_OLD():
    def __init__(
        self,
        vlm_ins,
        vlm_path,
        vlm_cls="Qwen2.5-VL",
        device='cuda',
        dtype=torch.bfloat16,
        ):
        with open(vlm_ins, 'r', encoding='utf-8') as f:
            self.ins_dic = json.load(f)  

        if vlm_cls == "Qwen2.5-VL":
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            self.pipeline = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                vlm_path, attn_implementation="flash_attention_2", torch_dtype=dtype, device_map=device
            )
            self.processor = AutoProcessor.from_pretrained(vlm_path)
        else:
            raise NotImplementedError
        
        self.id_yes = self.processor(text='yes')['input_ids'][0][0]
        self.id_no = self.processor(text='no')['input_ids'][0][0]
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def score_whole_video(self, instruction_for_vlms, video_paths):
        score = []
        for video_path, instruction_for_vlm in zip(video_paths, instruction_for_vlms):
            messages = [
                {"role": "system", "content": "You should evaluate the quality of the video based on user instructions. "
                "And answer only yes if video qulity is good, or no otherwise."},
                {   
                    "role": "user",
                    "content": [
                        {
                            
                            "type": "video",
                            "video": video_path,
                            "max_pixels": 360 * 420,
                            "fps": 15.0,
                        },
                        {"type": "text", "text": instruction_for_vlm},
                    ],
                }
            ]

            # Preparation for inference
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            # Inference: Generation of the output
            # generated_ids = self.pipeline.generate(**inputs, max_new_tokens=128)
            # generated_ids_trimmed = [
            #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            # ]
            # output_text = self.processor.batch_decode(
            #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            # )
            # score.append(float(output_text[0]))
            logits = self.pipeline(**inputs)['logits']      # TODO: yes no normalization
            # prob = torch.softmax(logits[0][-1], dim=0)      # batch 1, last token
            # score.append(prob[self.id_yes].float().cpu())
            prob = torch.softmax(torch.cat([logits[0][-1][self.id_yes:self.id_yes+1], logits[0][-1][self.id_no:self.id_no+1]], dim=0), dim=-1)
            score.append(prob[0].float().cpu())
            gc.collect()
            torch.cuda.empty_cache()
        return np.array(score)


    @torch.no_grad()
    def score_sub_video(self, instruction_for_vlms, video_paths, compare_indices_list):
        score = []
        for video_path, instruction_for_vlm, compare_indices in zip(
            video_paths, instruction_for_vlms, compare_indices_list
        ):
            vr = VideoReader(video_path, ctx=cpu(0))
            compare_frames = vr.get_batch(compare_indices).asnumpy()

            # Pack frames into images for VLM (as 'images' not 'video')
            image_list = [*compare_frames]
            image_inputs = [image for image in image_list]

            messages = [
                {"role": "system", "content": "You should compare the keyframes based on the user instruction. "
                "If the video quality or behavior meets the instruction, answer yes, otherwise no."},
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img} for img in image_inputs],
                        {"type": "text", "text": instruction_for_vlm}
                    ]
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                return_tensors="pt",
                padding=True,
            )
            inputs = inputs.to(self.device)

            logits = self.pipeline(**inputs)['logits']
            prob = torch.softmax(
                torch.cat([logits[0][-1][self.id_yes:self.id_yes+1], logits[0][-1][self.id_no:self.id_no+1]], dim=0),
                dim=-1
            )
            score.append(prob[0].float().cpu())
            gc.collect()
            torch.cuda.empty_cache()
        return np.array(score)
    
    @torch.no_grad()
    def reward(self, video_paths, prompts, **kwargs):
        # ins_for_vlm = [self.ins_dic[prompt] for prompt in prompts]
        ins_for_vlm = prompts
        score = self.score_whole_video(ins_for_vlm, video_paths)
        return score
        

class GlobalConsisWarper():
    def __init__(
        self,
        device='cuda',
        dtype=torch.bfloat16,
        vlm_path=None,
        *args, 
        **kwargs,
        ):
        mode_path = vlm_path

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            mode_path, attn_implementation="sdpa", torch_dtype=torch.bfloat16, device_map=device
        )

        # default processer
        self.processor = AutoProcessor.from_pretrained(mode_path)
        

        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def vlm_score(self, model, processor, video_path):
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "you should only answer 'yes' or 'no'"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 128 * 28 * 28,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": "Are there any colorful mosaic-like artifacts in this video? answer yes or no"},
                ],
            }
        ]
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if 'yes' in output_text[0] or 'Yes' in output_text[0]:
            score = 0
        elif 'no' in output_text[0] or 'No' in output_text[0]:
            score = 1
        else:
            score = 0

        return score
    
    @torch.no_grad()
    def reward(self, video_paths, prompts, **kwargs):
        scores_all = []
        for video_path in video_paths:
            scores_all.append(self.vlm_score(self.model, self.processor, video_path))
        return np.array(scores_all)
        
