# Standard libraries
import os
import random
import logging
from io import BytesIO

# Third-party libraries
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import decord
import huggingface_hub

# Hugging Face Transformers
from transformers import (
    CLIPModel,
    CLIPProcessor,
    CLIPImageProcessor,
    AutoTokenizer
)

# Local modules

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)   

class ImageRewardWarper():
    def __init__(
        self,
        use_rm=False,
        rm_path=None,
        rm_config=None,
        use_hps=False,
        enc_path=None,
        hps_path=None,
        use_mps=False,
        mps_path=None,
        mps_processor_path=None,
        use_clip=False,
        clip_path=None,
        use_clipflan=False,
        device='cuda',
        **kwargs,
    ):
        self.reward_models = []
        self.reward_funcs = []
        self.hps_tokenizer = None
        self.hps_preprocess_val = None
        self.enc_path = enc_path
        self.hps_path = hps_path
        self.device = device
        if use_rm:
            import ImageReward as RM
            self.reward_models.append(RM.load(name=rm_path, med_config=rm_config, device=device))
            self.reward_funcs.append(self.avg_frame_quality_rm)
        if use_hps:
            hps_model, self.hps_tokenizer, self.hps_preprocess_val = self.load_hps(device=device)
            self.reward_models.append(hps_model)
            self.reward_funcs.append(self.avg_frame_quality_hps)
        if use_mps:
            self.mps_model, self.mps_tokenize, self.mps_processor = self.load_mps(mps_path, mps_processor_path, device)
            self.mps_condition = "light, color, clarity, tone, style, ambiance, artistry, shape, face, hair, hands, limbs, structure, instance, texture, quantity, attributes, position, number, location, word, things." 
            self.reward_models.append(self.mps_model)
            self.reward_funcs.append(self.avg_frame_quality_mps)
        if use_clip:
            self.clip_model, self.clip_processor = self.load_clip(clip_path, self.device)
            self.reward_models.append(self.clip_model)
            self.reward_funcs.append(self.avg_frame_quality_clip)
        if use_clipflan:
            import t2v_metrics
            clip_flant5 = t2v_metrics.VQAScore(model='clip-flant5-xxl', device=device, cache_dir='./hf_cache') # our recommended scoring model
            self.reward_models.append(clip_flant5)
            self.reward_funcs.append(self.avg_frame_quality_clipflan)

    def load_clip(self, clip_path, device):
        clip_model = CLIPModel.from_pretrained(clip_path).to(device)
        clip_processor = CLIPProcessor.from_pretrained(clip_path)
        return clip_model, clip_processor
    
    def load_hps(self, device):
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        from hpsv2.utils import root_path
        model_dict = {}
        model, _, preprocess_val = create_model_and_transforms(
            'ViT-H-14',
            self.enc_path,
            precision='amp',
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )
        model_dict['model'] = model
        model_dict['preprocess_val'] = preprocess_val


        model = model_dict['model']
        preprocess_val = model_dict['preprocess_val']

        if not os.path.exists(root_path):
            os.makedirs(root_path)

        checkpoint = torch.load(self.hps_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        tokenizer = get_tokenizer('ViT-H-14')
        model = model.to(device)
        model.eval()
        return model, tokenizer, preprocess_val
    
    def load_mps(self, mps_path, mps_processor_path, device):
        mps_image_processor = CLIPImageProcessor.from_pretrained(mps_processor_path)
        mps_tokenizer = AutoTokenizer.from_pretrained(mps_processor_path, trust_remote_code=True)
        from .mps.clip_model import CLIPModel
        # import pdb; pdb.set_trace()
        model = CLIPModel(mps_processor_path)
        model.load_state_dict(torch.load(mps_path), strict=False)
        model.eval().to(device)

        return model, mps_tokenizer, mps_image_processor

    @torch.no_grad()
    def hps_score_image(self, hps_model, image: Image, prompt: str, tokenizer, preprocess_val, device):
        # Process the image
        image = preprocess_val(image).unsqueeze(0).to(device=device, non_blocking=True)
        # Process the prompt
        text = tokenizer([prompt]).to(device=device, non_blocking=True)
        # Calculate the HPS
        with torch.cuda.amp.autocast():
            outputs = hps_model(image, text)
            image_features, text_features = outputs["image_features"], outputs["text_features"]
            logits_per_image = image_features @ text_features.T
            hps_score = torch.diagonal(logits_per_image).cpu().numpy()
            
        return hps_score[0]


    @torch.no_grad()
    def mps_score_image(self, prompt, images):
        def _process_image(image):
            if isinstance(image, dict):
                image = image["bytes"]
            if isinstance(image, bytes):
                image = Image.open(BytesIO(image))
            if isinstance(image, str):
                image = Image.open( image )
            image = image.convert("RGB")
            pixel_values = self.mps_processor(image, return_tensors="pt")["pixel_values"]
            return pixel_values
        
        def _tokenize(caption):
            input_ids = self.mps_tokenize(
                caption,
                max_length=self.mps_tokenize.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids
            return input_ids
        

        image_inputs = torch.concatenate([_process_image(images[i]).to(self.device) for i in range(len(images))])
        text_inputs = _tokenize(prompt).to(self.device)
        condition_inputs = _tokenize(self.mps_condition).to(self.device)

        with torch.no_grad():
            mps_out = self.mps_model(text_inputs, image_inputs, condition_inputs)
            text_features = mps_out[0]
            # image_0_features = image_0_features / image_0_features.norm(dim=-1, keepdim=True)
            # image_1_features = image_1_features / image_1_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            img_feat = [self.mps_model.logit_scale.exp() * torch.diag(torch.einsum('bd,cd->bc', text_features,feat/feat.norm(dim=-1,keepdim=True))) for feat in mps_out[1]]
            img_feat = torch.tensor(img_feat).cpu()
            # image_0_scores = self.mps_model.logit_scale.exp() * torch.diag(torch.einsum('bd,cd->bc', text_features, image_0_features))
            # image_1_scores = self.mps_model.logit_scale.exp() * torch.diag(torch.einsum('bd,cd->bc', text_features, image_1_features))
            # scores = torch.stack([image_0_scores, image_1_scores], dim=-1)
            # probs = torch.softmax(scores, dim=-1)[0]

        return img_feat

    def load_video_rand(self, video_path, select_frames=8):
            video_reader = decord.VideoReader(video_path)
            video_frames = video_reader._num_frame
            video = video_reader.get_batch(random.choices(range(0, video_frames), k=select_frames)).asnumpy()
            video = [Image.fromarray(frame) for frame in video]
            return video
    
    def load_video_all(self, video_path):
            video_reader = decord.VideoReader(video_path)
            video_frames = video_reader._num_frame
            video = video_reader.get_batch(range(0, video_frames, 4)).asnumpy()
            video = [Image.fromarray(frame) for frame in video]
            return video

    def load_video_last(self, video_path):
            video_reader = decord.VideoReader(video_path)
            video_frames = video_reader._num_frame
            video = video_reader.get_batch([video_frames-1]).asnumpy()
            video = [Image.fromarray(frame) for frame in video]
            return video
    
    @torch.no_grad()
    def avg_frame_quality_hps(self, hps_model, video_paths, prompts, hps_tokenizer, hps_preprocess_val, device, *args, **kwargs):
        rewards_all = []
        for video_path, prompt in zip(video_paths, prompts):
            video = self.load_video_all(video_path)
            hps_scores = 0.
            for frame in video:
                hps_scores += self.hps_score_image(hps_model, frame, prompt, hps_tokenizer, hps_preprocess_val, device)
            rewards_all.append(hps_scores / len(video))
        
        return torch.Tensor(rewards_all)
        
    @torch.no_grad()
    def avg_frame_quality_rm(self, image_reward_model, video_paths, prompts, *args,  **kwargs):
        rewards_all = []
        for video_path, prompt in zip(video_paths, prompts):
            video = self.load_video_all(video_path)
            rewards = 0.
            for frame in video:
                # rewards = min(rewards, image_reward_model.score(prompt=prompt, image=frame))
                rewards += image_reward_model.score(prompt=prompt, image=frame)
            rewards /= len(video)
            rewards_all.append(rewards)
        
        return torch.Tensor(rewards_all)
    
    @torch.no_grad()
    def avg_frame_quality_mps(self, model, video_paths, prompts, *args,  **kwargs):
        rewards_all = []
        for video_path, prompt in zip(video_paths, prompts):
            video = self.load_video_all(video_path)
            reward_frames = self.mps_score_image(prompt=prompt, images=video)
            rewards_all.append(reward_frames.mean())
        
        return torch.Tensor(rewards_all)
    
    @torch.no_grad()
    def avg_frame_quality_clip(self, model, video_paths, prompts, *args,  **kwargs):
        rewards_all = []
        for video_path, prompt in zip(video_paths, prompts):
            video = self.load_video_all(video_path)
            inputs = self.clip_processor(text=[prompt], images=video, return_tensors="pt", padding=True).to(self.device)
            outputs = self.clip_model(**inputs)
            image_embeds = outputs['image_embeds']  
            text_embeds  = outputs['text_embeds']       
            image_embeds_norm = F.normalize(image_embeds, p=2, dim=-1) 
            text_embeds_norm  = F.normalize(text_embeds,  p=2, dim=-1)  # [1, 768]
            cosine_sims = image_embeds_norm @ text_embeds_norm.T 
            cosine_sims = cosine_sims.squeeze(1).mean()

            rewards_all.append(cosine_sims.cpu())
        return torch.Tensor(rewards_all)

    @torch.no_grad()
    def avg_frame_quality_clipflan(self, model, video_paths, prompts, *args,  **kwargs):
        rewards_all = []
        for video_path, prompt in zip(video_paths, prompts):
            video = self.load_video_all(video_path)
            score = model(images=video, texts=[prompt]).mean().cpu()
            rewards_all.append(score)

        return torch.Tensor(rewards_all)

    @torch.no_grad()
    def reward(self, video_paths, prompts, **kwargs):
        # scores = torch.Tensor([0.])
        scores = 0
        for i, (reward_func, reward_model) in enumerate(zip(self.reward_funcs, self.reward_models)):
            scores_ = reward_func(reward_model, video_paths, prompts, self.hps_tokenizer, self.hps_preprocess_val, self.device)
            logger.info(f"image reward - {i}: {scores_}")
            scores += scores_
        return scores
    
    @torch.no_grad()
    def reward_each(self, video_paths, prompts, **kwargs):
        # scores = torch.Tensor([0.])
        # assert len(self.reward_models) == 2
        scores = ()
        for i, (reward_func, reward_model) in enumerate(zip(self.reward_funcs, self.reward_models)):
            scores_ = reward_func(reward_model, video_paths, prompts, self.hps_tokenizer, self.hps_preprocess_val, self.device)
            logger.info(f"image reward - {i}: {scores_}")
            scores = scores + (scores_, )
        return scores
