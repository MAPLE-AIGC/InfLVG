import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)   

class VideoRewardWarp():
    def __init__(
        self,
        load_from_pretrained,
        use_rm,
        use_hps,
        use_face,
        rm_config,
        rm_path,
        enc_path,
        hps_path,
        face_path,
        use_mps,
        mps_path,
        mps_processor_path,
        use_vlm,
        vlm_ins,
        vlm_path,
        device,
        dtype,
        use_video_reward_model,
        use_clip,
        clip_path,
        face_mode,
        use_clipflan,
        use_dino=None,
    ):
        self.use_video_reward_model = use_video_reward_model

        if use_video_reward_model:
            from .video_align.model import VideoVLMRewardInference
        self.video_reward_model = VideoVLMRewardInference(
            load_from_pretrained=load_from_pretrained,
            device=device, dtype=dtype
        ) if use_video_reward_model else None

        if use_rm or use_hps or use_mps or use_clip or use_clipflan:
            from .coarse2fine.image_reward import ImageRewardWarper
        self.image_reward_model = ImageRewardWarper(
            use_rm=use_rm,
            rm_path=rm_path,
            rm_config=rm_config,
            use_hps=use_hps,
            enc_path=enc_path,
            hps_path=hps_path,
            use_mps=use_mps,
            mps_path=mps_path,
            mps_processor_path=mps_processor_path,
            use_clip=use_clip,
            clip_path=clip_path,
            use_clipflan=use_clipflan,
            device=device,
        ) if use_rm or use_hps or use_mps or use_clip or use_clipflan else None
        
        if use_face:
            from .arcface.arcface import ArcFace
        self.arcface = ArcFace(
            model_root=face_path,
            score_mode=face_mode,
            device=device
        ) if use_face else None

        if use_dino:
            from .animal.dino import DINO
        self.dino = DINO(device=device) if use_dino else None

        if use_vlm:
            from .coarse2fine.global_consis import GlobalConsisWarper
        self.vlm_reward_model = GlobalConsisWarper(
            vlm_ins=vlm_ins,
            vlm_path=vlm_path,
            device=device,
            dtype=dtype
        ) if use_vlm else None

        logger.info(f'video reward model:{self.use_video_reward_model}, use_rm:{use_rm}, use_hps:{use_hps}, use_mps: {use_mps}')
    
    @torch.no_grad()
    def reward(self, video_paths, prompts, **kwargs):
        video_scores = self.video_reward_model.reward(video_paths, prompts) if self.use_video_reward_model else 0.
        image_score = self.image_reward_model.reward(video_paths, prompts) if self.image_reward_model else 0.
        face_score = self.arcface(video_paths) if self.arcface else 0.
        
        return video_scores + image_score + face_score
    
    @torch.no_grad()
    def reward_pillist(self, video_pil_list, prompts, **kwargs):
        video_scores = self.video_reward_model.reward_latents(video_pil_list, prompts)
        return video_scores
