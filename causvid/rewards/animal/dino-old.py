from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import decord
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torchvision.transforms as T

class DINO():
    def __init__(self, device, score_mode="leftbottom"):
        config = {
            'repo_or_dir': f'/storage/qiguojunLab/qiguojun/.cache/torch/hub/facebookresearch_dino_main/',
            'path': f'/storage/qiguojunLab/fangxueji/Models/vbench/dino_model/dino_vitbase16_pretrain.pth', 
            'model': 'dino_vitb16',
            'source': 'local',
            }
        self.model = torch.hub.load(**config).to(device)
        self.image_transform = self.dino_transform(224)
        self.device = device
        self.score_mode = score_mode
        
    def dino_transform(self, n_px):
        return Compose([
            Resize(size=n_px, antialias=False),
            transforms.Lambda(lambda x: x.float().div(255.0)),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def load_video(
        self,
        video_path,
        mode='default',
    ):
        video_reader = decord.VideoReader(video_path)
        num_frames = video_reader._num_frame
        # import pdb; pdb.set_trace()
        # from diffusers.utils import make_image_grid
        # video = [Image.fromarray(it) for it in video_reader.get_batch(list(range(num_frames))).asnumpy() ]
        # make_image_grid(video, cols=9, rows=5).save('./tmp_grid_all.png')

        if mode == 'default':
            frames_idx = np.linspace(0, num_frames-1, (num_frames-1)//4+1, dtype=int)
        elif mode == 'half':
            frames_idx = list(range(0, (num_frames-1)//2, 4))
        elif mode == "half+half":
            half_frames = num_frames // 2
            ref_frames = list(range(3, 3+4))              # 只取4帧ref，但是抛弃开头几帧有artifacts
            evl_frames = list(range(half_frames, num_frames, half_frames//4)[-4:])    # 也只取4帧用来评估
            frames_idx = np.concatenate([ref_frames, evl_frames])
        else:
            raise NotImplementedError
        
        frames = video_reader.get_batch(frames_idx).asnumpy()
        frames = [torch.Tensor(frame).unsqueeze(0).permute(0, 3, 1, 2) for frame in frames]
        frames = torch.cat(frames, dim=0)

        return frames
    
    def init_mask(self, num_frames):
        N = num_frames # (num_frames-1)//4 + 1
        if self.score_mode == 'vbench':
            self.mask = torch.zeros((N, N), dtype=torch.bool, device=self.device)
            self.mask[:, 0] = True
            for i in range(1, N):
                self.mask[i, i-1] = True
        elif self.score_mode == "half":
            self.mask = torch.zeros((N, N), dtype=torch.bool, device=self.device)
            self.mask[:, 0] = True
            for i in range(N//2, N):
                self.mask[i, i-1] = True
        elif self.score_mode == "leftbottom":
            self.mask = torch.zeros((N, N), dtype=torch.bool, device=self.device)
            half = N // 2
            self.mask[half:, :half] = True
            self.mask[half, half-1] = False
        else:
           self.mask = ~torch.eye(N, dtype=torch.bool, device=self.device)

    @torch.no_grad()
    def reward(self, video_paths, *args, **kwargs):
        import pdb; pdb.set_trace()
        score_all = []
        for video_path in video_paths:
            images = self.load_video(video_path, mode="half+half")
            images = self.image_transform(images)
            embed_all = []
            for i in range(len(images)):
                with torch.no_grad():
                    image = images[i].unsqueeze(0)
                    image = image.to(self.device)
                    image_features = self.model(image)
                    embed_all.append(F.normalize(image_features, dim=-1, p=2))
            embed_all = torch.cat(embed_all, dim=0)

            cos_similarity = embed_all @ embed_all.transpose(0,1)
            cos_similarity = cos_similarity.clip(min=0, max=1)
            self.init_mask(cos_similarity.shape[0])
            avg_similarity = cos_similarity[self.mask].mean().cpu()
            score_all.append(avg_similarity)

        return torch.stack(score_all)
    
class DINO_NEW():
    def __init__(self, device, score_mode="leftbottom"):

        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"

        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.eval().to(device)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)

        # ---- Step 6: 加载 DINOv2 模型并提取 embedding ----
        self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
        self.dino_model.eval().to(device)

        self.device = device
        self.score_mode = score_mode

    def load_video(
        self,
        video_path,
        mode='default',
    ):
        video_reader = decord.VideoReader(video_path)
        num_frames = video_reader._num_frame

        if mode == 'default':
            frames_idx = np.linspace(0, num_frames-1, (num_frames-1)//4+1, dtype=int)
        elif mode == 'half':
            frames_idx = list(range(0, (num_frames-1)//2, 4))
        elif mode == "half+half":
            half_frames = num_frames // 2
            ref_frames = list(range(3, 3+4))              # 只取4帧ref，但是抛弃开头几帧有artifacts
            evl_frames = list(range(half_frames, num_frames, half_frames//4)[-4:])    # 也只取4帧用来评估
            frames_idx = np.concatenate([ref_frames, evl_frames])
        else:
            raise NotImplementedError
        
        frames = video_reader.get_batch(frames_idx).asnumpy()

        return frames
    
    def init_mask(self, num_frames):
        N = num_frames # (num_frames-1)//4 + 1
        if self.score_mode == 'vbench':
            self.mask = torch.zeros((N, N), dtype=torch.bool, device=self.device)
            self.mask[:, 0] = True
            for i in range(1, N):
                self.mask[i, i-1] = True
        elif self.score_mode == "half":
            self.mask = torch.zeros((N, N), dtype=torch.bool, device=self.device)
            self.mask[:, 0] = True
            for i in range(N//2, N):
                self.mask[i, i-1] = True
        elif self.score_mode == "leftbottom":
            self.mask = torch.zeros((N, N), dtype=torch.bool, device=self.device)
            half = N // 2
            self.mask[half:, :half] = True
            self.mask[half, half-1] = False
        else:
           self.mask = ~torch.eye(N, dtype=torch.bool, device=self.device)

    def subject_embed(self, image_np, i=0):

        masks = self.mask_generator.generate(image_np)

        # ---- Step 3: 选择最大的 mask（最可能是主体）----
        largest_mask = max(masks, key=lambda x: x['area'])['segmentation'].astype(np.uint8)

        # ---- Step 4: 应用掩码裁剪图像主体区域 ----
        masked_image = image_np * largest_mask[..., None]  # apply mask on RGB
        Image.fromarray(masked_image).save(f'./mask_img_{i}.png')
        # ---- Step 5: 预处理图像以匹配 DINOv2 要求 ----
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3),  # DINOv2 expects [-1, 1] input
        ])

        input_tensor = transform(masked_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.dino_model(input_tensor)  # [1, 768] for ViT-B/14

        return embedding
    
    @torch.no_grad()
    def reward(self, video_paths, *args, **kwargs):
        import pdb; pdb.set_trace()
        score_all = []
        for video_path in video_paths:
            frames = self.load_video(video_path)
            embed_all = [self.subject_embed(frame, i) for i, frame in enumerate(frames)]
            embed_all = torch.cat(embed_all, dim=0)
            cos_similarity = embed_all @ embed_all.transpose(0,1)
            cos_similarity = cos_similarity.clip(min=0, max=1)
            self.init_mask(cos_similarity.shape[0])
            avg_similarity = cos_similarity[self.mask].mean().cpu()
            score_all.append(avg_similarity)
            print(cos_similarity)

        return torch.stack(score_all)