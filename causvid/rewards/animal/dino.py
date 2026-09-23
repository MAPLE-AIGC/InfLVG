import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F

try:
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
    from groundingdino.util.inference import annotate, load_image, predict
    import groundingdino.datasets.transforms as T
except ImportError:
    print("请安装GroundingDINO: pip install groundingdino-py")

class DINO():
    def __init__(self, device, model_config_path: str = None, model_checkpoint_path: str = None):
        """
        初始化GroundingDINO模型
        
        Args:
            model_config_path: 模型配置文件路径
            model_checkpoint_path: 模型权重文件路径
        """
        # 如果没有提供路径，使用默认配置
        if model_config_path is None:
            model_config_path = "/storage/qiguojunLab/fangxueji/Projects/nips25/InfLVGen/causvid/rewards/animal/GroundingDINO_SwinT_OGC.py"
        if model_checkpoint_path is None:
            model_checkpoint_path = "/storage/qiguojunLab/fangxueji/Projects/nips25/InfLVGen/causvid/rewards/animal/groundingdino_swint_ogc.pth"
            
        self.device = device
        
        # 加载模型
        args = SLConfig.fromfile(model_config_path)
        args.device = self.device
        self.model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = self.model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # 图像预处理
        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.dino_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        

        self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.dino_model = self.dino_model.to(self.device)
        self.dino_model.eval()
        self.score_mode = "leftbottom"

    def extract_frames(self, video_path: str, max_frames: int = 100) -> List[np.ndarray]:
        """
        从视频中提取帧
        
        Args:
            video_path: 视频文件路径
            max_frames: 最大提取帧数
            
        Returns:
            帧列表
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // max_frames)
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % step == 0:
                # 转换BGR到RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
            frame_idx += 1
            
        cap.release()
        return frames
    
    def get_object_embedding(self, image: np.ndarray, caption: str, 
                           box_threshold: float = 0.35, text_threshold: float = 0.25) -> torch.Tensor:
        """
        使用GroundingDINO获取物体的embedding
        
        Args:
            image: 输入图像 (numpy array)
            caption: 物体描述文本
            box_threshold: 检测框阈值
            text_threshold: 文本阈值
            
        Returns:
            物体的embedding特征向量
        """
        # 转换为PIL图像
        pil_image = Image.fromarray(image)
        original_w, original_h = pil_image.size
        # 预处理图像
        image_tensor, _ = self.transform(pil_image, None)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            # outputs = self.model(image_tensor, captions=[caption])
            boxes, logits, phrases = predict(
                model=self.model,
                image=image_tensor.squeeze(0),
                caption=caption,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=self.device
            )

        if len(boxes) == 0:
            print(f"警告: 未检测到物体 '{caption}'")
            return self._get_full_image_embedding(pil_image)
        

        best_idx = logits.argmax()
        best_box = boxes[best_idx]  # 格式: [cx, cy, w, h] 归一化坐标
        best_confidence = logits[best_idx].max()

        cx, cy, w, h = best_box

        x1 = int((cx - w/2) * original_w)
        y1 = int((cy - h/2) * original_h)
        x2 = int((cx + w/2) * original_w)
        y2 = int((cy + h/2) * original_h)
        

        x1 = max(0, min(x1, original_w))
        y1 = max(0, min(y1, original_h))
        x2 = max(0, min(x2, original_w))
        y2 = max(0, min(y2, original_h))
        

        cropped_image = pil_image.crop((x1, y1, x2, y2))
        
        if cropped_image.size[0] == 0 or cropped_image.size[1] == 0:
            return self._get_full_image_embedding(pil_image)
        
        return self._get_dino_embedding(cropped_image)
        
    @torch.no_grad()
    def _get_dino_embedding(self, pil_image: Image.Image) -> torch.Tensor:
        try:
            image_tensor = self.dino_transform(pil_image).unsqueeze(0).to(self.device)
            
            features = self.dino_model(image_tensor) 
            
            return features.squeeze(0).cpu()
            
        except Exception as e:
            print(f"DINO特征提取失败: {str(e)}")
            return torch.zeros(768)

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
    def reward(self, video_paths: List[str], captions: List[str], 
                                 max_frames: int = 99999, *args, **kwargs) -> List[List[float]]:
        all_similarities = []
        
        for video_path, caption in zip(video_paths, captions):
            object_name = " ".join(caption.split()[:3])
            
            frames = self.extract_frames(video_path, max_frames)
            if len(frames) < 2:
                print(f"视频 {video_path} 帧数不足，跳过")
                all_similarities.append([])
                continue
            
            # 获取每一帧的embedding
            embeddings = []
            for i, frame in enumerate(frames):
                embedding = self.get_object_embedding(frame, object_name).unsqueeze(0)
                embeddings.append(F.normalize(embedding, dim=-1, p=2))
            
            embeddings = torch.cat(embeddings, dim=0).cuda()
            cos_similarity = embeddings @ embeddings.T

            cos_similarity = cos_similarity.clip(min=0, max=1).cpu()
            self.init_mask(cos_similarity.shape[0])
            avg_similarity = cos_similarity[self.mask.cpu()].mean().cpu()
            all_similarities.append(avg_similarity)

        return torch.stack(all_similarities)
