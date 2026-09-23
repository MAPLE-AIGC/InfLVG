# Original code
# https://github.com/ZhaoJ9014/face.evoLVe.PyTorch/blob/master/align/face_align.py
# https://github.com/ZhaoJ9014/face.evoLVe.PyTorch/blob/master/util/extract_feature_v1.py


import cv2
import os
import decord
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from .align.align_trans import (
    get_reference_facial_points,
    warp_and_crop_face,
)
from .align.detector import detect_faces
from .backbone import Backbone
from PIL import Image
from time import time

import logging

import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)   


class ArcFace():
    def __init__(
        self,
        model_root,
        crop_size=112,
        input_size=[112, 112],
        score_mode="vbench",
        device='cuda',
    ):
        super().__init__()
        self.crop_size = crop_size
        self.input_size = input_size
        self.score_mode = score_mode

        # self.transform = transforms.Compose(
        #     [
        #         transforms.Resize(
        #             [int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)],
        #         ),  # smaller side resized
        #         transforms.CenterCrop([input_size[0], input_size[1]]),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        #     ],
        # )

        self.device = device
        # self.backbone = Backbone(input_size)
        # self.backbone.load_state_dict(torch.load(model_root, map_location=self.device))
        # self.backbone.to(device)
        # self.backbone.eval()

        try:
            ctx_id = device.index if device.index is not None else 0
        except:
            ctx_id = -1
        logger.info(f"ctx_id={ctx_id}")

        # 初始化 FaceAnalysis 对象
        # self.face_analysis = FaceAnalysis(name='buffalo_l')  # 可以选择不同的模型，如 'buffalo_l'、'antelopev2' 等
        # self.face_analysis.prepare(ctx_id=ctx_id, det_size=(640, 640))  # ctx_id=0 表示使用 GPU，ctx_id=-1 表示使用 CPU

        # self.handler = insightface.model_zoo.get_model("/storage/qiguojunLab/fangxueji/Models/insightface/antelopev2/glintr100.onnx")
        # self.handler.prepare(ctx_id=ctx_id)

        self.app = FaceAnalysis(name='antelopev2', root=os.environ["INSIGHTFACE_ROOT"], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

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


    def plot_similarity_grid(sefl, cos_similarity, input_size):
        n = len(cos_similarity)
        rows = []
        for i in range(n):
            row = []
            for j in range(n):
                # create small colorful image from value in distance matrix
                value = cos_similarity[i][j]
                cell = np.empty(input_size)
                cell.fill(value)
                cell = (cell * 255).astype(np.uint8)
                # color depends on value: blue is closer to 0, green is closer to 1
                img = cv2.applyColorMap(cell, cv2.COLORMAP_WINTER)

                # add distance value as text centered on image
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = f"{value:.2f}"
                textsize = cv2.getTextSize(text, font, 1, 2)[0]
                text_x = (img.shape[1] - textsize[0]) // 2
                text_y = (img.shape[0] + textsize[1]) // 2
                cv2.putText(
                    img, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA,
                )
                row.append(img)
            rows.append(np.concatenate(row, axis=1))
        grid = np.concatenate(rows)
        return grid


    def face_emb_old(
        self,
        frame,
    ):
        """ face align """
        scale = self.crop_size / 112.0
        reference = get_reference_facial_points(default_square=True) * scale
        _, landmarks = detect_faces(frame)
        if (len(landmarks) == 0):
            # print("non-detected landmarks!")
            # return 0.
            raise ValueError("non-detected landmarks!")
        facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
        warped_face, mask_face = warp_and_crop_face(
            np.array(frame),
            facial5points,
            reference,
            crop_size=(self.crop_size, self.crop_size),
        )
        img_warped = Image.fromarray(warped_face)
        """ face embed """
        img_trans = self.transform(img_warped).unsqueeze(0)             # 1 c h w
        embed = F.normalize(self.backbone(img_trans.to(self.device)))   # 1 512
        return embed, img_warped, mask_face
    
    def face_emb(
        self,
        frame,
    ):
        frame = np.array(frame)        # H W 3
        # face = self.face_analysis.get(frame)[0]
        # embed = self.handler.get(face=face, img=frame)
        # embed = F.normalize(torch.tensor(embed).unsqueeze(0).to(self.device))
        faces = self.app.get(frame)
        faces = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # select largest face (if more than one detected)
        id_emb = torch.tensor(faces['embedding'], dtype=torch.float16)[None].to(self.device)
        id_emb = id_emb/torch.norm(id_emb, dim=1, keepdim=True)   # normalize embedding 

        return id_emb, None, None
    
    def load_video(
        self,
        video_path,
        mode='default',
        target_num=None,
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
        elif mode=="uniform_num":
            frames_idx = np.linspace(0, num_frames-1, target_num, dtype=int)
        else:
            raise NotImplementedError
        
        frames = video_reader.get_batch(frames_idx).asnumpy()
        frames = [Image.fromarray(frame) for frame in frames]
        return frames
        
    def score_video(
        self,
        video_path=None,
        frames=None,
    ):
        # import pdb; pdb.set_trace()
        if frames is None:
            frames = self.load_video(video_path, mode="half+half")    # list
        # from diffusers.utils import make_image_grid
        # make_image_grid(frames, cols=4, rows=2).save('./tmp_grid2.png')
        embeds = []
        frames_warped = []
        bad_frames = 0
        try:
            for i, frame in enumerate(frames):
                try:
                    embed, img_warped, _ = self.face_emb(frame)
                    embeds.append(embed)
                    frames_warped.append(img_warped)
                except BaseException as e:
                    bad_frames += 1
                    logger.info(f"Error frame [{i+1}/{len(frames)}]: {e}")
            # import pdb; pdb.set_trace()
            embeds = torch.cat(embeds)
            cos_similarity = embeds @ embeds.transpose(0,1)
            # import pdb;  pdb.set_trace()
            # print(cos_similarity)

            cos_similarity = cos_similarity.clip(min=0, max=1)
            self.init_mask(cos_similarity.shape[0])
            avg_similarity = cos_similarity[self.mask].mean()
            # avg_similarity = avg_similarity.sum()/(len(avg_similarity)+bad_frames*2)


            """ vis """
            # similarity_grid = self.plot_similarity_grid(cos_similarity, self.input_size)
            # # pad similarity grid with images of faces
            # horizontal_grid = np.hstack(frames_warped)
            # vertical_grid = np.vstack(frames_warped)
            # zeros = np.zeros((*self.input_size, 3))
            # vertical_grid = np.vstack((zeros, vertical_grid))
            # result = np.vstack((horizontal_grid, similarity_grid))
            # result = np.hstack((vertical_grid, result))
            # cv2.imwrite(f"debug.jpg", result)

            return avg_similarity.cpu()
        except BaseException as e:
            logger.info(f"Caught an exception: {e}")
            return torch.tensor(0.)

    def mask_video(
        self,
        video_path,
        mode="half"
    ):
        frames = self.load_video(video_path, mode=mode)    # list
        bad_frames = 0
        mask_face_all = []
        try:
            for i, frame in enumerate(frames):
                # try:
                _, _, mask_face = self.face_emb(frame)
                mask_face_all.append(mask_face)
                # except BaseException as e:
                #     bad_frames += 1
                #     logger.info(f"Error frame [{i}/{len(frames)}]: {e}")

            mask_face_all = torch.cat(mask_face_all, dim=-1) # B 1 FHW
            return mask_face_all
        except BaseException as e:
            logger.info(f"Caught an exception: {e}")
            return None
        
    @torch.no_grad()
    def __call__(
        self,
        video_paths,
    ):
        scores = []
        for video_path in video_paths:
            scores.append(self.score_video(video_path))
        
        return torch.stack(scores)

    @torch.no_grad()
    def multiclip(
        self,
        hist_clip_path,
        hist_num,
        video_path,
    ):
        logger.info(f"--- using multi clip ---")
        hist_frames = self.load_video(hist_clip_path, mode="uniform_num", target_num=8)
        curt_frames = self.load_video(video_path, mode="uniform_num", target_num=8)
        scores = []
        scores.append(self.score_video(frames=hist_frames+curt_frames))
        
        return torch.stack(scores)
    
if __name__ == '__main__':
    arcface = ArcFace(
        model_root="/storage/qiguojunLab/qiguojun/home/Models/arcface/backbone_ir50_ms1m_epoch120.pth",
        crop_size=112,
        input_size=[112, 112],
        device='cuda:1'
    )

    video_paths = [
        "/storage/qiguojunLab/fangxueji/Projects/nips25/toy_rl/tmp/debug_0402_8+8_paral_new_all/rank_5/batch_0_step_2_inner_5.mp4"
        # "/storage/qiguojunLab/fangxueji/Projects/nips25/toy_rl/tmp/debug_0401_8+8_paral_new/rank_0/batch_0_step_0_inner_0.mp4"
        # "/storage/qiguojunLab/fangxueji/Projects/nips25/toy_rl/tmp/debug_0327_8+8_paral/rank_0/batch_0_step_0_inner_0.mp4"
        # "/storage/qiguojunLab/fangxueji/Projects/nips25/toy_rl/tmp/debug_0401_8+8_paral_new/rank_5/batch_0_step_1_inner_5.mp4",
        # "/storage/qiguojunLab/fangxueji/Projects/nips25/toy_rl/tmp/debug_0401_8+8_paral_new/rank_5/batch_0_step_10_inner_5.mp4",
        # "/storage/qiguojunLab/fangxueji/Projects/nips25/toy_rl/tmp/debug_0401_8+8_paral_new/rank_5/batch_0_step_15_inner_5.mp4"
    ]

    t1 = time()
    score = arcface(video_paths)
    t2 = time()
    print(f"{t2-t1}s, {score, score.shape}")
