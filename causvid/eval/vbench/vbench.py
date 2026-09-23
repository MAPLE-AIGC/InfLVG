import os
import cv2
import glob
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from omegaconf import OmegaConf
import torch.nn.functional as F
import clip

from vbench.third_party.amt.utils.utils import (
    img2tensor, tensor2img,
    check_dim_and_resize
    )
from vbench.third_party.amt.utils.build_utils import build_from_cfg
from vbench.third_party.amt.utils.utils import InputPadder
from vbench.third_party.RAFT.core.raft import RAFT
from vbench.utils import load_video, load_dimension_info, clip_transform, dino_transform, dino_transform_Image


class FrameProcess:
    def __init__(self):
        pass


    def get_frames(self, video_path):
        frame_list = []
        video = cv2.VideoCapture(video_path)
        while video.isOpened():
            success, frame = video.read()
            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to rgb
                frame_list.append(frame)
            else:
                break
        video.release()
        assert frame_list != []
        return frame_list 
    

    def get_frames_from_img_folder(self, img_folder):
        exts = ['jpg', 'png', 'jpeg', 'bmp', 'tif', 
                'tiff', 'JPG', 'PNG', 'JPEG', 'BMP', 
                'TIF', 'TIFF']
        frame_list = []
        imgs = sorted([p for p in glob.glob(os.path.join(img_folder, "*")) if os.path.splitext(p)[1][1:] in exts])
        # imgs = sorted(glob.glob(os.path.join(img_folder, "*.png")))
        for img in imgs:
            frame = cv2.imread(img, cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_list.append(frame)
        assert frame_list != []
        return frame_list


    def extract_frame(self, frame_list, start_from=0):
        extract = []
        for i in range(start_from, len(frame_list), 2):
            extract.append(frame_list[i])
        return extract


class MotionSmoothness:
    def __init__(self, 
            config="/storage/qiguojunLab/fangxueji/Projects/nips25/toy_rl/configs/AMT-S.yaml",
            ckpt="/storage/qiguojunLab/fangxueji/Models/vbench/amt_model/amt-s.pth",
            device="cuda"
        ):
        self.device = device
        self.config = config
        self.ckpt = ckpt
        self.niters = 1
        self.initialization()
        self.load_model()

    
    def load_model(self):
        cfg_path = self.config
        ckpt_path = self.ckpt
        network_cfg = OmegaConf.load(cfg_path).network
        network_name = network_cfg.name
        print(f'Loading [{network_name}] from [{ckpt_path}]...')
        self.model = build_from_cfg(network_cfg)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt['state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()


    def initialization(self):
        if self.device == 'cuda':
            self.anchor_resolution = 1024 * 512
            self.anchor_memory = 1500 * 1024**2
            self.anchor_memory_bias = 2500 * 1024**2
            self.vram_avail = torch.cuda.get_device_properties(self.device).total_memory
            print("VRAM available: {:.1f} MB".format(self.vram_avail / 1024 ** 2))
        else:
            # Do not resize in cpu mode
            self.anchor_resolution = 8192*8192
            self.anchor_memory = 1
            self.anchor_memory_bias = 0
            self.vram_avail = 1

        self.embt = torch.tensor(1/2).float().view(1, 1, 1, 1).to(self.device)
        self.fp = FrameProcess()


    def motion_score(self, video_paths, *args, **kwargs):
        scores = []
        for video_path in tqdm(video_paths):
            iters = int(self.niters)
            # get inputs
            if video_path.endswith('.mp4'):
                frames = self.fp.get_frames(video_path)
            elif os.path.isdir(video_path):
                frames = self.fp.get_frames_from_img_folder(video_path)
            else:
                raise NotImplementedError
            frame_list = self.fp.extract_frame(frames, start_from=0)
            # print(f'Loading [images] from [{video_path}], the number of images = [{len(frame_list)}]')
            inputs = [img2tensor(frame).to(self.device) for frame in frame_list]
            assert len(inputs) > 1, f"The number of input should be more than one (current {len(inputs)})"
            inputs = check_dim_and_resize(inputs)
            h, w = inputs[0].shape[-2:]
            scale = self.anchor_resolution / (h * w) * np.sqrt((self.vram_avail - self.anchor_memory_bias) / self.anchor_memory)
            scale = 1 if scale > 1 else scale
            scale = 1 / np.floor(1 / np.sqrt(scale) * 16) * 16
            if scale < 1:
                print(f"Due to the limited VRAM, the video will be scaled by {scale:.2f}")
            padding = int(16 / scale)
            padder = InputPadder(inputs[0].shape, padding)
            inputs = padder.pad(*inputs)

            # -----------------------  Interpolater ----------------------- 
            # print(f'Start frame interpolation:')
            for i in range(iters):
                # print(f'Iter {i+1}. input_frames={len(inputs)} output_frames={2*len(inputs)-1}')
                outputs = [inputs[0]]
                for in_0, in_1 in zip(inputs[:-1], inputs[1:]):
                    in_0 = in_0.to(self.device)
                    in_1 = in_1.to(self.device)
                    with torch.no_grad():
                        imgt_pred = self.model(in_0, in_1, self.embt, scale_factor=scale, eval=True)['imgt_pred']
                    outputs += [imgt_pred.cpu(), in_1.cpu()]
                inputs = outputs

            # -----------------------  cal_vfi_score ----------------------- 
            outputs = padder.unpad(*outputs)
            outputs = [tensor2img(out) for out in outputs]
            vfi_score = self.vfi_score(frames, outputs)
            norm = (255.0 - vfi_score)/255.0
            scores.append(norm.mean())
        return scores


    def vfi_score(self, ori_frames, interpolate_frames):
        ori = self.fp.extract_frame(ori_frames, start_from=1)
        interpolate = self.fp.extract_frame(interpolate_frames, start_from=1)
        scores = []
        for i in range(len(interpolate)):
            scores.append(self.get_diff(ori[i], interpolate[i]))
        return np.array(scores) # np.mean(np.array(scores))


    def get_diff(self, img1, img2):
        img = cv2.absdiff(img1, img2)
        return np.mean(img)


class DynamicDegree:
    def __init__(self, model_path, device):
        self.device = device
        args_raft= edict({"small":False, "mixed_precision":False, "alternate_corr":False})
        self.model = RAFT(args_raft)

        ckpt = torch.load(model_path, map_location="cpu")
        new_ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        self.model.load_state_dict(new_ckpt)
        self.model.to(self.device)
        self.model.eval()


    def get_score(self, img, flo):
        img = img[0].permute(1,2,0).cpu().numpy()
        flo = flo[0].permute(1,2,0).cpu().numpy()

        u = flo[:,:,0]
        v = flo[:,:,1]
        rad = np.sqrt(np.square(u) + np.square(v))
        
        h, w = rad.shape
        rad_flat = rad.flatten()
        cut_index = int(h*w*0.05)

        max_rad = np.mean(abs(np.sort(-rad_flat))[:cut_index])

        return max_rad.item()


    def set_params(self, frame, count):
        scale = min(list(frame.shape)[-2:])
        self.params = {"thres":6.0*(scale/256.0), "count_num":round(4*(count/16.0))}


    def infer(self, video_path):
        with torch.no_grad():
            if video_path.endswith('.mp4'):
                frames = self.get_frames(video_path)
            elif os.path.isdir(video_path):
                frames = self.get_frames_from_img_folder(video_path)
            else:
                raise NotImplementedError
            self.set_params(frame=frames[0], count=len(frames))
            static_score = []
            for image1, image2 in zip(frames[:-1], frames[1:]):
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                _, flow_up = self.model(image1, image2, iters=20, test_mode=True)
                max_rad = self.get_score(image1, flow_up)
                static_score.append(max_rad)
            # whether_move = self.check_move(static_score)
            return np.array(static_score)


    def check_move(self, score_list):
        thres = self.params["thres"]
        count_num = self.params["count_num"]
        count = 0
        for score in score_list:
            if score > thres:
                count += 1
            if count >= count_num:
                return True
        return False


    def get_frames(self, video_path):
        frame_list = []
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS) # get fps
        interval = max(1, round(fps / 8))
        while video.isOpened():
            success, frame = video.read()
            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to rgb
                frame = torch.from_numpy(frame.astype(np.uint8)).permute(2, 0, 1).float()
                frame = frame[None].to(self.device)
                frame_list.append(frame)
            else:
                break
        video.release()
        assert frame_list != []
        frame_list = self.extract_frame(frame_list, interval)
        return frame_list 
    
    
    def extract_frame(self, frame_list, interval=1):
        extract = []
        for i in range(0, len(frame_list), interval):
            extract.append(frame_list[i])
        return extract


    def get_frames_from_img_folder(self, img_folder):
        exts = ['jpg', 'png', 'jpeg', 'bmp', 'tif', 
        'tiff', 'JPG', 'PNG', 'JPEG', 'BMP', 
        'TIF', 'TIFF']
        frame_list = []
        imgs = sorted([p for p in glob.glob(os.path.join(img_folder, "*")) if os.path.splitext(p)[1][1:] in exts])
        # imgs = sorted(glob.glob(os.path.join(img_folder, "*.png")))
        for img in imgs:
            frame = cv2.imread(img, cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame.astype(np.uint8)).permute(2, 0, 1).float()
            frame = frame[None].to(self.device)
            frame_list.append(frame)
        assert frame_list != []
        return frame_list

def dynamic_degree(dynamic, video_list):
    sim = []
    video_results = []
    for video_path in tqdm(video_list, disable=True):
        score_per_video = dynamic.infer(video_path)
        video_results.append({'video_path': video_path, 'video_results': score_per_video})
        sim.append(score_per_video)
    avg_score = np.mean(sim)
    return avg_score, video_results

class BackgroundConsistency:
    def __init__(self, vit_path="/storage/qiguojunLab/fangxueji/Models/vbench/clip/ViT-B-32.pt",device='cuda'):
        super().__init__()

        self.clip_model, _ = clip.load(vit_path, device=device)

    def background_consistency(self, video_list, device):
        sim = 0.0
        cnt = 0
        video_results = []
        image_transform = clip_transform(224)
        for video_path in tqdm(video_list, disable=False):
            video_sim = 0.0
            cnt_per_video = 0

            images = load_video(video_path)
            images = image_transform(images)

            images = images.to(device)
            image_features = self.clip_model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1, p=2)
            for i in range(len(image_features)):
                image_feature = image_features[i].unsqueeze(0)
                if i == 0:
                    first_image_feature = image_feature
                else:
                    sim_pre = max(0.0, F.cosine_similarity(former_image_feature, image_feature).item())
                    sim_fir = max(0.0, F.cosine_similarity(first_image_feature, image_feature).item())
                    cur_sim = (sim_pre + sim_fir) / 2
                    video_sim += cur_sim
                    cnt += 1
                    cnt_per_video += 1
                former_image_feature = image_feature
            sim_per_image = video_sim / (len(image_features) - 1)
            sim += video_sim
            video_results.append(sim_per_image)
            # video_results.append({
            #     'video_path': video_path, 
            #     'video_results': sim_per_image,
            #     'video_sim': video_sim,
            #     'cnt_per_video': cnt_per_video})
        # sim_per_video = sim / (len(video_list) - 1)
        # sim_per_frame = sim / cnt
        return video_results # sim_per_frame # , video_results
    
class SubjectConsistency:
    def __init__(self, 
                 device="cuda"):
        super().__init__()

        config = {
            'repo_or_dir': f'/storage/qiguojunLab/qiguojun/.cache/torch/hub/facebookresearch_dino_main/',
            'path': f'/storage/qiguojunLab/fangxueji/Models/vbench/dino_model/dino_vitbase16_pretrain.pth', 
            'model': 'dino_vitb16',
            'source': 'local',
            }
        self.model = torch.hub.load(**config).to(device)
    
    def subject_consistency(self, video_list, device):
        sim = 0.0
        cnt = 0
        video_results = []
        image_transform = dino_transform(224)
        for video_path in tqdm(video_list, disable=False):
            video_sim = 0.0
            images = load_video(video_path)
            images = image_transform(images)
            for i in range(len(images)):
                with torch.no_grad():
                    image = images[i].unsqueeze(0)
                    image = image.to(device)
                    image_features = self.model(image)
                    image_features = F.normalize(image_features, dim=-1, p=2)
                    if i == 0:
                        first_image_features = image_features
                    else:
                        sim_pre = max(0.0, F.cosine_similarity(former_image_features, image_features).item())
                        sim_fir = max(0.0, F.cosine_similarity(first_image_features, image_features).item())
                        cur_sim = (sim_pre + sim_fir) / 2
                        video_sim += cur_sim
                        cnt += 1
                former_image_features = image_features
            sim_per_images = video_sim / (len(images) - 1)
            sim += video_sim
            # video_results.append({'video_path': video_path, 'video_results': sim_per_images})
            video_results.append(sim_per_images)
        # sim_per_video = sim / (len(video_list) - 1)
        # sim_per_frame = sim / cnt
        return video_results #  sim_per_frame, video_results

class TemporalFlickering:
    def __init__(self):
        super().__init__()


    def get_frames(self, video_path):
        frames = []
        video = cv2.VideoCapture(video_path)
        while video.isOpened():
            success, frame = video.read()
            if success:
                frames.append(frame)
            else:
                break
        video.release()
        assert frames != []
        return frames


    def mae_seq(self, frames):
        ssds = []
        for i in range(len(frames)-1):
            ssds.append(self.calculate_mae(frames[i], frames[i+1]))
        return np.array(ssds)


    def calculate_mae(self, img1, img2):
        """Computing the mean absolute error (MAE) between two images."""
        if img1.shape != img2.shape:
            print("Images don't have the same shape.")
            return
        return np.mean(cv2.absdiff(np.array(img1, dtype=np.float32), np.array(img2, dtype=np.float32)))


    def cal_score(self, video_path):
        """please ensure the video is static"""
        frames = self.get_frames(video_path)
        score_seq = self.mae_seq(frames)
        return (255.0 - np.mean(score_seq).item())/255.0

    def temporal_flickering(self, video_list, *args, **kwargs):
        sim = []
        video_results = []
        for video_path in tqdm(video_list, disable=False):
            # try:
            score_per_video = self.cal_score(video_path)
            # except AssertionError:
            #     continue
            video_results.append({'video_path': video_path, 'video_results': score_per_video})
            sim.append(score_per_video)
        # avg_score = np.mean(sim)
        return sim # , video_results
    
class VBenchMy:
    def __init__(self, device):
        self.device = device

        # ----- 构造各子模型实例 -----
        self.background = BackgroundConsistency(device=device)
        self.subject    = SubjectConsistency(device=device)
        self.flicker    = TemporalFlickering()
        self.motion     = MotionSmoothness(device=device)

        # 将它们的批量接口都注册进来
        # 注意 motion_score 只能处理单个 video_path，所以我们在 score 中 special case
        self.vbench_models = [
            self.background,
            self.subject,
            self.flicker,
            self.motion,
        ]
        self.vbench_funs = [
            self.background.background_consistency,
            self.subject.subject_consistency,
            self.flicker.temporal_flickering,
            self.motion.motion_score,
        ]

    def score(self, video_list):
        """
        对于除了 MotionSmoothness 之外的模型，都调用 func(video_list, device)
        对于 MotionSmoothness，则对列表里每个视频单独调用 motion_score，再取平均。
        返回：四元组 [bg_consistency, subject_consistency, flicker, motion_smoothness]
        """
        results = []
        for model, func in zip(self.vbench_models, self.vbench_funs):
            # if isinstance(model, MotionSmoothness):
            #     # motion_score 只能接受单个 path
            #     scores = [ model.motion_score(vp) for vp in video_list ]
            #     results.append( float(np.mean(scores)) )
            # else:
            # 其它都有 (video_list, device) 签名
            results.append( func(video_list, self.device) )
        return results
