import os
import cv2
import glob
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from omegaconf import OmegaConf

from vbench.third_party.amt.utils.utils import (
    img2tensor, tensor2img,
    check_dim_and_resize
    )
from vbench.third_party.amt.utils.build_utils import build_from_cfg
from vbench.third_party.amt.utils.utils import InputPadder
from vbench.third_party.RAFT.core.raft import RAFT

from .image_reward import ImageRewardWarper

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
    def __init__(self, config, ckpt, device):
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


    def motion_score(self, video_path):
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
        return norm


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

class LocalConsisWarper():
    def __init__(
        self,
        ms_config,
        ms_path,
        dd_path,
        device,
        use_rm,
        use_hps,
        rm_path,
        rm_config,
        enc_path,
        hps_path,
        w1=1,
        w2=1,
        **kwargs,
    ):
        self.ms = MotionSmoothness(
            config=ms_config, ckpt=ms_path, device=device
        )

        self.dd = DynamicDegree(
            model_path=dd_path,
            device=device
        )

        self.w1 = w1
        self.w2 = w2

        self.image_reward_model = ImageRewardWarper(
            use_rm=use_rm,
            rm_path=rm_path,
            rm_config=rm_config,
            use_hps=use_hps,
            enc_path=enc_path,
            hps_path=hps_path,
            device=device,
        )
    
    @torch.no_grad()
    def reward(self, video_paths, prompts, **kwargs):
        motion_score = []
        for video_path in video_paths:
            dynamicness = self.dd.infer(video_path)
            smoothness = self.ms.motion_score(video_path)
            motion_score.append(self.w1*dynamicness.mean() - self.w2*smoothness.mean())
            # reward.append(- self.w2*smoothness.mean())
            # print(f"dynamicness: {dynamicness.mean()}, smoothness: {smoothness.mean()}")
        
        quality_score = self.image_reward_model.reward(video_paths, prompts)
        total_score = torch.Tensor(motion_score) + quality_score

        return total_score