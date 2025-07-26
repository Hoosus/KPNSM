import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
import OpenEXR
import Imath
import matplotlib.pyplot as plt
import cv2
import imageio

def lookat(eye, target, up):
    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0: 
            return v
        return v / norm
    mz = normalize( (eye[0]-target[0], eye[1]-target[1], eye[2]-target[2]) ) # inverse line of sight
    mx = normalize( np.cross( up, mz ) )
    my = normalize( np.cross( mz, mx ) )
    tx = -np.dot( mx, eye )
    ty = -np.dot( my, eye )
    tz = -np.dot( mz, eye )   
    return np.array([mx[0], mx[1], mx[2], tx, 
                     my[0], my[1], my[2], ty,
                     mz[0], mz[1], mz[2], tz,
                     0.0,   0.0,   0.0,   1.0], dtype=np.float32).reshape((4, 4))

def perspective(fov, aspect, near, far):
    frustumDepth = far - near
    oneOverDepth = 1 / frustumDepth

    result = np.zeros((4, 4), dtype=np.float32)
    result[1][1] = -1 / np.tan(0.5 * fov)
    result[0][0] = -result[1][1] / aspect
    result[2][2] = -far * oneOverDepth
    result[2][3] = (-far * near) * oneOverDepth
    result[3][2] = -1
    result[3][3] = 0
    return result

coefs = {
    "classroom": 10.0,
    "living-room-2": 5.4,
    "dining-room": 11.0,
    "living-room": 5.2,
    "bedroom": 5.2,
    "kitchen": 6.2,
    "staircase": 8.0,
    'living-room-3': 4.8,
    'bathroom': 4.0,
    'bathroom2': 4.0,
    'deadtree': 6.0,
    'window': 10.0
}

# for testing
coefs["classroom1"] = coefs["classroom"]
coefs["classroom2"] = coefs["classroom"]
coefs["staircase1"] = coefs["staircase"]
coefs["staircase2"] = coefs["staircase"]
coefs["living-room-31"] = coefs["living-room-3"]
coefs["bedroom1"] = coefs["bedroom"]
coefs["dining-room1"] = coefs["dining-room"]

def read_exr_channel(file_path, channel_name, scale):
    exr_file = OpenEXR.InputFile(file_path)
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    FLOAT = Imath.PixelType(Imath.PixelType.HALF)
    channel_data = exr_file.channel(channel_name, FLOAT)
    channel_data_np = np.frombuffer(channel_data, dtype=np.float16)
    channel_data_np = channel_data_np.astype(np.float32)
    channel_data_np.shape = (height, width)
    return channel_data_np / scale

def read_exr_data(folder_path, exr_file, channels, scale=1.0):
    full_exr_file = os.path.join(folder_path, f"{exr_file}.exr")
    data = [read_exr_channel(full_exr_file, ch, scale) for ch in channels]
    return np.stack(data, axis=0)

class KNSMDataset(Dataset):
    def __init__(self, base_directory, scenes, use_temporal=False, penumbra_clamp=20, use_msm=False, use_pcf=False,
                 extra_inputs=None, video_order=False, video_length=10, data_range=None, temporal_group_num=1,
                 ce_add_re=False, cv_div_d=False, penumbra_width_choice="w_sqrt", odd_only=False):
        super().__init__()
        self.base_directory = base_directory
        self.scenes = scenes
        self.folder_names = []
        for scene in self.scenes:
            scene_dir = os.path.join(base_directory, scene)
            # sort according to number
            temp = sorted([int(f) for f in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, f))])
            if data_range is not None:
                l, r = data_range
                assert l >= 0 and l < r and r <= len(temp)
                temp = temp[l:r]
            self.folder_names.extend([os.path.join(scene_dir, str(f)) for f in temp])
        self.len = len(self.folder_names) * 6
        self.odd_only = odd_only
        if odd_only:
            assert temporal_group_num == 2
            assert not video_order
            self.len = self.len // 2
        self.penumbra_clamp = penumbra_clamp
        self.use_msm = use_msm
        self.use_pcf = use_pcf
        self.extra_inputs = [] if extra_inputs is None else extra_inputs
        self.video_order = video_order # whether output according to videos' order: for one scene, output 6 video with different light size
        self.video_length = video_length
        self.use_temporal = use_temporal
        if use_temporal:
            self.temporal_group_num = temporal_group_num
            assert temporal_group_num in [2]
            assert not video_order
        self.ce_add_re = ce_add_re
        self.cv_div_d = cv_div_d
        self.penumbra_width_choice = penumbra_width_choice
        print(f"Loading Dataset with scenes {self.scenes}, in total {self.len} data")
    
    def __len__(self):
        return self.len

    def parse_run_info(self, folder_path, scale=1.0):
        run_info_path = os.path.join(folder_path, "run_info.txt")
        run_info = {}
        if os.path.exists(run_info_path):
            with open(run_info_path, 'r') as file:
                for line in file:
                    if line.startswith("Light Position:"):
                        run_info['light_position'] = np.array(list(map(float, line.split(":")[1].strip().strip('[]').split(',')))) / scale
                    elif line.startswith("Camera Position:"):
                        run_info['camera_position'] = np.array(list(map(float, line.split(":")[1].strip().strip('[]').split(',')))) / scale
                    elif line.startswith("Camera Direction:"):
                        run_info['camera_direction'] = np.array(list(map(float, line.split(":")[1].strip().strip('[]').split(','))))
        return run_info
        
    def mygetitem(self, idx, is_perturb): 
        if not self.video_order:
            folder_idx = idx // 6
            repetition_idx = idx % 6
        else:
            total_length = 6 * self.video_length
            scene_idx = idx // total_length
            repetition_idx = (idx % total_length) // self.video_length
            folder_idx = scene_idx * self.video_length + (idx % self.video_length)
        folder_path = self.folder_names[folder_idx]        
        scene_id = int(os.path.split(folder_path)[-1])
        scene_name = os.path.split(os.path.split(folder_path)[-2])[-1]

        coef = coefs[scene_name]

        normW = read_exr_data(folder_path, 'Mogwai.MyGBuffer.normW.15', ['R', 'G', 'B'])
        # normV = read_exr_data(folder_path, 'Mogwai.MyGBuffer.normV.15', ['R', 'G', 'B'])
        # normE = read_exr_data(folder_path, 'Mogwai.MyGBuffer.normE.15', ['R', 'G', 'B'])
        
        posW = read_exr_data(folder_path, 'Mogwai.MyGBuffer.posW.15', ['R', 'G', 'B'], coef)
        depth = read_exr_data(folder_path, 'Mogwai.MyGBuffer.distV.15', ['R'], coef)
        distE = read_exr_data(folder_path, 'Mogwai.MyGBuffer.distE.15', ['R'], coef)
        depthInShadowMap = read_exr_data(folder_path, 'Mogwai.MyShadowMap.color.15', ['B'], coef)
        shadowMap = read_exr_data(folder_path, 'Mogwai.MyShadowMap.color.15', ['G'])
        ce = read_exr_data(folder_path, 'Mogwai.MyGBuffer.ce.15', ['R'])
        cv = read_exr_data(folder_path, 'Mogwai.MyGBuffer.cv.15', ['R'])

        gt = read_exr_data(folder_path, 'Mogwai.AccumulatePass1.output.90' if repetition_idx < 3 else 'Mogwai.AccumulatePass2.output.90', [['R', 'G', 'B'][repetition_idx % 3]])

        if self.use_msm:
            # MSM
            if not self.use_pcf:
                msm = read_exr_data(folder_path, 'Mogwai.MyShadowMap.msm.15', ['G']) 
                msm[ce < 0] = 0.0
            # PCF
            else:
                msm = read_exr_data(folder_path, 'Mogwai.MyGBuffer.visibility.70', ['R']) 

        info = self.parse_run_info(folder_path, coef)
        info['light_radius'] = 0.0025 * repetition_idx / 2.0

        mask = distE != 0
        H, W = gt.shape[-2:]
        R = info['light_radius']

        shadowMap[distE == 0] = 1.0  # set missing pixels to visible
        if self.use_msm:
            msm[distE == 0] = 1.0

        if self.penumbra_width_choice == "w_sqrt":
            penumbra_width_original = np.zeros_like(distE)
            penumbra_width_original[mask] = np.clip(
                (distE[mask] - depthInShadowMap[mask]) * (R / (depthInShadowMap[mask] + 1e-5)) / depth[mask] * W * np.sqrt(np.abs(cv[mask]) / (np.abs(ce[mask]) + 1e-5)),
                a_min = 0, a_max = self.penumbra_clamp)
            penumbra_width = penumbra_width_original / self.penumbra_clamp # to [0, 1]
        elif self.penumbra_width_choice == "w":
            penumbra_width_original = np.zeros_like(distE)
            penumbra_width_original[mask] = np.clip(
                (distE[mask] - depthInShadowMap[mask]) * (R / (depthInShadowMap[mask] + 1e-5)) / depth[mask] * W,
                a_min = 0, a_max = self.penumbra_clamp)
            penumbra_width = penumbra_width_original / self.penumbra_clamp # to [0, 1]
        elif self.penumbra_width_choice == "R":
            penumbra_width = np.ones((1, H, W), dtype=np.float32) * repetition_idx / 5.0
        elif self.penumbra_width_choice == "zero":
            penumbra_width = np.zeros((1, H, W), dtype=np.float32)
        elif self.penumbra_width_choice == "ce+R":
            penumbra_width = (ce + np.ones((1, H, W), dtype=np.float32) * repetition_idx) / 6.0
        else:
            raise NotImplemented

        res = {
            "scene": self.scenes.index(scene_name), 
            "scene_id": scene_id,
            "repetition_id": repetition_idx,
            "id": idx,
            "posW": posW,
            "normW": normW,
            # "normV": normV,
            # "normE": normE,
            "depth": depth,
            "distE": distE,
            "depthInShadowMap": depthInShadowMap,
            "shadowMap": shadowMap,
            "ce": (ce + repetition_idx) / 6.0 if self.ce_add_re else ce,
            "cv": cv / (depth + 1e-10) if self.cv_div_d else cv,
            "info": info,
            "gt": gt,
            "mask": mask,
            "penumbra_width": penumbra_width,
        }
            
        # for exporting colored results
        if "mscolor" in self.extra_inputs:
            mscolor = read_exr_data(folder_path, 'Mogwai.AccumulatePassColor.output.90.exr' , ['R', 'G', 'B'])
            res["mscolor"] = mscolor

        if "all_msm" in self.extra_inputs:
            res["msm3_0_2"] = read_exr_data(folder_path, 'Mogwai.MyShadowMap.msm.15', ['G']) 
            res["msm9_1_0"] = read_exr_data(folder_path, 'Mogwai.MyShadowMap.msm.15', ['B']) 

        if self.use_temporal and not is_perturb:
            g = self.temporal_group_num
            group_id = folder_idx // g
            cur_id = folder_idx % g
            perturb_id = np.random.randint(g - 1)
            if perturb_id >= cur_id:
                perturb_id += 1
            perturb_folder_id = group_id * g + perturb_id
            perturb_dataset_id = perturb_folder_id * 6 + repetition_idx
            res_temporal = self.mygetitem(perturb_dataset_id, True)
            assert scene_id - res_temporal["scene_id"] - 1 + 2 * (scene_id % 2) == 0 # for now assuming is a pair, for example (1, 2), (2, 1), (10, 9)
            res["perturb_id"] = perturb_dataset_id
            for key in ["posW", "normW", "depth", "distE", "depthInShadowMap", "shadowMap", 
                         "ce", "cv", "gt", "mask", "penumbra_width"]:
                res[f"{key}_t"] = res_temporal[key]
        
        # used for motion vector calculation
        if self.use_temporal or "viewproj" in self.extra_inputs:
            view = lookat(info['camera_position'], info['camera_position'] + info['camera_direction'], np.array([0.0, 1.0, 0.0]))
            proj = perspective(fov=55.0/180.0*np.pi, aspect=2.0, near=0.1, far=1000.0)
            # print("proj = ", proj)
            res["view_proj"] = proj @ view
            if self.use_temporal and not is_perturb:
                res["view_proj_t"] = res_temporal["view_proj"]

        if self.use_msm:
            res["msm"] = msm
            if self.use_temporal and not is_perturb:
                res["msm_t"] = res_temporal["msm"]
        return res

    def __getitem__(self, idx):
        if self.odd_only:
            folder_idx = idx // 6
            repetition_idx = idx % 6
            folder_idx = folder_idx * 2 + 1
            idx = folder_idx * 6 + repetition_idx
        return self.mygetitem(idx, False)

if __name__ == "__main__":
    
# --------start penumbra width--------
    ### calculate 95 percent large
    scenes = ["classroom", "kitchen", "living-room", "staircase", "living-room-2"]
    dataset = KNSMDataset('..\\final_data', scenes, penumbra_clamp=1000, penumbra_width_choice="w_sqrt",
                          data_range=[0, 1800], use_msm=False, use_temporal=False, temporal_group_num=2)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16)

    hist = np.zeros((len(scenes), 1000), dtype=np.float32)
    for i, batch_data in tqdm(enumerate(dataloader)):
        # print(batch_data["id"], batch_data["perturb_id"])
        for k in batch_data.keys():
            if isinstance(batch_data[k], torch.Tensor):
                assert not torch.isnan(batch_data[k].min())
                # assert batch_data[k].min() > -1.05
                # assert not torch.isnan(batch_data[k].max())
                # assert batch_data[k].max() < 1.05
        scene = batch_data["scene"]
        image = batch_data["penumbra_width"] * 1000
        image_hist, _ = np.histogram(image, bins=1000, range=(0, 1000))
        hist[scene] += image_hist

    for i, scene in enumerate(scenes):
        cdf = np.cumsum(hist[i]) / np.sum(hist[i])

        threshold_index = np.searchsorted(cdf, 0.95)
        print(f"scene {scene} 95 percent corresponds to", threshold_index)
        threshold_index = np.searchsorted(cdf, 0.98)
        print(f"scene {scene} 98 percent corresponds to", threshold_index)



