import torch
from model import create_model
from loss import create_loss
from dataset import KNSMDataset
from knsm_utils import save_image, save_kernel, apply_kernel, apply_kernel_with_mask, expand_kernels, save_dilated_kernels, dilation_kernel_actual_size, reconstruct_full_kernel, visualize_and_save_difference, find_good_pixel_with_penumbra
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import argparse
import yaml
import os
from datetime import datetime
import cv2
from config_check import check_config
from pytorch_msssim import ssim

def test(args):
    print(f"config file: {args.config}")
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    run_name = config["name"]
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    check_config(run_name, config)

    filter_config = config["filter"]
    filter_size = filter_config["filter_size"]
    dilation = filter_config["dilation"]
    assert filter_size % 2 == 1
    kernel_actual_size = dilation_kernel_actual_size(filter_size, dilation)
    kernel_half_actual_size = kernel_actual_size // 2
    print(f"kernel: size={filter_size}, dilation={dilation}, actual_size={kernel_actual_size}")

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"{config.get("output_dir", "test")}/{run_name}_{timestamp}"
    os.makedirs(log_dir, exist_ok=False)

    with open(os.path.join(log_dir, "config.yaml"), 'w') as file:
        yaml.safe_dump(config, file)
    
    model = create_model(config)
    if "ckpt" in config:
        model.load_state_dict(torch.load(config["ckpt"]))
        print("loading ckpt from", config["ckpt"])
    else:
        assert False, "missing ckpt for analysis!"
    model = model.to(device)
    print("number of parameters:", sum(p.numel() for p in model.parameters()))

    dataset_config = config["dataset"]
    # dataset = KNSMDataset(dataset_config["train"], dataset_config["scenes_train"], **dataset_config["configs"])
    # dataloader = DataLoader(dataset, batch_size=config["training"]["train_batch_size"], shuffle=True, num_workers=dataset_config["train_workers"])
    val_scenes = config["test_scenes"]
    validation = KNSMDataset(config.get("test_dataset", "data/validation"), val_scenes,
                             use_temporal=False, **dataset_config["configs"])
    valDataloader = DataLoader(validation, batch_size=1, shuffle=False, num_workers=6)

    for scene in val_scenes:
        os.makedirs(os.path.join(log_dir, scene), exist_ok=False)

    gt_error_l1_scenes = [0.0] * len(val_scenes)
    gt_error_l2_scenes = [0.0] * len(val_scenes)
    gt_ssim_scenes = [0.0] * len(val_scenes)
    gt_error_cnt = [0.0] * len(val_scenes)

    output_gt = True
    concise_output = config.get("concise_output", False)

    model.eval()
    with torch.no_grad():
        for val_input_dict in valDataloader:
            print(f"get scene {val_input_dict["scene"][0]}, id {val_input_dict["scene_id"][0]}, repetition {val_input_dict["repetition_id"][0]}")
            idx = val_input_dict["scene_id"][0]
            rep_idx = val_input_dict["repetition_id"][0]
            
            ce = val_input_dict["ce"].to(device)
            cv = val_input_dict["cv"].to(device)
            depthInShadowMap = val_input_dict["depthInShadowMap"].to(device)
            distE = val_input_dict["distE"].to(device)
            # normalE = val_input_dict["normE"].to(device)
            # normalV = val_input_dict["normV"].to(device)
            depthDiff = distE - depthInShadowMap
            depthDiv = depthInShadowMap / (distE + 1e-10)
            shadowMap = val_input_dict["shadowMap"].to(device)
            gt = val_input_dict["gt"].to(device)
            mask = val_input_dict["mask"].to(device)
            penumbra_width = val_input_dict["penumbra_width"].to(device)

            B, _, H, W = shadowMap.shape
            if config["include_de_ds"]:
                if config["dataset"]["configs"]["penumbra_width_choice"] != "zero":
                    feat = torch.cat([ce, cv, distE, penumbra_width, depthDiff, depthDiv, depthInShadowMap], dim=1)
                else:
                    feat = torch.cat([ce, cv, distE, depthDiff, depthDiv, depthInShadowMap], dim=1)
            else:
                if config["dataset"]["configs"]["penumbra_width_choice"] != "zero":
                    feat = torch.cat([ce, cv, penumbra_width, depthDiff, depthDiv], dim=1)
                else:
                    feat = torch.cat([ce, cv, depthDiff, depthDiv], dim=1)
            if config["include_sm"]:
                if "msm" in val_input_dict:
                    msm = val_input_dict["msm"].to(device)
                    feat = torch.cat([feat, msm], dim=1)
                else:
                    feat = torch.cat([feat, shadowMap], dim=1)

            if filter_size > 1:
                kernel = model(feat)
                kernel = expand_kernels(kernel, filter_size, dilation)
                if "msm" in val_input_dict:
                    # shadowMap_filtered = apply_kernel(msm, kernel, filter_size, dilation)
                    shadowMap_filtered = apply_kernel_with_mask(msm, kernel, filter_size, dilation, mask)
                else:
                    # shadowMap_filtered = apply_kernel(shadowMap, kernel, filter_size, dilation)
                    shadowMap_filtered = apply_kernel_with_mask(shadowMap, kernel, filter_size, dilation, mask)
            else:
                shadowMap_filtered = torch.clamp(model(feat), 0.0, 1.0)

            scene = val_input_dict["scene"][0]
            
            shadowMap_masked = shadowMap_filtered * mask
            gt_masked = gt * mask
            gt_error_l1_scenes[scene] += F.l1_loss(shadowMap_masked, gt_masked, reduction='mean').item()
            gt_error_l2_scenes[scene] += F.mse_loss(shadowMap_masked, gt_masked, reduction='mean').item()
            gt_ssim_scenes[scene] += ssim(shadowMap_masked, gt_masked, data_range=1.0, size_average=True).item()
            gt_error_cnt[scene] += 1.0

            # mscolor = val_input_dict["mscolor"]
            # move to cpu to save image
            shadowMap_masked = shadowMap_masked.cpu()
            gt_masked = gt_masked.cpu()
            save_image(shadowMap_masked, os.path.join(log_dir, val_scenes[scene], f"filtered{scene}_{idx}_{rep_idx}.png"))            
            # if not concise_output:
            #     save_image((shadowMap_masked * mscolor).squeeze().permute((1, 2, 0)), os.path.join(log_dir, val_scenes[scene], f"filtered_color{scene}_{idx}_{rep_idx}.png"))
            #     save_image(torch.abs(shadowMap_masked - gt_masked), os.path.join(log_dir, val_scenes[scene], f"diff{scene}_{idx}_{rep_idx}.png"))

            if output_gt:
                save_image(gt_masked, os.path.join(log_dir, val_scenes[scene], f"gt{scene}_{idx}_{rep_idx}.png"))
            #     save_image((gt_masked * mscolor).squeeze().permute((1, 2, 0)), os.path.join(log_dir, val_scenes[scene], f"gtcolor{scene}_{idx}_{rep_idx}.png"))


    with open(os.path.join(log_dir, "_results.txt"), "w") as f:
        print("total gt l1", sum(gt_error_l1_scenes) / sum(gt_error_cnt), file=f)
        print("total gt l2", sum(gt_error_l2_scenes) / sum(gt_error_cnt), file=f)
        print("total gt ssim", sum(gt_ssim_scenes) / sum(gt_error_cnt), file=f)

        for scene in range(len(gt_error_cnt)):
            print("scene", scene, "name", val_scenes[scene], file=f)
            gt_l1 = gt_error_l1_scenes[scene] / gt_error_cnt[scene]
            gt_l2 = gt_error_l2_scenes[scene] / gt_error_cnt[scene]
            gt_ssim = gt_ssim_scenes[scene] / gt_error_cnt[scene]
            print(f"gt l1 = {gt_l1}, l2 = {gt_l2} ssim = {gt_ssim}", file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training configuration.")
    parser.add_argument('--config', type=str, default="config/evaluate.yaml", help='Path to the YAML configuration file.')
    args = parser.parse_args()

    test(args)
