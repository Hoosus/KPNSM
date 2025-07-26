import torch
from knsm_utils import save_image, save_kernel, apply_kernel, expand_kernels, save_dilated_kernels, dilation_kernel_actual_size, reconstruct_full_kernel, visualize_and_save_difference, find_good_pixel_with_penumbra
from temporal import depth_tolerance, normal_tolerance
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import time
import copy
from pytorch_msssim import ssim

def evaluate(step, orig_model, config, val_log_dir, logger, valDataloader, device, half=False, prefix=""):
    os.makedirs(val_log_dir, exist_ok=False)
    print(f"evaluating! precision={"half" if half else "float"}")
    dataset_config = config["dataset"]
    filter_config = config["filter"]
    val_use_temporal = dataset_config.get("val_use_temporal", False)
    filter_size = filter_config["filter_size"]
    dilation = filter_config["dilation"]
    kernel_actual_size = dilation_kernel_actual_size(filter_size, dilation)
    kernel_half_actual_size = kernel_actual_size // 2

    if half:
        orig_model.to("cpu")
        model = copy.deepcopy(orig_model).half().to(device)
    else:
        model = orig_model
    model.eval()

    total_inference_time = 0
    total_inference_cnt = 0
    total_image_cnt = 0
    val_loss_scene_cnt = np.zeros(len(dataset_config["scenes_val"]), dtype=np.float32)
    val_mse_scene_sum = np.zeros(len(dataset_config["scenes_val"]), dtype=np.float32)
    val_l1_scene_sum = np.zeros(len(dataset_config["scenes_val"]), dtype=np.float32)
    val_ssim_scene_sum = np.zeros(len(dataset_config["scenes_val"]), dtype=np.float32)
    val_temporal_l1_scene_sum = np.zeros(len(dataset_config["scenes_val"]), dtype=np.float32)
    val_temporal_l2_scene_sum = np.zeros(len(dataset_config["scenes_val"]), dtype=np.float32)
    val_temporal_e3_scene_sum = np.zeros(len(dataset_config["scenes_val"]), dtype=np.float32)

    validation_cnt = 0
    total_mse = 0.0
    total_l1 = 0.0
    total_ssim = 0.0
    total_temporal_l1 = 0.0
    total_temporal_l2 = 0.0
    total_temporal_e3 = 0.0

    with torch.no_grad():
        for val_input_dict in valDataloader:
            start_time = time.time()
            # ---- Temporal Start ----
            if val_use_temporal:
                ce = val_input_dict["ce_t"]
                cv = val_input_dict["cv_t"]
                depthInShadowMap = val_input_dict["depthInShadowMap_t"]
                distE = val_input_dict["distE_t"]
                # normalE = input_dict["normE"]
                # normalV = input_dict["normV"]
                depthDiff = (distE - depthInShadowMap)
                depthDiv = (depthInShadowMap / (distE + 1e-10))
                # gt_prev = val_input_dict["gt_t"].to(device)
                # mask = val_input_dict["mask_t"].to(device)
                penumbra_width = val_input_dict["penumbra_width_t"]
                if config["include_de_ds"]:
                    if config["dataset"]["configs"]["penumbra_width_choice"] != "zero":
                        feat = [ce, cv, distE, penumbra_width, depthDiff, depthDiv, depthInShadowMap]
                    else:
                        feat = [ce, cv, distE, depthDiff, depthDiv, depthInShadowMap]
                else:
                    if config["dataset"]["configs"]["penumbra_width_choice"] != "zero":
                        feat = [ce, cv, penumbra_width, depthDiff, depthDiv]
                    else:
                        feat = [ce, cv, depthDiff, depthDiv]
                if config["include_sm"]:
                    if "msm" in val_input_dict:
                        msm = val_input_dict["msm_t"]
                        feat.append(msm)
                    else:
                        shadowMap = val_input_dict["shadowMap_t"]
                        feat.append(shadowMap)
                else:
                    shadowMap = val_input_dict["shadowMap_t"]
                feat = torch.cat(feat, dim=1).to(device)
                if half:
                    feat = feat.half()

                if filter_size > 1:
                    kernel = model(feat).to(torch.float32)
                    kernel = expand_kernels(kernel, filter_size, dilation)
                    if "msm" in val_input_dict:
                        shadowMap_filtered_prev = apply_kernel(msm.to(device), kernel, filter_size, dilation)
                    else:
                        shadowMap_filtered_prev = apply_kernel(shadowMap.to(device), kernel, filter_size, dilation)
                else:
                    shadowMap_filtered_prev = model(feat).to(torch.float32)     
                view_proj_prev = val_input_dict["view_proj_t"].to(device) 
                normalW_prev = val_input_dict["normW_t"].to(device)
                depth_prev = val_input_dict["depth_t"].to(device)

            # ---- Temporal End ----

            ce = val_input_dict["ce"]
            cv = val_input_dict["cv"]
            depthInShadowMap = val_input_dict["depthInShadowMap"]
            distE = val_input_dict["distE"]
            # normalE = val_input_dict["normE"].to(device)
            # normalV = val_input_dict["normV"].to(device)
            depthDiff = distE - depthInShadowMap
            depthDiv = depthInShadowMap / (distE + 1e-10)
            gt = val_input_dict["gt"].to(device)
            mask = val_input_dict["mask"].to(device)
            penumbra_width = val_input_dict["penumbra_width"]

            B, _, H, W = ce.shape
            if config["include_de_ds"]:
                if config["dataset"]["configs"]["penumbra_width_choice"] != "zero":
                    feat = [ce, cv, distE, penumbra_width, depthDiff, depthDiv, depthInShadowMap]
                else:
                    feat = [ce, cv, distE, depthDiff, depthDiv, depthInShadowMap]
            else:
                if config["dataset"]["configs"]["penumbra_width_choice"] != "zero":
                    feat = [ce, cv, penumbra_width, depthDiff, depthDiv]
                else:
                    feat = [ce, cv, depthDiff, depthDiv]
            if config["include_sm"]:
                if "msm" in val_input_dict:
                    msm = val_input_dict["msm"]
                    feat.append(msm)
                else:
                    shadowMap = val_input_dict["shadowMap"]
                    feat.append(shadowMap)
            else:
                shadowMap = val_input_dict["shadowMap"]
            feat = torch.cat(feat, dim=1).to(device)
            
            if half:
                feat = feat.half()
            
            if filter_size > 1:
                # if validation_cnt == 0:
                #     save_path = os.path.join(val_log_dir, f"model{step}.onnx")
                #     torch.onnx.export(model, feat[0:1], save_path, verbose=False) # save a single batch onnx model
                kernel = model(feat)
                kernel = expand_kernels(kernel, filter_size, dilation)
                if "msm" in val_input_dict:
                    shadowMap_filtered = apply_kernel(msm.to(device), kernel, filter_size, dilation)
                else:
                    shadowMap_filtered = apply_kernel(shadowMap.to(device), kernel, filter_size, dilation)
            else:
                shadowMap_filtered = torch.clamp(model(feat), 0.0, 1.0)
            shadowMap_filtered = shadowMap_filtered.to(torch.float32) # back to float32 for loss calculation

            if val_use_temporal:
                pos = val_input_dict["posW"].to(device) # (B, 3, H, W)
                pos = torch.cat([pos, torch.ones_like(pos[:, :1])], dim=1) # (B, 4, H, W)

                last_ss_pos = torch.einsum('bij,bjhw->bihw', view_proj_prev, pos) # (B, 4, H, W)
                last_ss_pos = last_ss_pos[:, :2] / last_ss_pos[:, 3:4] # (B, 2, H, W)
                last_ss_pos = last_ss_pos / 2.0 + 0.5
                last_i, last_j = last_ss_pos[:, 1] * H, last_ss_pos[:, 0] * W # (B, H, W)
                last_i = torch.floor(last_i).to(torch.int32)
                last_j = torch.floor(last_j).to(torch.int32)
                # print(last_i)
                # print(last_j)
                valid = ((last_i >= 0) & (last_i < H) & (last_j >= 0) & (last_j < W)).unsqueeze(1) & mask # (B, 1, H, W)
                last_i = torch.clamp(last_i, 0, H-1)
                last_j = torch.clamp(last_j, 0, W-1)
                b = torch.arange(B).view(B, 1, 1).expand(B, H, W).to(device)
                shadowMap_filtered_projected = torch.permute(shadowMap_filtered_prev[b, :, last_i, last_j], (0, 3, 1, 2))
                # gt_projected = torch.permute(gt_prev[b, :, last_i, last_j], (0, 3, 1, 2)) # (B, 1, H, W)
                depth_projected = torch.permute(depth_prev[b, :, last_i, last_j], (0, 3, 1, 2)) # (B, 1, H, W)
                normal_projected = torch.permute(normalW_prev[b, :, last_i, last_j], (0, 3, 1, 2)) # (B, 3, H, W)

                normal = val_input_dict["normW"].to(device)
                depth = val_input_dict["depth"].to(device)

                combined_mask = mask & valid & (torch.abs(depth - depth_projected) < depth_tolerance) & (torch.norm(normal - normal_projected, dim=1, p=2, keepdim=True) < normal_tolerance)

            shadowMap_filtered_masked = shadowMap_filtered * mask
            gt_masked = gt * mask
            mse_loss = torch.mean((shadowMap_filtered_masked - gt_masked) ** 2, dim=[1, 2, 3]) # (B,)
            total_mse += mse_loss.mean().item()
            l1_loss = torch.mean(torch.abs(shadowMap_filtered_masked - gt_masked), dim=[1, 2, 3]) # (B,)
            total_l1 += l1_loss.mean().item()
            val_ssim = ssim(shadowMap_filtered_masked, gt_masked, data_range=1.0, size_average=False) # (B,)
            total_ssim += val_ssim.mean().item()

            if val_use_temporal:
                projected_diff_masked = (shadowMap_filtered_projected - shadowMap_filtered) * combined_mask
                temporal_mse = torch.mean(projected_diff_masked ** 2, dim=[1, 2, 3]) # (B,)
                total_temporal_l2 += temporal_mse.mean().item()
                temporal_l1 = torch.mean(torch.abs(projected_diff_masked), dim=[1, 2, 3]) # (B,)
                total_temporal_l1 += temporal_l1.mean().item()
                temporal_e3 = torch.mean((torch.exp(3 * torch.abs(shadowMap_filtered_projected - shadowMap_filtered)) - 1.0) * combined_mask, dim=[1, 2, 3]) # (B,)
                total_temporal_e3 += temporal_e3.mean().item()

            validation_cnt += 1

            for t in range(shadowMap_filtered.shape[0]):
                scene = val_input_dict["scene"][t]
                val_mse_scene_sum[scene] += mse_loss[t].item()
                val_l1_scene_sum[scene] += l1_loss[t].item()
                val_ssim_scene_sum[scene] += val_ssim[t].item()
                if val_use_temporal:
                    val_temporal_l2_scene_sum[scene] += temporal_mse[t].item()
                    val_temporal_l1_scene_sum[scene] += temporal_l1[t].item()
                    val_temporal_e3_scene_sum[scene] += temporal_e3[t].item()

                val_loss_scene_cnt[scene] += 1.0

                if val_loss_scene_cnt[scene] > 18.0: continue # don't visualize too much
                
                # if total_image_cnt % 6 == 0:
                #     x, y = find_good_pixel_with_penumbra(penumbra_width[t], kernel_half_actual_size)
                x, y = H // 2, W // 2
                index = val_input_dict["id"][t]

                with open(os.path.join(val_log_dir, f"visualize_pixel{index}.txt"), "w") as file:
                    print("x, y, index is", x, y, index, file=file)

                save_image(shadowMap_filtered[t], os.path.join(val_log_dir, f"filtered{step}_{index}.png"))
                visualize_and_save_difference(shadowMap_filtered[t], gt[t], os.path.join(val_log_dir, f"diff_visualization{step}_{index}.png"))
                
                if filter_size > 1:
                    vis_kernels = kernel[t, :, :, x - kernel_half_actual_size : x + kernel_half_actual_size + 1, y - kernel_half_actual_size : y + kernel_half_actual_size + 1] # (dilation + 1, filter_size * filter_size, h, w)

                    save_dilated_kernels(vis_kernels[:, :, kernel_half_actual_size, kernel_half_actual_size].view(dilation+1, filter_size, filter_size), os.path.join(val_log_dir, f"dilated_kernel{step}_{index}.png"))

                    full_kernel = reconstruct_full_kernel(vis_kernels.view(dilation + 1, filter_size, filter_size, vis_kernels.shape[-2], vis_kernels.shape[-1]))
                    save_kernel(full_kernel, os.path.join(val_log_dir, f"full_kernel{step}_{index}.png"))

                if val_use_temporal:
                    save_image(torch.abs(shadowMap_filtered[t] - shadowMap_filtered_projected[t]) * combined_mask[t], os.path.join(val_log_dir, f"temporal_diff_masked{step}_{index}.png"))

                    save_image(combined_mask[t], os.path.join(val_log_dir, f"combined_mask{step}_{index}.png"))
                    # save_image(mask[t], os.path.join(val_log_dir, f"mask{step}_{index}.png"))
                    # save_image(valid[t], os.path.join(val_log_dir, f"valid{step}_{index}.png"))
                    # save_image((torch.abs(depth - depth_projected) < depth_tolerance)[t], os.path.join(val_log_dir, f"depth_mask{step}_{index}.png"))
                    # save_image((torch.norm(normal - normal_projected, dim=1, p=2, keepdim=True) < normal_tolerance)[t], os.path.join(val_log_dir, f"normal_mask{step}_{index}.png"))
                    # save_image(torch.abs(depth - depth_projected)[t], os.path.join(val_log_dir, f"depth_diff{step}_{index}.png"))
                    # save_image(torch.norm(normal - normal_projected, dim=1, p=2, keepdim=True)[t], os.path.join(val_log_dir, f"normal_diff{step}_{index}.png"))
                    # save_image(depth_projected[t], os.path.join(val_log_dir, f"projected_depth{step}_{index}.png"))
                    # save_image(normal_projected[t].permute((1, 2, 0)) / 2.0 + 0.5, os.path.join(val_log_dir, f"projected_normal{step}_{index}.png"))
                    # save_image(normal[t].permute((1, 2, 0)) / 2.0 + 0.5, os.path.join(val_log_dir, f"normal{step}_{index}.png"))

                total_image_cnt += 1

            end_time = time.time()  
            total_inference_time += (end_time - start_time)
            total_inference_cnt += 1

    avg_inference_time = total_inference_time / total_inference_cnt 
    avg_mse = total_mse / validation_cnt
    avg_l1 = total_l1 / validation_cnt
    avg_ssim = total_ssim / validation_cnt
    avg_temporal_l2 = total_temporal_l2 / validation_cnt
    avg_temporal_l1 = total_temporal_l1 / validation_cnt
    avg_temporal_e3 = total_temporal_e3 / validation_cnt

    logger.add_scalar(f'Speed/val_{prefix}', avg_inference_time, step)
    logger.add_scalar(f'Loss_val{prefix}/MSE_val', avg_mse, step)
    logger.add_scalar(f'Loss_val{prefix}/L1_val', avg_l1, step)
    logger.add_scalar(f'Loss_val{prefix}/ssim', avg_ssim, step)
    logger.add_scalar(f'Loss_val{prefix}/temporal_L2_val', avg_temporal_l2, step)
    logger.add_scalar(f'Loss_val{prefix}/temporal_L1_val', avg_temporal_l1, step)
    logger.add_scalar(f'Loss_val{prefix}/temporal_e3_val', avg_temporal_e3, step)

    scene_names = dataset_config["scenes_val"]
    for i, scene_name in enumerate(scene_names):
        avg_mse_scene = val_mse_scene_sum[i] / val_loss_scene_cnt[i]
        avg_l1_scene = val_l1_scene_sum[i] / val_loss_scene_cnt[i]
        avg_ssim_scene = val_ssim_scene_sum[i] / val_loss_scene_cnt[i]
        logger.add_scalar(f"Loss_val_scene_{prefix}/{scene_name}/MSE_val", avg_mse_scene, step)
        logger.add_scalar(f"Loss_val_scene_{prefix}/{scene_name}/L1_val", avg_l1_scene, step)
        logger.add_scalar(f"Loss_val_scene_{prefix}/{scene_name}/ssim", avg_ssim_scene, step)

    print(f"Average inference time per forward pass: {avg_inference_time} seconds")

    if not half:
        model.train()
    else:
        del model
        orig_model = orig_model.to(device)