import torch
import torch.nn.functional as F
import imageio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def save_image(tensor, filepath):
  array = tensor.cpu().numpy().squeeze()  
  array = (array * 255).astype(np.uint8)
  imageio.imwrite(filepath, array)

def save_kernel(kernel, filepath):
  if isinstance(kernel, torch.Tensor):
    kernel = kernel.cpu().numpy().squeeze()
  fig, ax = plt.subplots(figsize=(6, 6))
  im = ax.imshow(kernel, cmap='viridis')
    
  # Add colorbar
  divider = make_axes_locatable(ax)
  cax = divider.append_axes('right', size='5%', pad=0.1)
  plt.colorbar(im, cax=cax)
  
  ax.axis('off')
  plt.tight_layout()
  
  plt.savefig(filepath, bbox_inches='tight', pad_inches=1)
  plt.close(fig)


def apply_kernel(x, kernel, filter_size, dilation):
  """
  apply a dilated kernel
  x: (B, C, H, W)
  kernel: (B, dilation + 1, filter_size^2, H, W) from expand_kernels
  """
  half_filter_size = filter_size // 2
  B, C, H, W = x.shape
  x = torch.cat([x, torch.ones((B, 1, H, W), dtype=x.dtype, device=x.device)], dim=1)
  for i in range(dilation + 1):
    d = 1 << i
    x_unfold = F.unfold(x, kernel_size=filter_size, dilation=d, padding=d*half_filter_size).view(B, C+1, filter_size*filter_size, H, W)
    k = kernel[:, i:i+1, ...] # (B, 1, filter_size*filter_size, H, W)
    x = (x_unfold * k).sum(dim=2)
    x = x / (x[:, C:C+1, :, :] + 1e-10) # normalize according to sum of weights
  return x[:, :C, :, :]
  
def apply_kernel_with_mask(x, kernel, filter_size, dilation, mask):
  """
  apply a dilated kernel
  x: (B, C, H, W)
  kernel: (B, dilation + 1, filter_size^2, H, W) from expand_kernels
  mask: (B, 1, H, W) masking invalid pixels
  """
  half_filter_size = filter_size // 2
  B, C, H, W = x.shape
  x = torch.cat([x * mask, mask], dim=1)
  for i in range(dilation + 1):
    d = 1 << i
    x_unfold = F.unfold(x, kernel_size=filter_size, dilation=d, padding=d*half_filter_size).view(B, C+1, filter_size*filter_size, H, W)
    k = kernel[:, i:i+1, ...] # (B, 1, filter_size*filter_size, H, W)
    x = (x_unfold * k).sum(dim=2)
    x = torch.where(x[:, C:C+1, :, :] > 1e-5, x / x[:, C:C+1, :, :], 0.0) # normalize according to sum of weights
    x[:, C:C+1, :, :] = mask
    x[:, :C, :, :] *= mask
  return x[:, :C, :, :]

def expand_kernels(kernel, filter_size, dilation):
  """
  expand the output of a network (B, filter_size * filter_size * (dilation + 1), ...)
  to (B, dilation + 1, filter_size * filter_size, ...)
  with softmax applied
  """
  B, _, H, W = kernel.shape
  kernel = kernel.reshape(B, dilation+1, filter_size * filter_size, H, W)
  kernel = F.softmax(kernel, dim=2)
  # fixed_weight = kernel_fixed_weight(filter_size).to(kernel.device)[None, None, :, None, None]
  # kernel * fixed_weight, normalize
  return kernel

def save_dilated_kernels(kernels, path):
  """
  save visualization of the kernels
  kernels: kernels of only one pixel, of shape (dilation + 1, filter_size, filter_size)
  """
  L = kernels.shape[0]
  kernels = kernels.cpu().numpy()
  vmin = min([kernel.min() for kernel in kernels])
  vmax = max([kernel.max() for kernel in kernels])
  fig, axes = plt.subplots(1, L, figsize=(L * 3, 3), sharey=True)
  if L == 1:
    axes = [axes] # make it iterable
  for i, ax in enumerate(axes):
    im = ax.imshow(kernels[i], cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title(f'Kernel{i}')
    ax.axis('off')
  
  cax = make_axes_locatable(axes[-1]).append_axes("right", size="5%", pad=0.05)
  plt.colorbar(im, cax=cax)
  
  plt.tight_layout()
  plt.savefig(path)
  plt.close(fig)

def reconstruct_full_kernel(kernels):
  """
  reconstruct a full kernel from dilated kernels
  input kernels should be [(dilation + 1), filter_size, filter_size, kernel_actual_size, kernel_actual_size]
  return a numpy array [kernel_actual_size, kernel_actual_size] representing full kernel of the centered pixel
  """
  kernels = kernels.cpu().numpy()
  dilation, filter_size, actual_size = kernels.shape[0] - 1, kernels.shape[1], kernels.shape[3]
  half_actual_size, half_filter_size = actual_size // 2, filter_size // 2
  results = np.zeros((actual_size, actual_size), dtype=np.float32)
  temp = np.zeros((actual_size, actual_size), dtype=np.float32)
  results[half_actual_size, half_actual_size] = 1.0

  for j in reversed(range(dilation + 1)):
    d = 1 << j
    temp = results * kernels[j, half_filter_size, half_filter_size, :, :]
    for k in range(half_filter_size + 1):
      o_x = k * d
      m_o_x = actual_size - o_x
      for l in range(half_filter_size + 1):
        o_y = l * d
        m_o_y = actual_size - o_y
        if k > 0 or l > 0:
          temp[o_x:, o_y:] += kernels[j, half_filter_size + k, half_filter_size + l, :m_o_x, :m_o_y] * results[:m_o_x, :m_o_y]
          if k > 0:
            temp[:m_o_x, o_y:] += kernels[j, half_filter_size - k, half_filter_size + l, o_x:, :m_o_y] * results[o_x:, :m_o_y]
          if l > 0:
            temp[o_x:, :m_o_y] += kernels[j, half_filter_size + k, half_filter_size - l, :m_o_x, o_y:] * results[:m_o_x, o_y:]
          if k > 0 and l > 0:
            temp[:m_o_x, :m_o_y] += kernels[j, half_filter_size - k, half_filter_size - l, o_x:, o_y:] * results[o_x:, o_y:]
    results = temp
    # print(j, np.sum(results), np.sum(kernels[j], axis=(0, 1)).max(), np.sum(kernels[j], axis=(0, 1)).min())
  return results

def dilation_kernel_actual_size(filter_size, dilation):
  len = 1
  for i in range(dilation + 1):
    len = len + (1 << i) * (filter_size - 1)
  return len

def visualize_and_save_difference(shadowMap_filtered, gt, output_path, threshold=0.0001):
  diff_map = torch.abs(shadowMap_filtered - gt).squeeze(0).cpu().numpy()
  diff_map = np.clip(diff_map - threshold, 0, 1)
  plt.figure(figsize=(8, 8))
  plt.imshow(diff_map, cmap='jet')  
  plt.colorbar(label="Difference Magnitude") 
  plt.title('ShadowMap Difference Visualization')
  plt.axis('off')

  plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
  plt.close()

def find_good_pixel_with_penumbra(penumbra_size, half_actual_size):
  penumbra_size = penumbra_size.squeeze().cpu().numpy() # (H, W)
  penumbra_size = penumbra_size[half_actual_size + 1 : -half_actual_size - 1, half_actual_size + 1 : -half_actual_size - 1]
  ind = np.unravel_index(np.argmax(penumbra_size, axis=None), penumbra_size.shape)
  return ind[0] + half_actual_size + 1, ind[1] + half_actual_size + 1
