import os
import json
import torch
import numpy as np
from torch import Tensor
from typing import Dict
from scipy.ndimage import maximum_filter, generate_binary_structure

def hard_sal(sal_map: torch.Tensor, k:float=0.1, seed:int = None):
    """
    Anchor selection method for hard thresholding based on saliency
    """
    if seed is not None:
        torch.manual_seed(seed)
    assert sal_map.ndim == 2, "tensor must be 2D"
    assert 0 <= k <= 1.0, "dropout rate must be between 0 and 1"

    H, W  = sal_map.shape
    total_points = H * W
    num_points = max(1, int(total_points * (1-k)))

    flat_map = sal_map.flatten()
    values, indices = torch.topk(flat_map, num_points) # hard threshold

    ys = indices // W
    xs = indices % W

    # preserving anchors order
    max_x = xs.max().item() + 1
    keys = ys * max_x + xs
    sorted_indices = torch.argsort(keys)
    ys = ys[sorted_indices]
    xs = xs[sorted_indices]

    return ys, xs

def soft_sal(sal_map: torch.Tensor, k:float=0.1, seed:int=None):
    """
    Anchor selection method for probabilistic soft thresholding based on saliency.
    """
    if seed is not None:
        torch.manual_seed(seed)

    assert sal_map.ndim == 2, "tensor must be 2D"
    assert 0 <= k <= 1.0, "dropout rate must be between 0 and 1"

    H, W = sal_map.shape
    total_points = H * W
    num_points = max(1, int(total_points * (1 - k)))

    # Normalize to positive probabilities
    sal = sal_map - sal_map.min()
    sal = sal / (sal.sum() + 1e-8)   # IMPORTANT: multinomial expects weights

    flat_sal = sal.flatten()

    # Handle degenerate case (all zeros)
    if flat_sal.sum() == 0:
        flat_sal = torch.ones_like(flat_sal) / flat_sal.numel()

    # Sample WITHOUT replacement
    indices = torch.multinomial(
        flat_sal,
        num_samples=num_points,
        replacement=False
    )

    ys = indices // W
    xs = indices % W

    # Deterministic spatial ordering (optional, keeps your behavior)
    max_x = xs.max().item() + 1
    keys = ys * max_x + xs
    sorted_indices = torch.argsort(keys)
    ys = ys[sorted_indices]
    xs = xs[sorted_indices]

    return ys, xs


def random(sal_map: torch.Tensor, k: float = 1.0, seed: int = None):
    """
    Anchor selection method selected randomly
    """
    if seed is not None:   
        torch.manual_seed(seed)
    
    assert sal_map.ndim == 2, "tensor must be 2D"
    assert 0 <= k <= 1.0, "dropout rate must be between 0 and 1"

    H,W  = sal_map.shape
    total_points = H * W
    num_points = max(1, int(total_points * (1-k)))

    ys, xs = torch.meshgrid(
        torch.arange(H, device = sal_map.device),
        torch.arange(W, device = sal_map.device),
        indexing = "ij"
    )
    ys, xs = ys.flatten(), xs.flatten()
    indices = torch.randperm(total_points, device = sal_map.device)[:num_points] # random selection
    ys = ys[indices]
    xs = xs[indices]

    #preserving anchors order
    max_x = xs.max().item() + 1
    keys = ys * max_x + xs
    sorted_indices = torch.argsort(keys)
    ys = ys[sorted_indices]
    xs = xs[sorted_indices]
    return ys, xs

def gaussian_mask(sal_map: torch.Tensor, k: float = 0.1, sigma_scale: float = 0.15, device="cuda", seed=None):
    """
    Anchor selection method with a spatial prior (centerBias)
    """
    if seed is not None:
        torch.manual_seed(seed)

    H, W = sal_map.shape
    total_pixels = H * W
    num_keep = max(1, int(total_pixels * (1-k)))

    y = torch.arange(H, device=device).float()
    x = torch.arange(W, device=device).float()
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    cy = (H - 1) / 2
    cx = (W - 1) / 2
    sigma = sigma_scale * min(H, W)

    # gaussian mask
    mask = torch.exp(-((yy - cy)**2 + (xx - cx)**2) / (2 * sigma**2))

    # normalize to probabilities
    probs = mask.flatten() / mask.sum()

    # sample pixels according to probability
    indices = torch.multinomial(probs, num_keep, replacement=False)

    ys = indices // W
    xs = indices % W

    # preserving anchors order
    max_x = xs.max().item() + 1
    keys = ys * max_x + xs
    sorted_indices = torch.argsort(keys)
    ys = ys[sorted_indices]
    xs = xs[sorted_indices]

    return ys, xs

def create_saliency_mask(sal_map: torch.Tensor, k: float=0.5, sal_method: str = 'hard_sal', seed: int = 42, device: str = 'cuda'):
    """
    Helper function for creating a 2D saliency mask for objectness and regression scores
    """
    H, W = sal_map.shape
    mask = torch.zeros_like(sal_map, dtype=torch.float32, device=device)
    
    if sal_method == 'maxima':
        ys, xs = maxima(sal_map, k = k, seed = seed)
    elif sal_method == 'hard_sal':
        ys, xs = hard_sal(sal_map, k= k, seed = seed)
    elif sal_method == 'hard_sal_low':
        ys, xs = hard_sal_low(sal_map, k = k, seed = seed)
    elif sal_method == 'soft_sal':
        ys, xs = soft_sal(sal_map, k = k, seed = seed)
    elif sal_method == 'min_and_max':
        ys, xs = min_and_max(sal_map, k = k, seed = seed)
    elif sal_method == 'random':
        ys, xs = random(sal_map, k = k, seed = seed)
    elif sal_method == 'soft_sal_random':
        ys, xs = soft_sal_random(sal_map, k = k, seed = seed)
    elif sal_method == 'gaussian':
        ys, xs = gaussian_mask(sal_map, k = k, seed = seed)
    else:
        return ValueError("Sal_method must be 'maxima', 'hard_sal', 'hard_sal_low', 'soft_sal', 'min_and_max', 'random', 'soft_sal_random', or 'gaussian'.")

    # Set poitns to 1
    mask[ys, xs] = 1.0

    return mask


def concat_masked_box_prediction_layers(box_cls: list[torch.Tensor], box_regression: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Modification of R-CNN source code to apply the saliency mask to objectness and regression scores.
    """
    box_cls_flattened = []
    box_regression_flattened = []
        
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
            
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)

        box_cls_flattened.append(box_cls_per_level)
        box_regression_flattened.append(box_regression_per_level)

    # concatenate all levels
    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)

        # filtering out zero-entries
    non_zero_mask = (box_cls != 0).any(dim=1) # if any value in a row is nonzero

        # apply mask
    box_cls = box_cls[non_zero_mask]
    box_regression = box_regression[non_zero_mask]

    return box_cls, box_regression

def permute_and_flatten(layer: Tensor, N: int, A: int, C: int, H: int, W: int) -> Tensor:
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer