import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch_dct as dct
from torchvision.transforms.functional import gaussian_blur

class imgSigSal(nn.Module):
    """
    Image Signature saliency implementation with dynamic blur kernel and resizing.
    Based on: "Image Signature: Highlighting sparse salient regions"
    by Xiaodi Hou, Jonathan Harel, and Christof Koch (2011).
    """
    def __init__(
        self, 
        blur_sigma: float = 0.045, 
        map_width: int = 64,
        resize_to_input: bool = True,
        normalize: bool = True,
        resize_mode: str = 'bilinear'
    ):
        """
        Args:
            blur_sigma: Blur kernel size as fraction of image width 
                        (MATLAB default: 0.045)
            map_width: Width to resize image to for processing
                       (MATLAB default: 64)
            resize_to_input: Whether to resize output to match input dimensions
                             (MATLAB default: True)
            normalize: Whether to normalize output to [0, 1]
            resize_mode: Interpolation mode for resizing 
                         ('bilinear', 'bicubic', 'nearest')
        """
        super().__init__()
        self.blur_sigma = blur_sigma
        self.map_width = map_width
        self.resize_to_input = resize_to_input
        self.normalize = normalize
        self.resize_mode = resize_mode
    
    def _get_gaussian_kernel_2d(
        self, 
        kernel_size: int, 
        sigma: float, 
        device: torch.device
    ) -> Tensor:
        """
        Creates a 2D Gaussian kernel (equivalent to MATLAB's fspecial('gaussian')).
        """
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create 1D Gaussian kernel
        x = torch.arange(kernel_size, device=device, dtype=torch.float32)
        x = x - (kernel_size - 1) / 2
        gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
        
        # Create 2D kernel via outer product
        gauss_2d = gauss_1d.outer(gauss_1d)
        
        # Normalize
        gauss_2d = gauss_2d / gauss_2d.sum()
        
        return gauss_2d
    
    def _dynamic_gaussian_blur(self, x: Tensor) -> Tensor:
        """
        Applies Gaussian blur with kernel size based on image width.
        
        MATLAB equivalent:
            kSize = size(outMap,2) * param.blurSigma;
            outMap = imfilter(outMap, fspecial('gaussian', round([kSize, kSize]*4), kSize));
        """
        if self.blur_sigma <= 0:
            return x
        
        B, C, H, W = x.shape
        
        # Calculate sigma based on image width
        sigma = W * self.blur_sigma
        
        # Kernel size is 4x sigma (matching MATLAB)
        kernel_size = int(round(sigma * 4))
        
        # Ensure minimum and odd kernel size
        kernel_size = max(kernel_size, 3)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create Gaussian kernel
        kernel = self._get_gaussian_kernel_2d(kernel_size, sigma, x.device)
        
        # Reshape for conv2d: (out_channels, in_channels/groups, kH, kW)
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        kernel = kernel.expand(C, 1, kernel_size, kernel_size)
        
        # Padding for 'same' output size
        padding = kernel_size // 2
        
        # Apply channel-wise convolution
        x_blurred = F.conv2d(x, kernel, padding=padding, groups=C)
        
        return x_blurred
    
    def _resize(
        self, 
        x: Tensor, 
        target_height: int, 
        target_width: int
    ) -> Tensor:
        """
        Resize tensor to target dimensions.
        
        MATLAB equivalent:
            img = imresize(input_img, param.mapWidth/size(input_img, 2));
        """
        return F.interpolate(
            x, 
            size=(target_height, target_width), 
            mode=self.resize_mode,
            align_corners=False if self.resize_mode in ['bilinear', 'bicubic'] else None
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Compute Image Signature saliency map.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Saliency map of shape (B, 1, H, W) if resize_to_input=True
            Otherwise (B, 1, map_height, map_width)
        """
        # Store original dimensions for potential resize back
        B, C, H_orig, W_orig = x.shape
        
        # Calculate target dimensions maintaining aspect ratio
        # MATLAB: img = imresize(input_img, param.mapWidth/size(input_img, 2));
        scale = self.map_width / W_orig
        map_height = int(round(H_orig * scale))
        map_width = self.map_width
        
        # Resize input to processing size
        x_resized = self._resize(x, map_height, map_width)
        
        # DCT transform
        x_dct = dct.dct_2d(x_resized)
        
        # Extract sign (image signature)
        x_imSig = torch.sign(x_dct)
        
        # Inverse DCT reconstruction
        x_rec = dct.idct_2d(x_imSig)
        
        # Square
        x_squared = x_rec * x_rec
        
        # Dynamic Gaussian blur
        sal = self._dynamic_gaussian_blur(x_squared)
        
        # Mean across channels (matching MATLAB)
        sal = sal.mean(dim=1, keepdim=True)
        
        # Resize back to original dimensions if requested
        # MATLAB: outMap = imresize(outMap, [size(input_img,1) size(input_img,2)]);
        if self.resize_to_input:
            sal = self._resize(sal, H_orig, W_orig)
        
        # Normalize to [0, 1]
        if self.normalize:
            sal_flat = sal.view(B, -1)
            sal_min = sal_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            sal_max = sal_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            sal = (sal - sal_min) / (sal_max - sal_min + 1e-10)
        
        # Detach from computation graph
        sal = sal.detach()
        
        return sal