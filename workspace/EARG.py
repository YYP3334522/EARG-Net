import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.cuda.amp import autocast
import os

from IMDLBenCo.registry import MODELS

import random
import cv2
import numpy as np
from PIL import Image

from AEO import EdgeEnhancementBlock
from diffusers import StableDiffusionInpaintPipeline
from dual_segformer import mit_b5
from MLPDecoder import DecoderHead
from mask2label import MaskToLabelCNNMLP




@MODELS.register_module()
class EARG(nn.Module):
    def __init__(self, device) -> None:

        super().__init__()
    
        self.device = device
        sd_model_path = os.environ.get(
            "EARG_SD_MODEL_PATH",
            "/data/yuyanpu/model/SD/stable-diffusion-2-inpainting",
        )
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            sd_model_path,
            revision="fp16",
            torch_dtype=torch.float16,
            ).to(self.device)
        # Inpainting branch is inference-only; freeze internal trainable modules.
        for module_name in ["unet", "vae", "text_encoder"]:
            module = getattr(self.pipe, module_name, None)
            if module is not None:
                module.requires_grad_(False)
        self.aeo = EdgeEnhancementBlock(3).to(self.device)
        # Current forward uses AEO only for visualization, not for loss.
        self.aeo.requires_grad_(False)
        self.extractor = mit_b5().to(self.device)
        self.decoder = DecoderHead(
            in_channels=[64, 128, 320, 512],
            original_size=(512, 512),
            num_classes=1,
        ).to(self.device)
        self.det_head = MaskToLabelCNNMLP().to(self.device)

        pos_weight = torch.tensor(2.0).to(self.device)
        self.label_loss_fun = nn.BCEWithLogitsLoss()
        self.local_loss_fun = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )

    def denormalize_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        return image_tensor * self.imagenet_std + self.imagenet_mean
    
    def edge_aware_target_mask(self, image_tensor, mask_ratio=0.75, max_patch_size=16):

        B, C, H, W = image_tensor.shape
        device = image_tensor.device
        masked_tensor = image_tensor.clone()
        mask_tensor = torch.zeros((B, 1, H, W), dtype=torch.uint8, device=device)
        
        for b in range(B):
            img_np = image_tensor[b].permute(1, 2, 0).cpu().numpy()
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if C == 3 else img_np[..., 0]
            gray_uint8 = (gray * 255).astype(np.uint8)
            
            sobel_x = cv2.Sobel(gray_uint8, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_uint8, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
            
            laplacian = cv2.Laplacian(gray_uint8, cv2.CV_64F)
            laplacian = np.abs(laplacian)
            
            heatmap = 0.6 * gradient_mag + 0.4 * laplacian
            max_val = heatmap.max()
            if max_val <= 0:
                continue
            heatmap = heatmap / max_val
            
            _, binary = cv2.threshold(
                (heatmap * 255).astype(np.uint8), 
                0, 255, 
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            binary = (binary > 0).astype(np.float32)
            heatmap = torch.from_numpy(binary).float().to(device)
            
            current_ratio = 0
            
            edge_points = torch.where(heatmap > 0)
            if len(edge_points[0]) == 0:
                continue  
                
            for attempt in range(500):  
                if current_ratio >= mask_ratio:
                    break
                    
                idx = random.randint(0, len(edge_points[0]) - 1)
                y, x = edge_points[0][idx], edge_points[1][idx]
                
                size = max_patch_size
                
                y1 = max(0, y - size//2)
                y2 = min(H, y + size//2)
                x1 = max(0, x - size//2)
                x2 = min(W, x + size//2)
                if (y2-y1)*(x2-x1) < 4: 
                    continue
                    
                if mask_tensor[b, 0, y1:y2, x1:x2].sum() > 0:
                    continue
                    
                patch_ratio = (y2-y1)*(x2-x1)/(H*W)
                if current_ratio + patch_ratio > mask_ratio:
                    continue
                    
                masked_tensor[b, :, y1:y2, x1:x2] = 0
                mask_tensor[b, 0, y1:y2, x1:x2] = 255
                current_ratio += patch_ratio
        
        return masked_tensor, mask_tensor
    
    def pixel_difference_with_wrap(self, tensor1, tensor2):
    
        diff = tensor1 - tensor2
        abs_diff = torch.abs(diff)
        return torch.minimum(abs_diff, 1 - abs_diff)

    def inpainting(self, masked_tensor_batch: torch.Tensor, original_tensor_batch: torch.Tensor, mask_tensor_batch: torch.Tensor) -> torch.Tensor:

        B, C, H, W = masked_tensor_batch.shape
        masked_tensor = masked_tensor_batch.float() 
        mask_tensor = mask_tensor_batch.float()
        original_tensor_batch = original_tensor_batch.float()  
        with torch.no_grad():
            generated_image = self.pipe(
                    prompt=[""] * B,
                    image=masked_tensor,
                    mask_image=mask_tensor / 255.0,
                    height=H,  
                    width=W,   
                    num_inference_steps=50,  
                    strength=1,  
                    guidance_scale = 7.5,
                    ).images  
        
        transform = transforms.Compose([             
                transforms.ToTensor(),  
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
                transforms.Resize([H,W]),                        
            ])
        recons = None
        if isinstance(generated_image, list) and all(isinstance(img, Image.Image) for img in generated_image):
                tensor_images = [transform(img) for img in generated_image] 
                recons = torch.stack(tensor_images).to(self.device)  
        if recons is None:
            raise RuntimeError("Unexpected inpainting output format from StableDiffusionInpaintPipeline.")
                   
        diff = self.pixel_difference_with_wrap(original_tensor_batch, recons)

        return recons, diff
    def _create_2d_gaussian_kernel(self, sigma: float, kernel_size: int = 7, device='cpu'):

        x = torch.linspace(-3, 3, kernel_size, device=device)
        y = torch.linspace(-3, 3, kernel_size, device=device)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, kernel_size, kernel_size)
    
    def spatial_crf(self, prob_tensor: torch.Tensor, 
                    num_iterations: int = 5,
                    spatial_sigma: float = 3.0) -> torch.Tensor:
        
        assert prob_tensor.dim() == 4 and prob_tensor.size(1) == 1
        device = prob_tensor.device
        
        q = torch.clamp(prob_tensor, 1e-5, 1-1e-5)
        q = torch.cat([1-q, q], dim=1)  
        
        kernel = self._create_2d_gaussian_kernel(spatial_sigma, device=device)  
        kernel = kernel.expand(2, 1, 7, 7)  
        
        for _ in range(num_iterations):
            q = F.conv2d(q, kernel, padding=3, groups=2)
            q = torch.clamp(q, 1e-5, 1-1e-5)
            q = q / q.sum(dim=1, keepdim=True)
        
        return q[:,1:2]  
    
    def dice_loss(self, prediction, target, smooth=1.0):
        target = target.reshape(-1)
        prediction = prediction.reshape(-1)
        intersection = (target * prediction).sum()
        dice = (2.0 * intersection + smooth) / (torch.square(target).sum() + torch.square(prediction).sum() + smooth)

        return 1.0 - dice

    def forward(self, image, mask, label, *args, **kwargs):
        with autocast(enabled=False):
            image_for_sd = torch.clamp(self.denormalize_image(image), 0.0, 1.0)
            masked_image, mask_image = self.edge_aware_target_mask(image_for_sd)
            recon_image, diff_image = self.inpainting(masked_image, image, mask_image)
            sobel_feature = self.aeo(diff_image)
            
            artifact = self.extractor(image, diff_image)
            pred_mask_logits = self.decoder(artifact)
            pred_mask = self.spatial_crf(torch.sigmoid(pred_mask_logits))
            pred_label = self.det_head(pred_mask)

            bce_loss = self.local_loss_fun(pred_mask_logits, mask.float()) 

            dice_loss = self.dice_loss(pred_mask, mask.float())
            dice_weight = 0.1
            local_loss = bce_loss + dice_loss * dice_weight

            label_loss = self.label_loss_fun(pred_label, label.float())
            label_weight = 0.3
            total_loss = local_loss + label_loss * label_weight


        output_dict = {
            "backward_loss": total_loss,
            "pred_mask": pred_mask,
            "pred_label": pred_label,

            "visual_loss": {
            "predict_loss": total_loss,
            "bce_loss": bce_loss ,
            "dice_loss": dice_loss* dice_weight,
            "label_loss": label_loss * label_weight,
            },

            "visual_image": {
                "pred_mask": pred_mask,
                "recon_image": recon_image,
                "sobel_feature": sobel_feature,
            }
            # -----------------------------------------
        }

        return output_dict
if __name__ == "__main__":
    print(MODELS)