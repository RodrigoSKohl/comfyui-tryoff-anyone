import torch
import torch.nn.functional as F
from comfy.model_management import get_torch_device
from .tryoffanyone_pipeline import TryOffAnyone  # Importe o pipeline TryOffAnyone

class TryOffAnyoneNode:
    def __init__(self):
        self.device = get_torch_device()
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.pipeline = TryOffAnyone(device=self.device, dtype=self.dtype)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  
                "mask": ("MASK",),    
                "inference_steps": ("INT", {"default": 50, "min": 1, "max": 1000}),
                "guidance_scale": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": 40, "min": 0, "max": 0xffffffffffffffff}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
                "image_width": ("INT", {"default": 384, "min": 64, "max": 512, "step": 8}),
                "image_height": ("INT", {"default": 512, "min": 64, "max": 512, "step": 8}),
            },
        }

    RETURN_TYPES = ("IMAGE",)  # Retorna tanto a imagem gerada quanto o preview
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "inpainting"

    def process(self, image, mask, inference_steps, guidance_scale, seed, image_width, image_height, denoise):
        image = image_preprocess(image, image_height, image_width)
        mask = mask_preprocess(mask, image_height, image_width)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        output_image  = self.pipeline(
            image=image,
            mask=mask,
            inference_steps=inference_steps,
            scale=guidance_scale,
            denoise=denoise,
            generator=generator,
        )

        return output_image 
    
def image_preprocess(image: torch.Tensor, target_height, target_width):
    print(f"Original Image shape: {image.shape}")
    image = image.permute(0, 3, 1, 2) 
    if image.shape[1] == 4:
        image = image[:, :3, :, :]
    image = F.interpolate(image, size=(target_height, target_width), mode='bilinear', align_corners=False)
    image = image * 2 - 1
    print(f"Preprocessed Image shape: {image.shape}")
    return image

def mask_preprocess(mask: torch.Tensor, target_height, target_width):
    mask = mask.unsqueeze(1)
    mask = F.interpolate(mask, size=(target_height, target_width), mode='nearest')
    mask = (mask > 0.5).float()
    return mask



