# type: ignore
# Inspired from https://github.com/ixarchakos/try-off-anyone/blob/aa3045453013065573a647e4536922bac696b968/src/model/pipeline.py
# Inspired from https://github.com/ixarchakos/try-off-anyone/blob/aa3045453013065573a647e4536922bac696b968/src/model/attention.py

import torch
from accelerate import load_checkpoint_in_model
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import hf_hub_download
from PIL import Image


class Skip(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self,
        attn: torch.Tensor,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        temb: torch.Tensor = None,
    ) -> torch.Tensor:
        return hidden_states


def fine_tuned_modules(unet: UNet2DConditionModel) -> torch.nn.ModuleList:
    trainable_modules = torch.nn.ModuleList()

    for blocks in [unet.down_blocks, unet.mid_block, unet.up_blocks]:
        if hasattr(blocks, "attentions"):
            trainable_modules.append(blocks.attentions)
        else:
            for block in blocks:
                if hasattr(block, "attentions"):
                    trainable_modules.append(block.attentions)

    return trainable_modules


def skip_cross_attentions(unet: UNet2DConditionModel) -> dict[str, AttnProcessor | Skip]:
    attn_processors = {
        name: unet.attn_processors[name] if name.endswith("attn1.processor") else Skip()
        for name in unet.attn_processors.keys()
    }
    return attn_processors


def encode(image: torch.Tensor, vae: AutoencoderKL) -> torch.Tensor:
    image = image.to(memory_format=torch.contiguous_format).float().to(vae.device, dtype=vae.dtype)
    with torch.no_grad():
        return vae.encode(image).latent_dist.sample() * vae.config.scaling_factor


class TryOffAnyone:
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        concat_dim: int = -2,
    ) -> None:
        self.concat_dim = concat_dim
        self.device = device
        self.dtype = dtype

        self.noise_scheduler = DDIMScheduler.from_pretrained(
            pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-inpainting",
            subfolder="scheduler",
        )
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path="stabilityai/sd-vae-ft-mse",
        ).to(device, dtype=dtype)
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-inpainting",
            subfolder="unet",
            variant="fp16",
        ).to(device, dtype=dtype)

        self.unet.set_attn_processor(skip_cross_attentions(self.unet))
        load_checkpoint_in_model(
            model=fine_tuned_modules(unet=self.unet),
            checkpoint=hf_hub_download(
                repo_id="ixarchakos/tryOffAnyone",
                filename="model.safetensors",
            ),
        )

    @torch.no_grad()
    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        inference_steps: int,
        scale: float,
        denoise: float,
        generator: torch.Generator,
    ) -> list[Image.Image]:
        print(f"Original Image - min: {image.min()}, max: {image.max()}")
        print(f"Original Mask - min: {mask.min()}, max: {mask.max()}")
        print(f"Image shape: {image.shape}")
        print(f"Mask shape: {mask.shape}")
        image = image.to(self.device, dtype=self.dtype)
        mask = (mask > 0.5).to(self.device, dtype=self.dtype)
        masked_image = image * (mask < 0.5)
        print(f"Masked Image - min: {masked_image.min()}, max: {masked_image.max()}")
        print(f"Masked Image shape: {masked_image.shape}")
        masked_latent = encode(masked_image, self.vae)
        image_latent = encode(image, self.vae)
        print(f"masked_latent - min: {masked_latent.min()}, max: {masked_latent.max()}")
        print(f"image_latent - min: {image_latent.min()}, max: {image_latent.max()}")
        print(f"masked_latent shape: {masked_latent.shape}")
        print(f"image_latent shape: {image_latent.shape}")
        mask = torch.nn.functional.interpolate(mask, size=masked_latent.shape[-2:], mode="nearest")
        masked_latent_concat = torch.cat([masked_latent, image_latent], dim=self.concat_dim)
        mask_concat = torch.cat([mask, torch.zeros_like(mask)], dim=self.concat_dim)
        print(f"masked_latent_concat - min: {masked_latent_concat.min()}, max: {masked_latent_concat.max()}")


        latents = randn_tensor(
            shape=masked_latent_concat.shape,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

        self.noise_scheduler.set_timesteps(inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps

        if do_classifier_free_guidance := (scale > 1.0):
            masked_latent_concat = torch.cat(
                [
                    torch.cat([masked_latent, torch.zeros_like(image_latent)], dim=self.concat_dim),
                    masked_latent_concat,
                ]
            )

            mask_concat = torch.cat([mask_concat] * 2)

        extra_step = {"generator": generator, "eta": denoise}
        for t in timesteps:
            print(f"Latents before timestep {t} - min: {latents.min()}, max: {latents.max()}")
            input_latents = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            input_latents = self.noise_scheduler.scale_model_input(input_latents, t)

            input_latents = torch.cat([input_latents, mask_concat, masked_latent_concat], dim=1)

            noise_pred = self.unet(
                input_latents,
                t.to(self.device),
                encoder_hidden_states=None,
                return_dict=False,
            )[0]

            if do_classifier_free_guidance:
                noise_pred_unc, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_unc + scale * (noise_pred_text - noise_pred_unc)

            latents = self.noise_scheduler.step(noise_pred, t, latents, **extra_step).prev_sample

        latents = latents.split(latents.shape[self.concat_dim] // 2, dim=self.concat_dim)[0]
        latents = 1 / self.vae.config.scaling_factor * latents
        print(f"Latents before decoding - min: {latents.min()}, max: {latents.max()}")
        image = self.vae.decode(latents.to(self.device, dtype=self.dtype)).sample
        print(f"Decoded Image - min: {image.min()}, max: {image.max()}")
        image = (image / 2 + 0.5).clamp(0, 1)
        print(f"Final Image - min: {image.min()}, max: {image.max()}")
        image = image.permute(0, 2, 3, 1)
        return image
