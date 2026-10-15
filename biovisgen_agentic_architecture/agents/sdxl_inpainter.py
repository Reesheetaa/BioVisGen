"""Shared SDXL inpainting backend for text removal and localized refinement."""
from __future__ import annotations

import os
import torch
import numpy as np
from PIL import Image
from diffusers import AutoPipelineForInpainting

from config import SDXL_INPAINTING_MODEL, SDXL_CHECKPOINT_PATH, DEVICE

class SDXLInpainter:
    def __init__(self):
        source = SDXL_CHECKPOINT_PATH if os.path.exists(SDXL_CHECKPOINT_PATH) else SDXL_INPAINTING_MODEL
        dtype = torch.float16 if DEVICE == "cuda" and torch.cuda.is_available() else torch.float32
        self.device = "cuda" if DEVICE == "cuda" and torch.cuda.is_available() else "cpu"
        kwargs = {"torch_dtype": dtype}
        if dtype == torch.float16:
            kwargs["variant"] = "fp16"
        self.pipe = AutoPipelineForInpainting.from_pretrained(source, **kwargs)
        self.pipe.to(self.device)
        try:
            self.pipe.vae.enable_tiling()
        except Exception:
            pass
        if self.device == "cuda":
            try:
                self.pipe.enable_attention_slicing()
            except Exception:
                pass

    @staticmethod
    def _prepare(image: Image.Image, mask: Image.Image):
        image = image.convert("RGB")
        mask = mask.convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
        w, h = image.size
        nw, nh = max(8, (w // 8) * 8), max(8, (h // 8) * 8)
        if (nw, nh) != (w, h):
            image = image.resize((nw, nh), Image.Resampling.LANCZOS)
            mask = mask.resize((nw, nh), Image.Resampling.NEAREST)
        mask = mask.point(lambda p: 255 if p > 0 else 0)
        return image, mask, (w, h)

    def inpaint(self, image: Image.Image, mask: Image.Image, prompt: str,
                negative_prompt: str = "", strength: float = 0.99,
                guidance_scale: float = 7.5, steps: int = 30) -> Image.Image:
        image, mask, original_size = self._prepare(image, mask)
        generator = torch.Generator(device=self.device).manual_seed(42)
        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                image=image,
                mask_image=mask,
                strength=float(strength),
                guidance_scale=float(guidance_scale),
                num_inference_steps=int(steps),
                generator=generator,
            ).images[0]
        if result.size != original_size:
            result = result.resize(original_size, Image.Resampling.LANCZOS)
        return result
