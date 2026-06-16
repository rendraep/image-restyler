"""
Stable Diffusion XL img2img style transformation service.

Priority order:
  1. Local diffusers SDXL model  (set USE_LOCAL_SD=true, requires torch + diffusers)
  2. PIL fallback                (always available, applies a basic color tint)
"""

import io
import os

from PIL import Image, ImageFilter, ImageEnhance

_SD_MODEL = os.getenv("SD_MODEL", "Lykon/dreamshaper-xl-1-0")
_MAX_DIM = 1024  # SDXL native resolution

_pipe = None


def _load_pipe():
    global _pipe
    if _pipe is not None:
        return _pipe

    from diffusers import StableDiffusionXLImg2ImgPipeline
    import torch

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    _pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        _SD_MODEL,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if torch.cuda.is_available() else None,
    )

    if torch.cuda.is_available():
        _pipe.enable_model_cpu_offload()
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        _pipe = _pipe.to("mps")

    return _pipe


def _resize_for_sdxl(image: Image.Image) -> Image.Image:
    w, h = image.size
    ratio = min(_MAX_DIM / w, _MAX_DIM / h, 1.0)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    # SDXL requires multiples of 64
    new_w = max(64, new_w - new_w % 64)
    new_h = max(64, new_h - new_h % 64)
    return image.resize((new_w, new_h), Image.LANCZOS)


def _to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _pil_fallback(image_data: bytes, style_name: str) -> bytes:
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = image.filter(ImageFilter.SMOOTH_MORE)
    image = ImageEnhance.Color(image).enhance(1.6)
    image = ImageEnhance.Contrast(image).enhance(1.2)
    return _to_png_bytes(image)


async def transform_image(
    image_data: bytes,
    sd_prompt: str,
    negative_prompt: str = "",
    strength: float = 0.38,
    style_name: str = "",
) -> bytes:
    """
    Apply a style transformation to image_data using SDXL img2img.
    Falls back to PIL if diffusers is unavailable.
    """
    use_local = os.getenv("USE_LOCAL_SD", "false").lower() == "true"

    if use_local:
        try:
            pipe = _load_pipe()
            original = Image.open(io.BytesIO(image_data)).convert("RGB")
            orig_size = original.size
            resized = _resize_for_sdxl(original)

            import torch
            torch.cuda.empty_cache()

            result = pipe(
                prompt=sd_prompt,
                negative_prompt=negative_prompt or "blurry, low quality, deformed, watermark",
                image=resized,
                strength=strength,
                guidance_scale=7.5,
                num_inference_steps=30,
            ).images[0]

            result = result.resize(orig_size, Image.LANCZOS)
            return _to_png_bytes(result)

        except Exception as exc:
            raise RuntimeError(f"SDXL inference failed: {exc}") from exc

    return _pil_fallback(image_data, style_name)
