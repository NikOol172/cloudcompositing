#!/usr/bin/env python3
"""
High-Performance Local Image-to-Image & Inpainting Engine using PyTorch & HuggingFace Diffusers.
Optimized for 8GB VRAM GPUs (e.g. RTX 2080) with FP16, VAE slicing, and CPU offload support.
"""

import os
import sys
import site

# Ensure all user site-packages and system dist-packages are in sys.path
for p in [
    site.getusersitepackages(),
    os.path.expanduser("~/.local/lib/python3.10/site-packages"),
    os.path.expanduser("~/.local/lib/python3.11/site-packages"),
    os.path.expanduser("~/.local/lib/python3.12/site-packages"),
    "/usr/local/lib/python3.10/dist-packages",
    "/usr/lib/python3/dist-packages",
    "/usr/lib/python3.10",
]:
    if p and os.path.exists(p) and p not in sys.path:
        sys.path.append(p)

import argparse
import time
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Local Diffusers Image-to-Image & Inpainting Generation Engine")
    parser.add_argument("--image", required=True, help="Input source image file path")
    parser.add_argument("--mask", default=None, help="Optional binary mask image file path for Inpainting")
    parser.add_argument("--prompt", required=True, help="Text prompt for image transformation or inpainting")
    parser.add_argument("--output", required=True, help="Destination path for the generated image PNG/JPG")
    parser.add_argument(
        "--model",
        default="stabilityai/sdxl-turbo",
        help="HuggingFace model ID or local path (e.g. stabilityai/sdxl-turbo, ByteDance/SDXL-Lightning, runwayml/stable-diffusion-v1-5)",
    )
    parser.add_argument("--strength", type=float, default=0.70, help="Transformation strength / denoise (0.0 to 1.0, default: 0.70)")
    parser.add_argument("--width", type=int, default=0, help="Output image width (0 = maintain/auto based on image)")
    parser.add_argument("--height", type=int, default=0, help="Output image height (0 = maintain/auto based on image)")
    parser.add_argument("--steps", type=int, default=0, help="Inference steps (0 = auto based on model)")
    parser.add_argument("--guidance-scale", type=float, default=-1.0, help="Guidance scale (-1.0 = auto)")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 = random)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Compute device (auto, cuda, cpu)")
    return parser.parse_args()

def main():
    args = parse_args()

    try:
        import torch
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForInpainting
    except ImportError:
        print(
            "[ERROR] PyTorch or Diffusers is not installed.\n"
            "Please run: pip install -r requirements-local-gpu.txt\n"
            "Or: pip install torch diffusers transformers accelerate safetensors pillow",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.exists(args.image):
        print(f"[ERROR] Source image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    has_mask = bool(args.mask and os.path.exists(args.mask))

    # Determine compute device
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("[WARN] CUDA requested but not available. Falling back to CPU.", file=sys.stderr)
            device = "cpu"
        else:
            device = "cuda"
    elif args.device == "cpu":
        device = "cpu"
    else:  # auto
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[INFO] Using GPU acceleration on '{gpu_name}' ({vram_gb:.1f} GB VRAM) with FP16 precision.")
        dtype = torch.float16
    else:
        print("[INFO] Running on CPU with FP32 precision (for best speed, run on a machine with CUDA GPU).")
        dtype = torch.float32

    # Load and prepare source image
    init_image = Image.open(args.image).convert("RGB")
    orig_w, orig_h = init_image.size

    target_w = args.width if args.width > 0 else orig_w
    target_h = args.height if args.height > 0 else orig_h

    # Ensure multiple of 8
    target_w = (target_w // 8) * 8
    target_h = (target_h // 8) * 8

    # Limit maximum dimension for 8GB VRAM safety
    max_dim = 1024 if device == "cuda" else 512
    if target_w > max_dim or target_h > max_dim:
        ratio = min(max_dim / target_w, max_dim / target_h)
        target_w = (int(target_w * ratio) // 8) * 8
        target_h = (int(target_h * ratio) // 8) * 8

    init_image = init_image.resize((target_w, target_h), Image.LANCZOS)

    mask_image = None
    if has_mask:
        print(f"[INFO] Mask provided: {args.mask}. Loading in Inpainting mode.")
        mask_image = Image.open(args.mask).convert("RGB").resize((target_w, target_h), Image.NEAREST)

    # Determine optimal parameters based on model type
    model_lower = args.model.lower()
    is_turbo = "turbo" in model_lower
    is_lightning = "lightning" in model_lower

    steps = args.steps
    if steps <= 0:
        if is_turbo:
            steps = 4
        elif is_lightning:
            steps = 6
        else:
            steps = 25

    guidance_scale = args.guidance_scale
    if guidance_scale < 0:
        if is_turbo:
            guidance_scale = 0.0  # SDXL-Turbo doesn't use CFG guidance
        elif is_lightning:
            guidance_scale = 1.5
        else:
            guidance_scale = 7.5

    strength = max(0.05, min(1.0, args.strength))

    # Seed
    if args.seed >= 0:
        seed = args.seed
    else:
        seed = int(time.time() * 1000) % (2**31)

    generator = torch.Generator(device=device).manual_seed(seed)

    mode_label = "Inpainting (Masked)" if has_mask else "Image-to-Image"
    print(f"[INFO] Loading model '{args.model}' for {mode_label}...")
    start_load = time.time()

    pipe_class = AutoPipelineForInpainting if has_mask else AutoPipelineForImage2Image

    try:
        pipe = pipe_class.from_pretrained(
            args.model,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
            use_safetensors=True,
        )
    except Exception as e:
        print(f"[WARN] Standard load failed ({e}), attempting fallback load without variant='fp16'...", file=sys.stderr)
        pipe = pipe_class.from_pretrained(
            args.model,
            torch_dtype=dtype,
            use_safetensors=True,
        )

    # Memory optimizations for 8GB VRAM GPUs (RTX 2080)
    if device == "cuda":
        pipe.to("cuda")
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
    else:
        pipe.to("cpu")

    load_time = time.time() - start_load
    print(f"[INFO] Model loaded in {load_time:.2f}s.")
    print(f"[INFO] Processing: {target_w}x{target_h}, strength: {strength}, {steps} steps, CFG: {guidance_scale}, Seed: {seed}")
    print(f"[INFO] Prompt: '{args.prompt}'")

    # Step callback for streaming progress
    def progress_callback(pipe_obj, step_index, timestep, callback_kwargs):
        pct = ((step_index + 1) / max(1, steps)) * 100
        print(f"[PROGRESS] Step {step_index + 1}/{steps} ({pct:.0f}%)", flush=True)
        return callback_kwargs

    start_gen = time.time()

    # Run inference
    call_kwargs = {
        "prompt": args.prompt,
        "image": init_image,
        "strength": strength,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
        "callback_on_step_end": progress_callback,
    }

    if has_mask and mask_image is not None:
        call_kwargs["mask_image"] = mask_image

    image = pipe(**call_kwargs).images[0]

    gen_time = time.time() - start_gen
    print(f"[INFO] Transformation completed in {gen_time:.2f}s.")

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    image.save(args.output)
    print(f"[SUCCESS] Result saved successfully to: {args.output}")

if __name__ == "__main__":
    main()
