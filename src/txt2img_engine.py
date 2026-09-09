#!/usr/bin/env python3
"""
High-Performance Local Text-to-Image Engine using PyTorch & HuggingFace Diffusers.
Optimized for CUDA / Pod GPUs with FP16, VAE slicing, and CPU offload support.
"""

import os
import sys
import site

# Configure caches on Drive D: to prevent filling Drive C:
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HOME"] = os.path.join(base_dir, ".hf_cache")
os.environ["HF_HUB_CACHE"] = os.path.join(base_dir, ".hf_cache", "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(base_dir, ".hf_cache", "hub")
os.environ["DIFFUSERS_CACHE"] = os.path.join(base_dir, ".hf_cache", "diffusers")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(base_dir, ".hf_cache", "transformers")
os.environ["TORCH_HOME"] = os.path.join(base_dir, ".torch_cache")

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

def parse_args():
    parser = argparse.ArgumentParser(description="Local Diffusers Text-to-Image Generation Engine")
    parser.add_argument("--prompt", required=True, help="Text prompt for image generation")
    parser.add_argument("--output", required=True, help="Destination path for the generated image PNG/JPG")
    parser.add_argument(
        "--model",
        default="stabilityai/sdxl-turbo",
        help="HuggingFace model ID or local path (e.g. stabilityai/sdxl-turbo, ByteDance/SDXL-Lightning, runwayml/stable-diffusion-v1-5)",
    )
    parser.add_argument("--width", type=int, default=512, help="Output image width (default: 512)")
    parser.add_argument("--height", type=int, default=512, help="Output image height (default: 512)")
    parser.add_argument("--steps", type=int, default=0, help="Inference steps (0 = auto based on model)")
    parser.add_argument("--guidance-scale", type=float, default=-1.0, help="Guidance scale (-1.0 = auto)")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 = random)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Compute device (auto, cuda, cpu)")

    # LoRA options
    parser.add_argument("--lora", default=None, help="Optional path to a LoRA .safetensors file or HF repo")
    parser.add_argument("--lora-scale", type=float, default=0.8, help="LoRA influence weight (default: 0.8)")

    # ControlNet options
    parser.add_argument("--controlnet-image", default=None, help="Guidance source image path for ControlNet")
    parser.add_argument("--controlnet-type", choices=["canny", "depth", "pose"], default="canny", help="ControlNet conditioning type")
    parser.add_argument("--controlnet-scale", type=float, default=0.8, help="ControlNet conditioning scale (default: 0.8)")
    parser.add_argument("--controlnet-save-edge", default=None, help="Optional path to save preprocessed control edge map")
    return parser.parse_args()

def main():
    args = parse_args()

    try:
        import torch
        from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
    except ImportError:
        print(
            "[ERROR] PyTorch or Diffusers is not installed.\n"
            "Please run: pip install -r requirements-local-gpu.txt\n"
            "Or: pip install torch diffusers transformers accelerate safetensors",
            file=sys.stderr,
        )
        sys.exit(1)

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

    # Determine optimal parameters based on model type
    model_lower = args.model.lower()
    is_turbo = "turbo" in model_lower
    is_lightning = "lightning" in model_lower
    is_sdxl = "sdxl" in model_lower or is_turbo or is_lightning

    steps = args.steps
    if steps <= 0:
        if is_turbo:
            steps = 2
        elif is_lightning:
            steps = 4
        else:
            steps = 25

    guidance_scale = args.guidance_scale
    if guidance_scale < 0:
        if is_turbo:
            guidance_scale = 0.0  # SDXL-Turbo does not use CFG guidance
        elif is_lightning:
            guidance_scale = 1.0
        else:
            guidance_scale = 7.5

    # Target resolution adjustment
    width = args.width
    height = args.height
    if is_sdxl and width <= 512 and height <= 512 and not is_turbo:
        # Default SDXL base works best around 1024x1024, but Turbo can do 512x512
        width = 1024
        height = 1024

    # Seed
    if args.seed >= 0:
        seed = args.seed
    else:
        seed = int(time.time() * 1000) % (2**31)

    generator = torch.Generator(device=device).manual_seed(seed)

    # Preprocess ControlNet image if supplied
    has_controlnet = bool(args.controlnet_image and os.path.exists(args.controlnet_image))
    control_image = None
    controlnet = None

    if has_controlnet:
        import cv2
        import numpy as np
        from PIL import Image
        from diffusers import ControlNetModel

        print(f"[INFO] Initializing ControlNet (Type: '{args.controlnet_type}', Scale: {args.controlnet_scale})...")
        raw_control = Image.open(args.controlnet_image).convert("RGB")
        raw_control = raw_control.resize((width, height), Image.Resampling.LANCZOS)
        np_control = np.array(raw_control)

        if args.controlnet_type == "canny":
            print("[INFO] Computing Canny edges from control image...")
            canny_edges = cv2.Canny(np_control, 100, 200)
            canny_edges = np.stack([canny_edges] * 3, axis=2)
            control_image = Image.fromarray(canny_edges)
            control_model_id = "diffusers/controlnet-canny-sdxl-1.0"
        elif args.controlnet_type == "depth":
            print("[INFO] Using ControlNet Depth model...")
            control_image = raw_control
            control_model_id = "diffusers/controlnet-depth-sdxl-1.0"
        else:
            control_image = raw_control
            control_model_id = "diffusers/controlnet-canny-sdxl-1.0"

        if args.controlnet_save_edge:
            try:
                control_image.save(args.controlnet_save_edge)
                print(f"[INFO] Control map saved to: {args.controlnet_save_edge}")
            except Exception as e:
                print(f"[WARN] Could not save control map: {e}")

        print(f"[INFO] Loading ControlNet model weights: '{control_model_id}'...")
        controlnet = ControlNetModel.from_pretrained(
            control_model_id,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
            use_safetensors=True,
        )

    print(f"[INFO] Loading base model '{args.model}'...")
    start_load = time.time()

    if has_controlnet:
        from diffusers import StableDiffusionXLControlNetPipeline

        base_sdxl = "stabilityai/stable-diffusion-xl-base-1.0" if not is_sdxl else args.model
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_sdxl,
            controlnet=controlnet,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
            use_safetensors=True,
        )
    elif is_lightning and ("bytedance" in model_lower or args.model == "ByteDance/SDXL-Lightning"):
        from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        base = "stabilityai/stable-diffusion-xl-base-1.0"
        step_str = f"{steps}step" if steps in [2, 4, 8] else "4step"
        ckpt_name = f"sdxl_lightning_{step_str}_unet.safetensors"
        print(f"[INFO] Loading SDXL-Lightning UNet '{ckpt_name}' on top of '{base}'...")

        unet_config = UNet2DConditionModel.load_config(base, subfolder="unet")
        unet = UNet2DConditionModel.from_config(unet_config).to(dtype=dtype)
        ckpt_file = hf_hub_download("ByteDance/SDXL-Lightning", ckpt_name)
        state_dict = load_file(ckpt_file, device="cpu")
        unet.load_state_dict(state_dict)
        del state_dict

        pipe = StableDiffusionXLPipeline.from_pretrained(
            base,
            unet=unet,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
            use_safetensors=True,
        )
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    else:
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(
                args.model,
                torch_dtype=dtype,
                variant="fp16" if dtype == torch.float16 else None,
                use_safetensors=True,
            )
        except Exception as e:
            print(f"[WARN] Standard load failed ({e}), attempting fallback load without variant='fp16'...", file=sys.stderr)
            pipe = AutoPipelineForText2Image.from_pretrained(
                args.model,
                torch_dtype=dtype,
                use_safetensors=True,
            )

    # Load custom LoRA if provided
    if args.lora:
        lora_target = args.lora.strip()
        if os.path.exists(lora_target):
            print(f"[INFO] Loading custom LoRA weights from '{lora_target}' (Scale: {args.lora_scale})...")
            try:
                if os.path.isfile(lora_target):
                    lora_dir = os.path.dirname(os.path.abspath(lora_target))
                    lora_file = os.path.basename(lora_target)
                    pipe.load_lora_weights(lora_dir, weight_name=lora_file)
                else:
                    pipe.load_lora_weights(lora_target)

                if hasattr(pipe, "fuse_lora"):
                    pipe.fuse_lora(lora_scale=args.lora_scale)
                print("[INFO] LoRA weights successfully merged into pipeline.")
            except Exception as e:
                print(f"[WARN] Failed to load LoRA weights: {e}", file=sys.stderr)
        else:
            print(f"[WARN] Specified LoRA file '{lora_target}' was not found.", file=sys.stderr)

    # Memory optimizations for CUDA GPUs
    if device == "cuda":
        try:
            pipe.enable_model_cpu_offload()
            print("[INFO] Model CPU offloading active (optimal for 8GB VRAM).")
        except Exception:
            pipe.to("cuda")

        try:
            pipe.enable_vae_slicing()
            pipe.enable_vae_tiling()
        except Exception:
            pass
    else:
        pipe.to("cpu")

    load_time = time.time() - start_load
    print(f"[INFO] Model loaded in {load_time:.2f}s.")
    print(f"[INFO] Generating image: {width}x{height}, {steps} steps, CFG: {guidance_scale}, Seed: {seed}")
    if has_controlnet:
        print(f"[INFO] ControlNet Active: Type={args.controlnet_type}, Scale={args.controlnet_scale}")
    print(f"[INFO] Prompt: '{args.prompt}'")

    # Step callback for streaming progress
    def progress_callback(pipe_obj, step_index, timestep, callback_kwargs):
        pct = ((step_index + 1) / steps) * 100
        print(f"[PROGRESS] Step {step_index + 1}/{steps} ({pct:.0f}%)", flush=True)
        return callback_kwargs

    start_gen = time.time()
    
    # Run inference with or without ControlNet
    if has_controlnet and control_image is not None:
        image = pipe(
            prompt=args.prompt,
            image=control_image,
            controlnet_conditioning_scale=args.controlnet_scale,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            callback_on_step_end=progress_callback,
        ).images[0]
    else:
        image = pipe(
            prompt=args.prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            callback_on_step_end=progress_callback,
        ).images[0]

    gen_time = time.time() - start_gen
    print(f"[INFO] Generation completed in {gen_time:.2f}s.")

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    image.save(args.output)
    print(f"[SUCCESS] Image saved successfully to: {args.output}")

if __name__ == "__main__":
    main()
