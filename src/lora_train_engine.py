#!/usr/bin/env python3
"""
LoRA Training Engine for Stable Diffusion (SD 1.5 / SDXL).
Optimized for 8GB VRAM (NVIDIA RTX 2080) with:
- Pre-cached VAE latents & pre-encoded text embeddings
- LoRA adapters on UNet attention layers (PEFT / Diffusers)
- FP16 mixed precision and gradient accumulation
- Real-time streaming progress logs: [PROGRESS] Step X/Y (Z%), Loss: L, Epoch: E
- Direct export to .safetensors in loras/
"""

import os
import sys
import gc

# Ensure unbuffered and line-buffered I/O on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

import argparse
import time
import math
from pathlib import Path
from PIL import Image

# Ensure project local packages & caches are configured on Drive D:
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HOME"] = os.path.join(workspace_root, ".hf_cache")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(workspace_root, ".hf_cache", "hub")
os.environ["TORCH_HOME"] = os.path.join(workspace_root, ".torch_cache")
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["SAFETENSORS_BACKEND"] = "pread"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA adapters locally on GPU")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing training images and .txt caption files")
    parser.add_argument("--instance-prompt", default="a photo of sks person", help="Fallback prompt if .txt file not found")
    parser.add_argument("--output-name", required=True, help="Filename of output LoRA, e.g. 'mon_lora.safetensors'")
    parser.add_argument("--output-dir", default="loras", help="Destination folder for .safetensors")
    parser.add_argument("--base-model", default="runwayml/stable-diffusion-v1-5", help="Base model identifier (SD1.5 or SDXL)")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution for training (512 for SD1.5, 768/1024 for SDXL)")
    parser.add_argument("--train-steps", type=int, default=500, help="Total training steps")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for AdamW")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank dimension (4, 8, 16, 32)")
    parser.add_argument("--lora-alpha", type=int, default=8, help="LoRA alpha scaling factor")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Number of gradient accumulation steps")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--mixed-precision", default="fp16", choices=["fp16", "bf16", "no"], help="Mixed precision mode (default: fp16 for optimal 2.75GB VRAM and maximum speed)")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing (not recommended on Windows CUDA)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"], help="Device to use")
    return parser.parse_args()

def load_dataset_samples(dataset_dir: Path, fallback_prompt: str, resolution: int):
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.PNG", "*.JPG", "*.JPEG", "*.WEBP")
    image_paths = []
    for ext in extensions:
        image_paths.extend(dataset_dir.glob(ext))
    image_paths = sorted(list(set(image_paths)))

    if not image_paths:
        raise ValueError(f"No valid images found in dataset directory '{dataset_dir}'")

    samples = []
    for img_path in image_paths:
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            caption = txt_path.read_text(encoding="utf-8").strip()
        else:
            caption = fallback_prompt

        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                # Center crop to square and resize
                w, h = im.size
                min_dim = min(w, h)
                left = (w - min_dim) // 2
                top = (h - min_dim) // 2
                im = im.crop((left, top, left + min_dim, top + min_dim))
                im = im.resize((resolution, resolution), Image.Resampling.BICUBIC)
                samples.append((im.copy(), caption, img_path.name))
        except Exception as e:
            print(f"[WARN] Skipping corrupted image {img_path.name}: {e}", file=sys.stderr)

    if not samples:
        raise ValueError(f"Could not load any valid images from '{dataset_dir}'")

    return samples

def main():
    args = parse_args()
    start_time = time.time()

    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        print(f"[ERROR] Dataset directory '{dataset_path}' not found.", file=sys.stderr)
        sys.exit(1)

    # Resolve output path
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_filename = args.output_name if args.output_name.endswith(".safetensors") else f"{args.output_name}.safetensors"
    final_output_path = out_dir / out_filename

    # Imports
    try:
        import torch
        import torch.nn.functional as F
        from torchvision import transforms
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from diffusers.training_utils import cast_training_params
        from transformers import AutoTokenizer, CLIPTextModel
        from peft import LoraConfig, get_peft_model
        from safetensors.torch import save_file
    except ImportError as e:
        print(f"[ERROR] Missing required Python package: {e}", file=sys.stderr)
        print("Please install requirements with: pip install torch diffusers transformers accelerate peft safetensors", file=sys.stderr)
        sys.exit(1)

    # Compute device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.", file=sys.stderr)
        device = torch.device("cpu")
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    weight_dtype = torch.float16 if (device.type == "cuda" and args.mixed_precision == "fp16") else torch.float32

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[INFO] Training device: GPU '{gpu_name}' ({vram_gb:.1f} GB VRAM) with {weight_dtype}")
    else:
        print(f"[INFO] Training device: CPU (Precision: {weight_dtype})")

    torch.manual_seed(args.seed)

    print(f"[INFO] Loading dataset from: {dataset_path}")
    raw_samples = load_dataset_samples(dataset_path, args.instance_prompt, args.resolution)
    print(f"[INFO] Loaded {len(raw_samples)} images for LoRA training.")

    # 1. Load VAE and Tokenizer / Text Encoder to Pre-cache Latents and Text Embeddings
    print(f"[INFO] Loading base model components from: '{args.base_model}'...")
    variant = "fp16" if (device.type == "cuda" and args.mixed_precision in ("fp16",)) else None

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, subfolder="tokenizer", use_fast=False)
    
    try:
        text_encoder = CLIPTextModel.from_pretrained(args.base_model, subfolder="text_encoder", torch_dtype=weight_dtype, variant=variant, low_cpu_mem_usage=True).to(device)
    except Exception:
        text_encoder = CLIPTextModel.from_pretrained(args.base_model, subfolder="text_encoder", torch_dtype=weight_dtype, low_cpu_mem_usage=True).to(device)

    try:
        vae = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae", torch_dtype=weight_dtype, variant=variant, low_cpu_mem_usage=True).to(device)
    except Exception:
        vae = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae", torch_dtype=weight_dtype, low_cpu_mem_usage=True).to(device)

    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")

    vae.eval()
    text_encoder.eval()

    # Preprocess images to normalized tensors: [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    print("[INFO] Pre-caching VAE latents and text embeddings to optimize VRAM...")
    cached_latents = []
    cached_embeddings = []

    with torch.no_grad():
        for pil_img, caption, name in raw_samples:
            tensor_img = transform(pil_img).unsqueeze(0).to(device, dtype=weight_dtype)
            # Encode image to latent space
            latent_dist = vae.encode(tensor_img).latent_dist
            latent = latent_dist.sample() * vae.config.scaling_factor
            cached_latents.append(latent.squeeze(0).cpu())

            # Encode caption to embedding
            inputs = tokenizer(
                caption,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(device)
            text_embed = text_encoder(inputs)[0].squeeze(0)
            # Pad text embedding sequence from 77 to 80 tokens (aligned to 16-byte boundaries)
            # This fixes hardware out-of-bounds access in CUTLASS MemEfficient attention on Turing GPUs (RTX 2080)
            text_embed = F.pad(text_embed, (0, 0, 0, 3))
            cached_embeddings.append(text_embed.cpu())

    print(f"[INFO] Cached {len(cached_latents)} latents and text representations.")

    # Free VAE, text encoder and raw PIL images from GPU and CPU RAM to free up memory
    del raw_samples
    del vae
    del text_encoder
    del tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 2. Load UNet and inject LoRA layers
    print(f"[INFO] Loading UNet and injecting LoRA adapters (Rank: {args.lora_rank}, Alpha: {args.lora_alpha})...", flush=True)
    try:
        print(f"[DEBUG] Loading UNet with variant='{variant}', low_cpu_mem_usage=True...", flush=True)
        unet = UNet2DConditionModel.from_pretrained(
            args.base_model,
            subfolder="unet",
            torch_dtype=weight_dtype,
            variant=variant,
            low_cpu_mem_usage=True,
        ).to(device)
        print("[DEBUG] UNet loaded with variant successfully!", flush=True)
    except Exception as e:
        print(f"[WARN] UNet variant='{variant}' failed ({e}). Loading without variant...", flush=True)
        unet = UNet2DConditionModel.from_pretrained(
            args.base_model,
            subfolder="unet",
            torch_dtype=weight_dtype,
            low_cpu_mem_usage=True,
        ).to(device)
    
    # Configure cuDNN and SDPA kernels for maximum speed and minimal memory
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

    # Enable gradient checkpointing to reduce VRAM from >9GB down to ~2.5GB during UNet backprop
    unet.enable_gradient_checkpointing()

    # Configure PEFT LoRA on UNet attention cross/self-attention projections
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)

    # Upcast trainable LoRA parameters to float32 safely using Diffusers utility (avoids invalid tensor storage mutations)
    cast_training_params(unet, dtype=torch.float32)

    unet.print_trainable_parameters()

    # Optimizer operating on float32 LoRA parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    use_amp = (device.type == "cuda" and args.mixed_precision in ("fp16", "bf16"))
    amp_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16

    total_steps = args.train_steps
    grad_accum = max(1, args.gradient_accumulation_steps)
    num_samples = len(cached_latents)
    epochs = math.ceil(total_steps / (num_samples / grad_accum))

    print(f"\n[INFO] Starting LoRA Fine-Tuning (AMP: {use_amp}, Dtype: {amp_dtype}):")
    print(f"  • Total Steps       : {total_steps}")
    print(f"  • Estimated Epochs  : ~{epochs}")
    print(f"  • Learning Rate     : {args.learning_rate}")
    print(f"  • Grad Accumulation : {grad_accum}")
    print(f"  • Target LoRA File  : {final_output_path.name}\n")

    current_step = 0
    epoch = 0
    running_loss = 0.0
    accumulated_steps = 0

    unet.train()
    optimizer.zero_grad()

    while current_step < total_steps:
        epoch += 1
        indices = torch.randperm(num_samples).tolist()

        for idx in indices:
            latent = cached_latents[idx].unsqueeze(0).to(device, dtype=weight_dtype)
            encoder_hidden_states = cached_embeddings[idx].unsqueeze(0).to(device, dtype=weight_dtype)

            # Sample noise to add to the latents
            noise = torch.randn_like(latent)
            # Sample random timesteps
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (1,), device=device
            ).long()

            # Add noise to latents (Forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latent, noise, timesteps)

            # High-speed Memory-Efficient mixed-precision forward pass
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                # Predict the noise residual with UNet + LoRA
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get target for loss calculation
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latent, noise, timesteps)
                else:
                    target = noise

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss = loss / grad_accum

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARN] NaN/Inf detected in loss at step {current_step}, skipping batch.", file=sys.stderr)
                optimizer.zero_grad()
                continue

            loss.backward()

            running_loss += loss.item() * grad_accum
            accumulated_steps += 1

            if accumulated_steps % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, unet.parameters()), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                current_step += 1
                pct = (current_step / total_steps) * 100.0
                avg_loss = running_loss / grad_accum
                running_loss = 0.0

                # Streaming progress log for UI and backend parsing
                print(f"[PROGRESS] Step {current_step}/{total_steps} ({pct:.1f}%), Loss: {avg_loss:.4f}, Epoch: {epoch}", flush=True)

                if current_step >= total_steps:
                    break

    # 3. Extract and save LoRA safetensors
    print(f"\n[INFO] Training complete in {time.time() - start_time:.1f}s! Extracting LoRA weights...")

    # Extract LoRA state dict
    lora_state_dict = {}
    for name, param in unet.named_parameters():
        if "lora" in name and param.requires_grad:
            # Format key compatible with Kohya/Diffusers LoRA loaders
            clean_name = name.replace("base_model.model.", "")
            lora_state_dict[clean_name] = param.detach().cpu().to(torch.float16)

    print(f"[INFO] Saving {len(lora_state_dict)} LoRA tensors to: {final_output_path}")
    save_file(lora_state_dict, str(final_output_path))

    file_size_mb = final_output_path.stat().st_size / (1024 * 1024)
    print(f"[SUCCESS] LoRA saved successfully: {final_output_path} ({file_size_mb:.2f} MB)", flush=True)

if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL ERROR] LoRA training crashed: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)
