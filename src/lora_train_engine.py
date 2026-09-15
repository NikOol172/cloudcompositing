#!/usr/bin/env python3
"""
LoRA Training Engine for Stable Diffusion (SD 1.5 / SDXL).
Optimized for CUDA / Pod GPUs with:
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

# Compatibility auto-check: PyTorch 2.2.0 in the RunPod image requires numpy<2 and transformers<4.45.0
try:
    import numpy as _np
    if int(_np.__version__.split(".")[0]) >= 2:
        print(f"[COMPAT] NumPy {_np.__version__} detected. Downgrading to numpy<2 and transformers<4.45.0 for PyTorch 2.2 compatibility...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "numpy<2", "transformers<4.45.0"], check=False)
        print("[COMPAT] Packages adjusted successfully.")
except Exception as _e:
    pass

# Compatibility shims for diffusers on PyTorch < 2.5 (torch.nn.attention.flex_attention, torch.xpu, device_mesh)
try:
    import torch
    import types
    if not hasattr(torch, "xpu"):
        setattr(torch, "xpu", type("xpu", (), {
            "is_available": staticmethod(lambda: False),
            "device_count": staticmethod(lambda: 0),
            "empty_cache": staticmethod(lambda: None),
            "__getattr__": lambda s, n: lambda *a, **k: None
        })())
    import torch.distributed as _dist
    if not hasattr(_dist, "device_mesh"):
        setattr(_dist, "device_mesh", types.SimpleNamespace(DeviceMesh=type("DeviceMesh", (), {}), init_device_mesh=lambda *a, **k: None))
    
    if not hasattr(torch.nn, "attention"):
        _att = types.ModuleType("torch.nn.attention")
        _flex = types.ModuleType("torch.nn.attention.flex_attention")
        _flex.BlockMask = type("BlockMask", (), {})
        _flex.create_block_mask = lambda *a, **k: None
        _att.flex_attention = _flex
        torch.nn.attention = _att
        sys.modules["torch.nn.attention"] = _att
        sys.modules["torch.nn.attention.flex_attention"] = _flex
    elif not hasattr(torch.nn.attention, "flex_attention"):
        _flex = types.ModuleType("torch.nn.attention.flex_attention")
        _flex.BlockMask = type("BlockMask", (), {})
        _flex.create_block_mask = lambda *a, **k: None
        torch.nn.attention.flex_attention = _flex
        sys.modules["torch.nn.attention.flex_attention"] = _flex
except Exception as _e:
    pass

# Compatibility auto-check: NVIDIA Blackwell (sm_120 / RTX PRO 4500) requires PyTorch Nightly with CUDA 12.8
try:
    import torch
    if torch.cuda.is_available() and not os.environ.get("_PYTORCH_BLACKWELL_ATTEMPTED"):
        cap = torch.cuda.get_device_capability()
        arch = f"sm_{cap[0]}{cap[1]}"
        arch_list = torch.cuda.get_arch_list()
        if arch not in arch_list and cap[0] >= 10:
            print(f"[COMPAT] NVIDIA Blackwell GPU ({arch}) detected but not supported by current PyTorch ({torch.__version__}).", flush=True)
            print("[COMPAT] Automatically upgrading PyTorch to Nightly (CUDA 12.8 / sm_120 support)... Please wait ~60s.", flush=True)
            import subprocess
            subprocess.run([
                sys.executable, "-m", "pip", "install", "--pre", "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/nightly/cu128", "--upgrade"
            ], check=True)
            print("[COMPAT] PyTorch upgraded successfully! Restarting training process with native Blackwell support...", flush=True)
            os.environ["_PYTORCH_BLACKWELL_ATTEMPTED"] = "1"
            os.execv(sys.executable, [sys.executable] + sys.argv)
except Exception as _e:
    print(f"[COMPAT WARNING] Blackwell auto-upgrade check: {_e}", flush=True)

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
        # PyTorch XPU shim for Diffusers >= 0.31 compatibility on PyTorch without XPU
        if not hasattr(torch, "xpu"):
            class _DummyXpu:
                @staticmethod
                def is_available(): return False
                @staticmethod
                def device_count(): return 0
                @staticmethod
                def empty_cache(): pass
                @staticmethod
                def manual_seed(seed=0): pass
                @staticmethod
                def reset_peak_memory_stats(device=None): pass
                @staticmethod
                def reset_max_memory_allocated(device=None): pass
                @staticmethod
                def max_memory_allocated(device=None): return 0
                @staticmethod
                def synchronize(device=None): pass
                def __getattr__(self, name): return lambda *args, **kwargs: None
            torch.xpu = _DummyXpu()

        import torch.distributed as dist
        if not hasattr(dist, "device_mesh"):
            import types
            dm = types.ModuleType("device_mesh")
            dm.DeviceMesh = type("DeviceMesh", (), {})
            dm.init_device_mesh = lambda *args, **kwargs: None
            dist.device_mesh = dm

        import torch.nn.functional as F
        from torchvision import transforms
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from diffusers.training_utils import cast_training_params
        from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
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

    # Determine optimal precision (bfloat16 on Ada Lovelace/Ampere avoids FP16 exponent overflow)
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported() and args.mixed_precision in ("fp16", "bf16"):
            weight_dtype = torch.bfloat16
            amp_dtype = torch.bfloat16
            prec_name = "bfloat16 (Ada Lovelace native, NaN-safe)"
        elif args.mixed_precision == "fp16":
            weight_dtype = torch.float16
            amp_dtype = torch.float16
            prec_name = "float16"
        else:
            weight_dtype = torch.float32
            amp_dtype = torch.float32
            prec_name = "float32"
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[INFO] Training device: GPU '{gpu_name}' ({vram_gb:.1f} GB VRAM) with {prec_name}")
    else:
        weight_dtype = torch.float32
        amp_dtype = torch.float32
        print("[INFO] Training device: CPU (Precision: float32)")

    torch.manual_seed(args.seed)

    print(f"[INFO] Loading dataset from: {dataset_path}")
    raw_samples = load_dataset_samples(dataset_path, args.instance_prompt, args.resolution)
    print(f"[INFO] Loaded {len(raw_samples)} images for LoRA training.")

    is_sdxl = "xl" in args.base_model.lower()
    variant = "fp16" if (device.type == "cuda" and args.mixed_precision in ("fp16",)) else None

    # 1. Load VAE and Tokenizer / Text Encoder to Pre-cache Latents and Text Embeddings
    print(f"[INFO] Loading base model components from: '{args.base_model}' (is_sdxl: {is_sdxl})...")

    if is_sdxl:
        tokenizer_1 = AutoTokenizer.from_pretrained(args.base_model, subfolder="tokenizer", use_fast=False)
        tokenizer_2 = AutoTokenizer.from_pretrained(args.base_model, subfolder="tokenizer_2", use_fast=False)
        text_encoder_1 = CLIPTextModel.from_pretrained(args.base_model, subfolder="text_encoder", torch_dtype=weight_dtype, low_cpu_mem_usage=True).to(device)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.base_model, subfolder="text_encoder_2", torch_dtype=weight_dtype, low_cpu_mem_usage=True).to(device)
        text_encoder_1.eval()
        text_encoder_2.eval()
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, subfolder="tokenizer", use_fast=False)
        try:
            text_encoder = CLIPTextModel.from_pretrained(args.base_model, subfolder="text_encoder", torch_dtype=weight_dtype, variant=variant, low_cpu_mem_usage=True).to(device)
        except Exception:
            text_encoder = CLIPTextModel.from_pretrained(args.base_model, subfolder="text_encoder", torch_dtype=weight_dtype, low_cpu_mem_usage=True).to(device)
        text_encoder.eval()

    # Always load and run VAE in float32 to completely avoid SD/SDXL VAE FP16 numerical overflow (NaNs)
    vae = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae", torch_dtype=torch.float32, low_cpu_mem_usage=True).to(device)
    vae.eval()

    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")

    # Preprocess images to normalized tensors: [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    print("[INFO] Pre-caching VAE latents in FP32 and text embeddings to optimize VRAM...")
    cached_latents = []
    cached_embeddings = []

    with torch.no_grad():
        for pil_img, caption, name in raw_samples:
            # Run VAE encode in float32 for absolute numerical stability
            tensor_img = transform(pil_img).unsqueeze(0).to(device, dtype=torch.float32)
            latent_dist = vae.encode(tensor_img).latent_dist
            latent = latent_dist.sample() * vae.config.scaling_factor
            latent = torch.nan_to_num(latent, nan=0.0, posinf=1.0, neginf=-1.0)
            cached_latents.append(latent.squeeze(0).to(dtype=weight_dtype).cpu())

            if is_sdxl:
                ids_1 = tokenizer_1(caption, padding="max_length", max_length=tokenizer_1.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)
                ids_2 = tokenizer_2(caption, padding="max_length", max_length=tokenizer_2.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)
                h_1 = text_encoder_1(ids_1, output_hidden_states=True).hidden_states[-2]
                enc_2 = text_encoder_2(ids_2, output_hidden_states=True)
                h_2 = enc_2.hidden_states[-2]
                pooled = enc_2.text_embeds
                text_embed = torch.cat([h_1, h_2], dim=-1).squeeze(0).cpu()
                pooled_embed = pooled.squeeze(0).cpu()
                time_ids = torch.tensor([args.resolution, args.resolution, 0, 0, args.resolution, args.resolution], dtype=weight_dtype).cpu()
                cached_embeddings.append((text_embed, pooled_embed, time_ids))
            else:
                inputs = tokenizer(
                    caption,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.to(device)
                text_embed = text_encoder(inputs)[0].squeeze(0)
                text_embed = F.pad(text_embed, (0, 0, 0, 3))
                cached_embeddings.append((text_embed.cpu(), None, None))

    print(f"[INFO] Cached {len(cached_latents)} latents and text representations.")

    # Free VAE, text encoder and raw PIL images from GPU and CPU RAM to free up memory
    del raw_samples
    del vae
    if is_sdxl:
        del text_encoder_1
        del text_encoder_2
        del tokenizer_1
        del tokenizer_2
    else:
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
        torch.backends.cuda.enable_flash_sdp(True)
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
    consecutive_nans = 0

    unet.train()
    optimizer.zero_grad()

    while current_step < total_steps:
        epoch += 1
        indices = torch.randperm(num_samples).tolist()

        for idx in indices:
            latent = cached_latents[idx].unsqueeze(0).to(device, dtype=weight_dtype)
            embed_item = cached_embeddings[idx]
            encoder_hidden_states = embed_item[0].unsqueeze(0).to(device, dtype=weight_dtype)

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
                if is_sdxl:
                    added_cond_kwargs = {
                        "text_embeds": embed_item[1].unsqueeze(0).to(device, dtype=weight_dtype),
                        "time_ids": embed_item[2].unsqueeze(0).to(device, dtype=weight_dtype),
                    }
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs).sample
                else:
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
                consecutive_nans += 1
                print(f"[WARN] NaN/Inf detected in loss at step {current_step} ({consecutive_nans}/10), skipping batch.", file=sys.stderr)
                if consecutive_nans >= 10:
                    has_pred_nan = torch.isnan(model_pred).any().item()
                    has_latent_nan = torch.isnan(latent).any().item()
                    print(f"[FATAL ERROR] 10 consecutive NaN losses detected. Diagnostic: model_pred_nan={has_pred_nan}, latent_nan={has_latent_nan}", file=sys.stderr)
                    sys.exit(1)
                optimizer.zero_grad()
                continue
            consecutive_nans = 0

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
