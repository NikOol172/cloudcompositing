#!/usr/bin/env python3
"""
High-Performance Local LTX-Video Inference Engine using PyTorch & HuggingFace Diffusers.
Optimized for Pod GPUs with BF16/FP16, model CPU offload, and VAE tiling.
Supports both Text-to-Video and Image-to-Video via Lightricks/LTX-Video.
"""

import argparse
import math
import os
import sys
import time
from PIL import Image

# Compatibility shims for diffusers on PyTorch < 2.5 (torch.nn.attention.flex_attention, torch.xpu, device_mesh, RMSNorm)
try:
    import torch
    import types
    import builtins
    if not hasattr(torch, "xpu"):
        setattr(torch, "xpu", type("xpu", (), {
            "is_available": staticmethod(lambda: False),
            "device_count": staticmethod(lambda: 0),
            "empty_cache": staticmethod(lambda: None),
            "__getattr__": lambda s, n: lambda *a, **k: None
        })())

    if not hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals = lambda *args, **kwargs: None

    def _rms_norm_impl(x, normalized_shape, weight=None, eps=1e-6):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        if weight is not None:
            x = x * weight
        return x

    class _RMSNormImpl(torch.nn.Module):
        def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True, device=None, dtype=None):
            super().__init__()
            if isinstance(normalized_shape, builtins.int):
                normalized_shape = (normalized_shape,)
            self.normalized_shape = tuple(normalized_shape)
            self.eps = eps
            self.elementwise_affine = elementwise_affine
            if self.elementwise_affine:
                self.weight = torch.nn.Parameter(torch.empty(self.normalized_shape, device=device, dtype=dtype))
                torch.nn.init.ones_(self.weight)
            else:
                self.register_parameter("weight", None)
        def forward(self, x):
            return _rms_norm_impl(x, self.normalized_shape, self.weight, self.eps)

    if not hasattr(torch.nn.functional, "rms_norm"):
        torch.nn.functional.rms_norm = _rms_norm_impl
    if not hasattr(torch.nn, "RMSNorm"):
        torch.nn.RMSNorm = _RMSNormImpl

    # SDPA compatibility for PyTorch < 2.5 (enable_gqa argument)
    import torch.nn.functional as _F
    if not getattr(_F, "_sdpa_gqa_patched", False):
        _orig_sdpa = _F.scaled_dot_product_attention
        def _sdpa_compat(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
            if enable_gqa and query.shape[1] != key.shape[1]:
                group = query.shape[1] // key.shape[1]
                key = key.repeat_interleave(group, dim=1)
                value = value.repeat_interleave(group, dim=1)
            return _orig_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
        _F.scaled_dot_product_attention = _sdpa_compat
        _F._sdpa_gqa_patched = True

    try:
        import torch.distributed as _dist
    except Exception:
        _dist = types.ModuleType("torch.distributed")
        sys.modules["torch.distributed"] = _dist
        torch.distributed = _dist

    if not hasattr(_dist, "device_mesh"):
        _dm = types.ModuleType("torch.distributed.device_mesh")
        class DeviceMesh: pass
        _dm.DeviceMesh = DeviceMesh
        _dist.device_mesh = _dm
        sys.modules["torch.distributed.device_mesh"] = _dm

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
except Exception:
    pass

# Compatibility auto-check: transformers < 4.45 requires huggingface-hub < 1.0
try:
    import huggingface_hub
    ver_str = getattr(huggingface_hub, "__version__", "")
    if ver_str and int(ver_str.split(".")[0]) >= 1:
        print(f"[COMPAT] huggingface-hub {ver_str} detected. Adjusting to huggingface-hub<1.0 for transformers compatibility...", flush=True)
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "huggingface-hub<1.0"], check=False)
except Exception:
    pass

def parse_args():
    parser = argparse.ArgumentParser(description="LTX-Video Local Inference Engine")
    parser.add_argument("--prompt", type=str, required=True, help="Text description of the video")
    parser.add_argument("--negative-prompt", type=str, default="worst quality, low quality, deformed, distorted, blurry", help="Negative prompt")
    parser.add_argument("--image", type=str, default=None, help="Path to source image for Image-to-Video")
    parser.add_argument("--duration", type=int, default=5, help="Duration in seconds (approx 24-30 fps)")
    parser.add_argument("--resolution", type=str, default="768x512", help="Resolution widthxheight (must be multiples of 32)")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument("--guidance-scale", type=float, default=3.0, help="CFG guidance scale")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 for random)")
    parser.add_argument("--output", type=str, default="output_ltx.mp4", help="Output MP4 file path")
    parser.add_argument("--device", type=str, default="cuda", help="Target device (cuda or cpu)")
    parser.add_argument("--model-id", type=str, default="Lightricks/LTX-Video", help="HuggingFace model ID or local directory")
    return parser.parse_args()

def parse_dimensions(res_str, source_image_path=None):
    res_str = str(res_str).lower().strip() if res_str else "720p"
    if "x" in res_str:
        parts = res_str.split("x")
        try:
            w = int(parts[0])
            h = int(parts[1])
            # Align to multiples of 32 for DiT / VAE
            return max(256, (w // 32) * 32), max(256, (h // 32) * 32)
        except Exception:
            pass

    target_area = 768 * 512
    if "480" in res_str:
        target_area = 704 * 480

    # Auto-detect aspect ratio from input image if available
    if source_image_path and os.path.exists(source_image_path):
        try:
            with Image.open(source_image_path) as img:
                img_w, img_h = img.size
                aspect = img_w / img_h
                w = int(round(math.sqrt(target_area * aspect)))
                h = int(round(target_area / w))
                w = max(256, (w // 32) * 32)
                h = max(256, (h // 32) * 32)
                return w, h
        except Exception as e:
            print(f"[WARN] Failed to read source image aspect ratio: {e}", flush=True)

    if "portrait" in res_str:
        return 512, 768
    return 768, 512

def save_video_safely(frames, output_path, fps=24):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # 1. Try diffusers export_to_video first
    try:
        from diffusers.utils import export_to_video
        export_to_video(frames, output_path, fps=fps)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            return
    except Exception as e:
        print(f"[WARN] Diffusers export_to_video failed: {e}", flush=True)

    # 2. Try installing imageio-ffmpeg and retrying export_to_video
    try:
        import subprocess
        print("[INFO] Attempting to install imageio-ffmpeg...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "imageio-ffmpeg"], check=False)
        from diffusers.utils import export_to_video
        export_to_video(frames, output_path, fps=fps)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            return
    except Exception as e:
        print(f"[WARN] Retry after imageio-ffmpeg install failed: {e}", flush=True)

    # 3. Direct FFmpeg CLI streaming fallback (fast, high-quality, native libx264)
    print("[INFO] Using direct FFmpeg CLI stream encoding fallback...", flush=True)
    import subprocess
    import numpy as np

    np_frames = []
    for f in frames:
        if hasattr(f, "convert"):
            np_frames.append(np.array(f.convert("RGB")))
        elif isinstance(f, np.ndarray):
            np_frames.append(f)

    if not np_frames:
        raise RuntimeError("No frames generated to save video.")

    h, w, _ = np_frames[0].shape
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "19",
        output_path
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for frame in np_frames:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        proc.wait(timeout=120)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            return
    except Exception as e:
        print(f"[WARN] Direct FFmpeg CLI failed: {e}", flush=True)

    # 4. OpenCV fallback
    try:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, float(fps), (w, h))
        for f in np_frames:
            bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            out.write(bgr)
        out.release()
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            return
    except Exception as e:
        raise RuntimeError(f"All video export strategies failed: {e}")

def main():
    args = parse_args()
    start_time = time.time()

    print("========================================================", flush=True)
    print("⚡ CloudCompositing.com - Local LTX-Video 2.5 Engine", flush=True)
    print(f"🎬 Prompt       : {args.prompt[:80]}...", flush=True)
    if args.image:
        print(f"🖼️ Source Image : {args.image}", flush=True)
    print(f"⏱️ Duration     : {args.duration}s", flush=True)
    print(f"📐 Resolution   : {args.resolution}", flush=True)
    print(f"⚙️ Steps / CFG  : {args.steps} steps / CFG {args.guidance_scale}", flush=True)
    print(f"📁 Output       : {args.output}", flush=True)
    print("========================================================", flush=True)

    try:
        import torch
        if not hasattr(torch, "xpu"):
            class _DummyXpu:
                @staticmethod
                def is_available(): return False
                @staticmethod
                def device_count(): return 0
                @staticmethod
                def empty_cache(): pass
                def __getattr__(self, name): return lambda *args, **kwargs: None
            torch.xpu = _DummyXpu()
        from diffusers.utils import export_to_video
    except ImportError as e:
        print(f"[ERROR] Missing dependencies: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU.", flush=True)
        device = "cpu"

    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    if device == "cpu":
        dtype = torch.float32

    # LTX-Video frame calculation (typically 8k + 1 frames, e.g. 97 or 121 frames for 24fps)
    fps = 24
    num_frames = int(args.duration * fps)
    num_frames = ((num_frames - 1) // 8) * 8 + 1
    num_frames = max(25, min(num_frames, 161))

    width, height = parse_dimensions(args.resolution, args.image)
    print(f"[INFO] Aligned resolution: {width}x{height}, Number of frames: {num_frames} ({fps} fps)", flush=True)

    generator = None
    if args.seed >= 0:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)

    model_path = args.model_id
    # Check if present in /workspace/models/ltx-video or .hf_cache
    possible_local_paths = [
        "/workspace/models/ltx-video",
        "/workspace/models/LTX-Video",
        "models/ltx-video",
        "models/LTX-Video"
    ]
    for p in possible_local_paths:
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "model_index.json")):
            model_path = p
            print(f"[INFO] Using local weights found: {model_path}", flush=True)
            break

    print(f"[STATUS] Loading LTX-Video pipeline from '{model_path}'...", flush=True)

    is_i2v = args.image is not None and os.path.exists(args.image)

    try:
        if is_i2v:
            try:
                from diffusers import LTXImageToVideoPipeline
            except (ImportError, AttributeError):
                print("[WARN] LTXImageToVideoPipeline missing in current diffusers. Upgrading diffusers>=0.33.0...", flush=True)
                import subprocess
                subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-U", "diffusers>=0.33.0"], check=False)
                from diffusers import LTXImageToVideoPipeline

            pipe = LTXImageToVideoPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype
            )
        else:
            try:
                from diffusers import LTXPipeline
            except (ImportError, AttributeError):
                print("[WARN] LTXPipeline missing in current diffusers. Upgrading diffusers>=0.33.0...", flush=True)
                import subprocess
                subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-U", "diffusers>=0.33.0"], check=False)
                from diffusers import LTXPipeline

            pipe = LTXPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype
            )
    except Exception as e:
        print(f"[ERROR] Failed to load LTX-Video model: {e}", file=sys.stderr, flush=True)
        print("[HINT] Model might require downloading first via Studio Model Downloader.", file=sys.stderr, flush=True)
        sys.exit(2)

    # VRAM optimizations for GPU Pod
    if device == "cuda":
        try:
            pipe.enable_model_cpu_offload()
            print("[INFO] Model CPU Offload enabled (saves VRAM).", flush=True)
        except Exception:
            pipe.to("cuda")

        try:
            pipe.enable_vae_tiling()
            print("[INFO] VAE Tiling enabled.", flush=True)
        except Exception:
            pass

    print(f"[STATUS] Starting video generation ({args.steps} steps)...", flush=True)
    gen_start = time.time()

    pipeline_kwargs = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "generator": generator,
    }

    if is_i2v:
        source_img = Image.open(args.image).convert("RGB")
        source_img = source_img.resize((width, height), Image.Resampling.LANCZOS)
        pipeline_kwargs["image"] = source_img

    with torch.inference_mode():
        output = pipe(**pipeline_kwargs)

    frames = output.frames[0]
    gen_elapsed = time.time() - gen_start
    print(f"[INFO] Rendering finished in {gen_elapsed:.1f}s. Encoding MP4 file...", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_video_safely(frames, args.output, fps=fps)

    total_time = time.time() - start_time
    file_size_mb = os.path.getsize(args.output) / (1024 * 1024) if os.path.exists(args.output) else 0

    print(f"\n[SUCCESS] LTX Video generated successfully!", flush=True)
    print(f"[OUTPUT] {args.output} ({file_size_mb:.2f} MB)", flush=True)
    print(f"[TOTAL_TIME] {total_time:.1f}s", flush=True)

if __name__ == "__main__":
    main()
