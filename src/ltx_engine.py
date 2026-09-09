#!/usr/bin/env python3
"""
High-Performance Local LTX-Video Inference Engine using PyTorch & HuggingFace Diffusers.
Optimized for Pod GPUs with BF16/FP16, model CPU offload, and VAE tiling.
Supports both Text-to-Video and Image-to-Video via Lightricks/LTX-Video.
"""

import argparse
import os
import sys
import time
from PIL import Image

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

def parse_dimensions(res_str):
    res_str = res_str.lower().strip()
    if res_str == "720p":
        return 768, 512
    if res_str == "480p":
        return 704, 480
    if "x" in res_str:
        parts = res_str.split("x")
        try:
            w = int(parts[0])
            h = int(parts[1])
            # Align to multiples of 32 for DiT / VAE
            w = (w // 32) * 32
            h = (h // 32) * 32
            return w, h
        except Exception:
            pass
    return 768, 512

def main():
    args = parse_args()
    start_time = time.time()

    print("========================================================", flush=True)
    print("⚡ RunPod Studio - Moteur Local LTX-Video 2.5", flush=True)
    print(f"🎬 Prompt       : {args.prompt[:80]}...", flush=True)
    if args.image:
        print(f"🖼️ Source Image : {args.image}", flush=True)
    print(f"⏱️ Durée        : {args.duration}s", flush=True)
    print(f"📐 Résolution   : {args.resolution}", flush=True)
    print(f"⚙️ Steps / CFG  : {args.steps} steps / CFG {args.guidance_scale}", flush=True)
    print(f"📁 Sortie       : {args.output}", flush=True)
    print("========================================================", flush=True)

    try:
        import torch
        from diffusers.utils import export_to_video
    except ImportError as e:
        print(f"[ERROR] Dépendances manquantes : {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA non disponible, basculement forcé sur CPU.", flush=True)
        device = "cpu"

    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    if device == "cpu":
        dtype = torch.float32

    # Calcul du nombre de frames (LTX-Video utilise habituellement 8k + 1 frames, ex: 97 ou 121 frames pour 24fps)
    # LTX fonctionne par paquets de 8 frames + 1
    fps = 24
    num_frames = int(args.duration * fps)
    num_frames = ((num_frames - 1) // 8) * 8 + 1
    num_frames = max(25, min(num_frames, 161))

    width, height = parse_dimensions(args.resolution)
    print(f"[INFO] Résolution alignée : {width}x{height}, Nombre de frames : {num_frames} ({fps} fps)", flush=True)

    generator = None
    if args.seed >= 0:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)

    model_path = args.model_id
    # Vérifier si présent dans /workspace/models/ltx-video ou .hf_cache
    possible_local_paths = [
        "/workspace/models/ltx-video",
        "/workspace/models/LTX-Video",
        "models/ltx-video",
        "models/LTX-Video"
    ]
    for p in possible_local_paths:
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "model_index.json")):
            model_path = p
            print(f"[INFO] Utilisation des poids locaux trouvés : {model_path}", flush=True)
            break

    print(f"[STATUS] Chargement du pipeline LTX-Video depuis '{model_path}'...", flush=True)

    is_i2v = args.image is not None and os.path.exists(args.image)

    try:
        if is_i2v:
            from diffusers import LTXImageToVideoPipeline
            pipe = LTXImageToVideoPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype
            )
        else:
            from diffusers import LTXPipeline
            pipe = LTXPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype
            )
    except Exception as e:
        print(f"[ERROR] Impossible de charger le modèle LTX-Video : {e}", file=sys.stderr, flush=True)
        print("[HINT] Le modèle nécessite peut-être un téléchargement préalable via le Gestionnaire de Modèles du Studio.", file=sys.stderr, flush=True)
        sys.exit(2)

    # Optimisations mémoire VRAM pour Pod GPU
    if device == "cuda":
        try:
            pipe.enable_model_cpu_offload()
            print("[INFO] Model CPU Offload activé (économie de VRAM).", flush=True)
        except Exception:
            pipe.to("cuda")

        try:
            pipe.enable_vae_tiling()
            print("[INFO] VAE Tiling activé.", flush=True)
        except Exception:
            pass

    print(f"[STATUS] Lancement de la génération vidéo ({args.steps} étapes)...", flush=True)
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
        # Redimensionner l'image source aux dimensions cibles
        source_img = source_img.resize((width, height), Image.Resampling.LANCZOS)
        pipeline_kwargs["image"] = source_img

    with torch.inference_mode():
        output = pipe(**pipeline_kwargs)

    frames = output.frames[0]
    gen_elapsed = time.time() - gen_start
    print(f"[INFO] Rendu achevé en {gen_elapsed:.1f}s. Encodage du fichier MP4...", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    export_to_video(frames, args.output, fps=fps)

    total_time = time.time() - start_time
    file_size_mb = os.path.getsize(args.output) / (1024 * 1024) if os.path.exists(args.output) else 0

    print(f"\n[SUCCESS] Vidéo LTX générée avec succès !", flush=True)
    print(f"[OUTPUT] {args.output} ({file_size_mb:.2f} Mo)", flush=True)
    print(f"[TOTAL_TIME] {total_time:.1f}s", flush=True)

if __name__ == "__main__":
    main()
