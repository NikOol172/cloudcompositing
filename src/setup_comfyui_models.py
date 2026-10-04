#!/usr/bin/env python3
"""
Téléchargeur automatique de modèles pour ComfyUI.
Ce script s'assure que les modèles nécessaires aux workflows (SDXL, LTX-Video, FaceSwap)
sont présents dans les dossiers ComfyUI avant son démarrage.
"""

import os
import sys

def download_model(repo_id, filename, dest_dir, token=None):
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)
    
    if os.path.exists(dest_path):
        print(f"[OK] Le modèle {filename} est déjà présent dans {dest_dir}.")
        return

    print(f"[DL] Téléchargement de {filename} depuis {repo_id}...")
    try:
        from huggingface_hub import hf_hub_download
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=dest_dir,
            local_dir_use_symlinks=False,
            token=token
        )
        print(f"[SUCCESS] {filename} téléchargé avec succès dans {downloaded_path}.")
    except Exception as e:
        print(f"[ERROR] Échec du téléchargement de {filename}: {e}", file=sys.stderr)

def main():
    try:
        import huggingface_hub
    except ImportError:
        import subprocess
        print("Installation de huggingface_hub...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        
    comfyui_base = os.environ.get("COMFYUI_DIR", "/workspace/ComfyUI")
    if not os.path.exists(comfyui_base):
        print(f"[WARN] Dossier ComfyUI introuvable à {comfyui_base}. Skipping model downloads.", flush=True)
        return
        
    checkpoints_dir = os.path.join(comfyui_base, "models", "checkpoints")
    insightface_dir = os.path.join(comfyui_base, "models", "insightface")
    
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    
    # Configuration des modèles
    # format: (repo_id, filename, destination_directory, requires_token)
    models = [
        ("stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors", checkpoints_dir, False),
        ("Lightricks/LTX-Video", "ltx-video-2.0.1.safetensors", checkpoints_dir, True),
        ("ezioruan/inswapper_128.onnx", "inswapper_128.onnx", insightface_dir, False)
    ]
    
    print("=== Vérification et Téléchargement des modèles ComfyUI ===", flush=True)
    for repo_id, filename, dest_dir, requires_token in models:
        token_to_use = hf_token if requires_token else None
        if requires_token and not token_to_use:
            print(f"[WARN] Ignoré {filename} : Token HuggingFace requis mais non fourni (HF_TOKEN).", flush=True)
            continue
        download_model(repo_id, filename, dest_dir, token=token_to_use)
    print("=== Terminé ===", flush=True)

if __name__ == "__main__":
    main()
