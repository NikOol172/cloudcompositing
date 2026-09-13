#!/usr/bin/env python3
"""
CloudCompositing.com Model Downloader
Downloads Hugging Face models (including gated models like Lightricks/LTX-Video)
with real-time progress reporting and clear authorization error handling.
"""

import argparse
import os
import sys
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Download AI models from Hugging Face Hub")
    parser.add_argument("--model-id", type=str, required=True, help="Hugging Face model repository (e.g. Lightricks/LTX-Video)")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face Access Token for gated models")
    parser.add_argument("--local-dir", type=str, default=None, help="Optional local directory to store weights directly")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory (defaults to .hf_cache/hub)")
    return parser.parse_args()

def main():
    args = parse_args()

    model_id = args.model_id.strip()
    hf_token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    print(f"========================================================", flush=True)
    print(f"⬇️  CloudCompositing.com Model Downloader", flush=True)
    print(f"📦 Target Model : {model_id}", flush=True)
    if hf_token:
        masked = hf_token[:4] + "..." + hf_token[-4:] if len(hf_token) > 8 else "***"
        print(f"🔑 HF Token     : {masked}", flush=True)
    else:
        print(f"⚠️  HF Token     : Not provided (gated models require an access token)", flush=True)
    print(f"========================================================", flush=True)

    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except ImportError:
        print("[ERROR] 'huggingface_hub' is not installed. Installing...", flush=True)
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "huggingface_hub"])
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    local_dir = args.local_dir
    cache_dir = args.cache_dir

    kwargs = {
        "repo_id": model_id,
        "token": hf_token,
    }

    # For repositories with massive collections of loose standalone checkpoints (e.g. Lightricks/LTX-Video which has 280 GB of dev checkpoints & GIFs),
    # only download the essential Diffusers pipeline components (~11 GB) needed for inference:
    if "ltx-video" in model_id.lower():
        print("🎯 Optimizing download: filtering for Diffusers pipeline files (~11 GB) instead of 280 GB full repository.", flush=True)
        kwargs["allow_patterns"] = [
            "model_index.json",
            "scheduler/*",
            "text_encoder/*",
            "tokenizer/*",
            "transformer/*",
            "vae/*",
        ]
        kwargs["ignore_patterns"] = [
            "*.gif",
            "media/*",
            "ltxv-*",
            "ltx-video-2b-*",
        ]

    if local_dir:
        os.makedirs(local_dir, exist_ok=True)
        print(f"📁 Destination folder : {local_dir}", flush=True)
        # Download directly to local_dir without duplicating files into cache_dir (saves 50% disk space)
        kwargs["local_dir"] = local_dir
    else:
        if not cache_dir:
            if os.path.exists("/workspace"):
                cache_dir = "/workspace/.hf_cache/hub"
            else:
                cache_dir = os.path.abspath(".hf_cache/hub")
        os.makedirs(cache_dir, exist_ok=True)
        print(f"📁 Cache folder : {cache_dir}", flush=True)
        kwargs["cache_dir"] = cache_dir

    start_time = time.time()

    try:
        print(f"[STATUS] Connecting to Hugging Face and verifying permissions...", flush=True)
        download_path = snapshot_download(**kwargs)

        elapsed = time.time() - start_time
        print(f"\n[SUCCESS] Model '{model_id}' downloaded and verified successfully!", flush=True)
        print(f"[PATH] {download_path}", flush=True)
        print(f"[TIME] Total duration: {elapsed:.1f}s", flush=True)

    except GatedRepoError as e:
        print("\n" + "=" * 60, file=sys.stderr, flush=True)
        print(f"[GATED_REPO_ERROR] Access denied to gated model '{model_id}'.", file=sys.stderr, flush=True)
        print(f"1. Visit: https://huggingface.co/{model_id}", file=sys.stderr, flush=True)
        print(f"2. Accept the model terms of service / license agreement.", file=sys.stderr, flush=True)
        print(f"3. Create a 'Read' token on https://huggingface.co/settings/tokens and save it in Studio Settings.", file=sys.stderr, flush=True)
        print("=" * 60, file=sys.stderr, flush=True)
        sys.exit(43)

    except RepositoryNotFoundError:
        print(f"\n[ERROR] Hugging Face repository '{model_id}' not found.", file=sys.stderr, flush=True)
        sys.exit(44)

    except Exception as e:
        err_str = str(e).lower()
        if "401" in err_str or "unauthorized" in err_str:
            print("\n[AUTH_ERROR] Invalid or expired Hugging Face access token (Error 401).", file=sys.stderr, flush=True)
            print("Please verify your token in the Studio Settings tab.", file=sys.stderr, flush=True)
            sys.exit(41)
        elif "403" in err_str or "forbidden" in err_str:
            print(f"\n[GATED_REPO_ERROR] License not accepted for '{model_id}' (Error 403).", file=sys.stderr, flush=True)
            print(f"Please accept the license on https://huggingface.co/{model_id}", file=sys.stderr, flush=True)
            sys.exit(43)
        elif "disk quota exceeded" in err_str or "122" in err_str or "no space left" in err_str:
            print(f"\n[DISK_FULL_ERROR] Disk quota exceeded while downloading '{model_id}'.", file=sys.stderr, flush=True)
            print("Please clean up unused caches to free disk space:", file=sys.stderr, flush=True)
            print("  rm -rf /workspace/models/ltx-video", file=sys.stderr, flush=True)
            print("  rm -rf /workspace/.hf_cache/hub/tmp*", file=sys.stderr, flush=True)
            print("  pip cache purge", file=sys.stderr, flush=True)
            sys.exit(122)
        else:
            print(f"\n[ERROR] Download failed: {e}", file=sys.stderr, flush=True)
            sys.exit(1)

if __name__ == "__main__":
    main()
