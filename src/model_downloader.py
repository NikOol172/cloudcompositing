#!/usr/bin/env python3
"""
RunPod Studio Model Downloader
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
    print(f"⬇️  RunPod Studio Model Downloader", flush=True)
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

    cache_dir = args.cache_dir
    if not cache_dir:
        # Prioritize /workspace for RunPod persistence
        if os.path.exists("/workspace"):
            cache_dir = "/workspace/.hf_cache/hub"
        else:
            cache_dir = os.path.abspath(".hf_cache/hub")

    os.makedirs(cache_dir, exist_ok=True)
    print(f"📁 Cache folder : {cache_dir}", flush=True)

    local_dir = args.local_dir
    if local_dir:
        os.makedirs(local_dir, exist_ok=True)
        print(f"📁 Destination folder : {local_dir}", flush=True)

    start_time = time.time()

    try:
        print(f"[STATUS] Connecting to Hugging Face and verifying permissions...", flush=True)
        download_path = snapshot_download(
            repo_id=model_id,
            token=hf_token,
            cache_dir=cache_dir,
            local_dir=local_dir,
            local_dir_use_symlinks="auto" if not local_dir else False,
            resume_download=True,
        )

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
        else:
            print(f"\n[ERROR] Download failed: {e}", file=sys.stderr, flush=True)
            sys.exit(1)

if __name__ == "__main__":
    main()
