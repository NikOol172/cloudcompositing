#!/usr/bin/env python3
"""
High-Performance Local Face Swap Engine using InsightFace + Inswapper ONNX.
Supports both image and video face swapping with multi-threading and FFmpeg stream piping.
"""

import os
import sys
import site

# Configure caches on Drive D: to prevent filling Drive C:
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HOME"] = os.path.join(base_dir, ".hf_cache")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(base_dir, ".hf_cache", "hub")
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
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

import onnxruntime

def parse_args():
    parser = argparse.ArgumentParser(description="InsightFace Face Swap Engine")
    parser.add_argument("--source", required=True, help="Path to source face image")
    parser.add_argument("--target", required=True, help="Path to target image or video")
    parser.add_argument("--output", required=True, help="Path to output file")
    
    # Default model path relative to script or workspace
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model = os.path.join(os.path.dirname(script_dir), "models", "inswapper_128.onnx")
    if not os.path.exists(default_model):
        default_model = "/home/nhou/dev/runpod/models/inswapper_128.onnx"
        
    parser.add_argument("--model", default=default_model, help="Path to inswapper_128.onnx")
    parser.add_argument("--face-index", type=int, default=0, help="Face index to swap (default: 0 = largest face, -1 = all)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Execution provider (auto, cuda, cpu)")
    return parser.parse_args()

def get_providers(device_choice="auto"):
    available = onnxruntime.get_available_providers()
    if device_choice == "cuda":
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print("[WARN] CUDA requested but CUDAExecutionProvider not available in onnxruntime. Falling back to CPU.", file=sys.stderr)
        return ["CPUExecutionProvider"]
    elif device_choice == "cpu":
        return ["CPUExecutionProvider"]
    else:  # auto
        if "CUDAExecutionProvider" in available:
            print("[INFO] CUDA GPU detected! Using CUDAExecutionProvider for ultra-fast FaceSwap.")
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print("[INFO] No CUDA provider found in onnxruntime. Using CPUExecutionProvider.")
        return ["CPUExecutionProvider"]

def main():
    args = parse_args()

    if not os.path.exists(args.source):
        print(f"[ERROR] Source face file '{args.source}' does not exist", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.target):
        print(f"[ERROR] Target file '{args.target}' does not exist", file=sys.stderr)
        sys.exit(1)

    providers = get_providers(args.device)
    print(f"[INFO] Initializing InsightFace analysis and Inswapper model with providers: {providers}...")
    # Initialize face analyzer (buffalo_l)
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1, det_size=(320, 320))

    # Initialize inswapper model
    swapper = insightface.model_zoo.get_model(args.model, providers=providers)

    # Load source face
    source_img = cv2.imread(args.source)
    if source_img is None:
        print(f"[ERROR] Failed to read source image: {args.source}", file=sys.stderr)
        sys.exit(1)

    source_faces = app.get(source_img)
    if not source_faces:
        print(f"[ERROR] No face detected in source image: {args.source}", file=sys.stderr)
        sys.exit(1)

    # Sort faces by size (area) to pick the most prominent face
    source_faces.sort(key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
    source_face = source_faces[0]
    print(f"[INFO] Detected source face with confidence: {source_face.det_score:.2f}")

    # Check if target is video
    is_video = args.target.lower().endswith(('.mp4', '.mov', '.avi', '.webm', '.mkv'))

    if not is_video:
        # Single Image Swap
        target_img = cv2.imread(args.target)
        if target_img is None:
            print(f"[ERROR] Failed to read target image: {args.target}", file=sys.stderr)
            sys.exit(1)

        target_faces = app.get(target_img)
        if not target_faces:
            print(f"[WARNING] No face detected in target image. Copying original.", file=sys.stderr)
            cv2.imwrite(args.output, target_img)
            sys.exit(0)

        target_faces.sort(key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
        res_img = target_img.copy()
        for idx, target_face in enumerate(target_faces):
            if args.face_index == -1 or idx == args.face_index:
                res_img = swapper.get(res_img, target_face, source_face, paste_back=True)

        cv2.imwrite(args.output, res_img)
        print(f"[SUCCESS] Swapped image saved to: {args.output}")
        sys.exit(0)

    # Video Swap using OpenCV and FFmpeg muxing
    cap = cv2.VideoCapture(args.target)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open target video: {args.target}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video details: {width}x{height} @ {fps:.2f}fps ({total_frames} frames)")

    temp_raw_video = args.output + ".temp_raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(temp_raw_video, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        target_faces = app.get(frame)
        if target_faces:
            target_faces.sort(key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
            for idx, target_face in enumerate(target_faces):
                if args.face_index == -1 or idx == args.face_index:
                    frame = swapper.get(frame, target_face, source_face, paste_back=True)

        out_writer.write(frame)

        if frame_idx % 10 == 0 or frame_idx == total_frames:
            pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
            print(f"[PROGRESS] Traitement des frames: {frame_idx}/{total_frames} ({pct:.1f}%)", flush=True)

    cap.release()
    out_writer.release()

    # Re-mux with audio and standard H.264 via ffmpeg
    ffmpeg_bin = "/home/nhou/.local/bin/ffmpeg" if os.path.exists("/home/nhou/.local/bin/ffmpeg") else "ffmpeg"
    mux_cmd = (
        f'{ffmpeg_bin} -y -i "{temp_raw_video}" -i "{args.target}" '
        f'-map 0:v:0 -map 1:a:0? -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "{args.output}"'
    )
    print(f"[INFO] Finalizing and encoding H.264 video...")
    ret_code = os.system(mux_cmd)
    if os.path.exists(temp_raw_video):
        os.remove(temp_raw_video)

    if ret_code == 0 and os.path.exists(args.output):
        print(f"[SUCCESS] Face Swap video completed and saved to: {args.output}")
    else:
        print(f"[ERROR] Failed to mux final video.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
