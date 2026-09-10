# RunPod Studio (Rust) 🦀⚡

High-performance, modular, and asynchronous Rust application orchestrating generative AI pipelines across **RunPod Serverless**, **Pod GPUs**, and **Public Endpoints** (Text-to-Video, Image-to-Video, Video-to-Video, Face Swap, Text-to-Image Flux, LoRA Fine-Tuning, Voice Cloning TTS, and LLM Prompt Enhancement).

---

## 🌟 Key Features

- **Web Studio & Media Gallery (`serve` / `ui`)**:
  - **Comprehensive Interactive Visual Dashboard** to launch and manage all generation pipelines.
  - **Built-in Media Gallery**: Browse images, audio, and videos, integrated video player, drag-and-drop file upload, instant download, deletion, and 1-click reuse of any media as a pipeline source.
  - **Real-time Live Job Monitoring**: Live progress status badges (`Idle`, `In Queue`, `Active Render`), compute time metrics, and streamed execution logs.
- **Cutting-edge Multi-Model Video Engines (`wan-2-5`, `ltx-2-5`, `minimax-h3`)**:
  - **Wan 2.5** (`wan-2-5`): Reference video model featuring high temporal consistency and cinematic fluidity.
  - **LTX-Video 2.5** (`ltx-2-5`): Ultra-fast DiT architecture optimized for record inference speeds directly on Pod GPU.
  - **MiniMax-H3 / Hailuo 3** (`minimax-h3`): 2K high-fidelity cinematic generation with synchronized ambient audio.
  - Selectable directly in the Web Studio or via CLI `--model <wan-2-5|ltx-2-5|minimax-h3>`.
- **Image-to-Video (`img2vid`)**:
  - Accepts **local image files** (`.png`, `.jpg`, `.webp`) automatically encoded as Base64 Data-URI, or remote URLs.
  - Generates high-quality animated videos via your chosen video engine.
- **Video-to-Video (`vid2vid`)**:
  - Restyle, transform, and alter existing videos (`.mp4`, `.webm`, or URLs) with precise control over transformation strength (`--strength`).
- **Face Swap (`faceswap`)**:
  - Swap faces on **images** or **videos** using a single source portrait photo with automatic high-definition restoration (*CodeFormer / GFPGAN*).
- **Interactive Crop Tool**:
  - Built-in visual cropping modal with preset ratios (`1:1 Square`, `9:16 Portrait`, `16:9 Landscape`) and direct destination routing.
- **Image-to-Image & Inpainting (`img2img`)**:
  - Transform, restyle, or synthesize new scenes from a reference image with fine-grained denoise strength (`--strength`).
  - **Interactive Inpainting Canvas**: Paint custom masks directly in the browser to modify specific regions while preserving the rest.
  - **ControlNet Spatial Guidance**: Precise edge (Canny) and Depth Map guidance for structural retention.
  - Supports both **RunPod Cloud (Flux / SDXL)** and **Local GPU (Diffusers)**.
- **Text-to-Image (`txt2img`)**:
  - High-speed image generation powered by **Flux 1 Schnell** or **Local Diffusers (SDXL Turbo / SDXL Lightning)** with custom steps, seeds, and dimensions.
- **LoRA Training Studio (`train-lora` / Web Studio)**:
  - Fine-tune custom concepts (faces, commercial products, artistic styles) directly on your Pod GPU.
  - Built-in dataset manager with multi-image drag-and-drop upload and AI vision **Auto-Captioning**.
  - Advanced VRAM optimizations: pre-cached VAE latents, FP16 mixed precision, and gradient accumulation.
  - Automatic export to `.safetensors` immediately usable across Text-to-Image and Image-to-Image.
- **Text-to-Speech & Voice Cloning (`tts`)**:
  - Ultra-natural local speech synthesis powered by **Kokoro-82M** (ultra-fast, English/multilingual).
  - High-fidelity zero-shot voice cloning with **Coqui XTTS-v2** from a 3-10 second reference audio clip.
- **Prompt Enhancer (`enhance`)**:
  - Automatic prompt refinement and creative detail expansion powered by **Qwen3-32B**.
- **Modern Terminal Experience & Architecture**:
  - Modular, extensible Rust stages.
  - Animated live status spinners, colored logs, and non-blocking asynchronous execution.
  - Pure `rustls` networking (no system OpenSSL dependencies required).

---

## 🚀 Installation & Build

Ensure Rust (1.75+) is installed on your system:

```bash
# Clone the repository
git clone https://github.com/NikOol172/cloudcompositing.git
cd cloudcompositing

# Build debug binary
cargo build

# Build optimized release binary
cargo build --release
```

The compiled binary will be located at `target/release/runpod-pipeline`.

---

## 🐳 Docker & RunPod Cloud Deployment

RunPod Studio provides two specialized container images:
- **GPU Edition** (`ghcr.io/nikool172/cloudcompositing:v1.0.0-gpu` or `:latest-gpu`): Full environment with CUDA 12.1, PyTorch GPU, ONNX Runtime GPU, Diffusers, and local model inference capabilities.
- **CPU Edition** (`ghcr.io/nikool172/cloudcompositing:v1.0.0-cpu` or `:latest-cpu`): Lightweight container (~1.5 GB) designed for affordable RunPod CPU Pods ($0.02–$0.05/hr) acting as a cloud orchestrator for Serverless APIs and lightweight local CPU tasks (Kokoro TTS).

```bash
# Pull the GPU container image
docker pull ghcr.io/nikool172/cloudcompositing:v1.0.0-gpu

# Run locally with GPU support
docker run --gpus all -p 3000:3000 -e RUNPOD_API_KEY="your_key" ghcr.io/nikool172/cloudcompositing:v1.0.0-gpu

# Or pull the lightweight CPU container image
docker pull ghcr.io/nikool172/cloudcompositing:v1.0.0-cpu
docker run -p 3000:3000 -e RUNPOD_API_KEY="your_key" ghcr.io/nikool172/cloudcompositing:v1.0.0-cpu
```

### Deploying as a 1-Click RunPod Template

#### Option A: GPU Template (Local Model Inference & AI Studio)
1. In the **RunPod Console** → **Templates** → **New Template**:
   - **Template Name**: `RunPod Studio (GPU)`
   - **Compute Type**: `GPU`
   - **Image Name**: `ghcr.io/nikool172/cloudcompositing:v1.0.0-gpu`
   - **Container Disk**: `30 GB`
   - **Volume Disk**: `20+ GB` (mount to `/workspace` or `/app/outputs`)
   - **Expose HTTP Ports**: `3000`
   - **Environment Variables**:
     - `PORT` = `3000`
     - `RUNPOD_API_KEY` = `your_runpod_api_key`
     - `HF_TOKEN` = `your_huggingface_token` (optional, for LTX-Video)
2. Deploy on any GPU pod (e.g. RTX 4090, A40, L40S).
3. Click **Connect → Connect to HTTP Service [Port 3000]** to launch the Studio.

#### Option B: CPU Template (Ultra-low cost API Orchestrator)
1. In the **RunPod Console** → **Templates** → **New Template**:
   - **Template Name**: `RunPod Studio (CPU Light)`
   - **Compute Type**: `CPU`
   - **Image Name**: `ghcr.io/nikool172/cloudcompositing:v1.0.0-cpu`
   - **Container Disk**: `10 GB`
   - **Expose HTTP Ports**: `3000`
   - **Environment Variables**:
     - `PORT` = `3000`
     - `RUNPOD_API_KEY` = `your_runpod_api_key`
2. Deploy on any RunPod CPU pod (~$0.02/hour).
3. Click **Connect → Connect to HTTP Service [Port 3000]**.

---

## 🔑 Configuration

Set your RunPod API key in your shell environment:

```bash
export RUNPOD_API_KEY="your_runpod_api_key"
```

Or create a `.env` file in the root of the project:

```env
RUNPOD_API_KEY=your_runpod_api_key
HF_TOKEN=your_huggingface_token
```

Or pass the global CLI flag `--api-key <KEY>`.

---

## 📖 Usage Guide

### 🌟 0. Web Studio & Media Gallery (`serve` / `ui`)

Launch the visual dashboard in your browser:

```bash
cargo run -- serve
# Or specify a custom port:
cargo run -- serve --port 3000
```

Open **`http://localhost:3000`** to access:
- Multi-pipeline creative studio.
- Interactive media gallery and file uploader.
- Prompt library and search engine.
- Real-time job logs and GPU balance monitoring.

---

### 1. Image-to-Video (`img2vid`)

Animate a local or remote image into video using your chosen engine (`wan-2-5`, `ltx-2-5`, `minimax-h3`):

```bash
# Using Wan 2.5 (Default)
cargo run -- img2vid \
  --image ./text-to-video/output_image.png \
  --prompt "Slow cinematic camera pan, soft golden hour reflections" \
  --duration 5 \
  --resolution 720p \
  --output animation.mp4

# Using MiniMax-H3 (Hailuo 3 with audio)
cargo run -- img2vid \
  --model minimax-h3 \
  --image ./my_image.png \
  --prompt "Dynamic cinematic movement with ambient sound, 4k detail" \
  --output hailuo_video.mp4

# Using LTX-Video 2.5 (Ultra-fast generation)
cargo run -- img2vid \
  --model ltx-2-5 \
  --image ./my_image.png \
  --output ltx_video.mp4
```

---

### 2. Video-to-Video (`vid2vid`)

Restyle or transform an existing video:

```bash
cargo run -- vid2vid \
  --model minimax-h3 \
  --video ./input_video.mp4 \
  --prompt "Anime style watercolor animation, vibrant colors, gentle movement, 8k uhd" \
  --strength 0.65 \
  --resolution 720p \
  --output ./restyled_video.mp4
```

---

### 3. Face Swap (`faceswap`)

Swap faces in an image or a video:

```bash
# Face Swap on Image
cargo run -- faceswap \
  --source ./portrait.jpg \
  --target ./target_scene.png \
  --output ./swapped_image.png

# Face Swap on Full Video
cargo run -- faceswap \
  --source ./portrait.jpg \
  --target ./target_clip.mp4 \
  --output ./swapped_video.mp4
```

---

### 4. Full Text-to-Video Pipeline (`txt2vid`)

Generate an image and automatically animate it into video:

```bash
cargo run -- txt2vid \
  --model wan-2-5 \
  --prompt "An elegant woman walking briskly toward modern airport terminal, cinematic lighting" \
  --resolution 720p \
  --duration 5 \
  --interactive \
  --image-output ./generated_image.png \
  --video-output ./final_video.mp4
```

---

### 5. Image-to-Image (`img2img`)

Transform or restyle from a reference image:

```bash
# Transform reference image
cargo run -- img2img \
  --image ./reference_item.png \
  --prompt "An athletic female pickleball player wearing pink compression socks, outdoor sunny court, 8k uhd photorealistic" \
  --strength 0.75 \
  --output ./transformed.png

# Inpainting with Mask
cargo run -- img2img \
  --image ./original.png \
  --mask ./mask.png \
  --prompt "A luxury sporty watch on wrist, realistic details" \
  --strength 0.80 \
  --output ./inpainted.png
```

---

### 6. Text-to-Image (`txt2img`)

Generate images using **RunPod Flux (Cloud)** or **Local GPU (Diffusers)**:

```bash
# Cloud Mode (Flux 1 Schnell)
cargo run -- txt2img \
  --prompt "Portrait photograph of an astronaut looking at distant galaxy, warm lighting" \
  --width 1024 \
  --height 1024 \
  --steps 4 \
  --output astronaut.png

# Local GPU Mode (e.g. RTX 2080 / 4090 with SDXL Turbo in < 2 seconds)
cargo run -- txt2img \
  --prompt "Portrait photograph of an astronaut looking at distant galaxy, 8k uhd" \
  --local \
  --local-model "stabilityai/sdxl-turbo" \
  --width 512 \
  --height 512 \
  --steps 2 \
  --output astronaut_local.png
```

---

### 7. LoRA Fine-Tuning (`train-lora`)

Train a custom concept directly on GPU:

```bash
python3 src/lora_train_engine.py \
  --dataset-dir ./datasets/my_subject \
  --instance-prompt "sks person" \
  --output-name my_lora.safetensors \
  --train-steps 500 \
  --lora-rank 8 \
  --base-model "runwayml/stable-diffusion-v1-5"
```

---

### 8. Text-to-Speech Synthesis (`tts`)

Synthesize speech or clone voices locally:

```bash
# Kokoro-82M Ultra-fast synthesis
python3 src/tts_engine.py \
  --engine kokoro \
  --text "Welcome to RunPod Studio. Experience state of the art generative AI." \
  --voice af_bella \
  --output welcome.wav

# XTTS-v2 Voice Cloning
python3 src/tts_engine.py \
  --engine xtts \
  --text "This speech mimics the exact timbre and intonation of the speaker sample." \
  --speaker-wav ./my_voice_sample.wav \
  --output cloned_voice.wav
```

---

### 9. Prompt Enhancer (`enhance`)

Enhance prompts using Qwen3-32B:

```bash
cargo run -- enhance \
  --prompt "A cozy coffee shop in winter"
```

---

## 🏗️ Codebase Architecture

```
src/
├── main.rs                   # Entry point and CLI orchestration
├── cli.rs                    # Clap subcommands and flags definition
├── server/                   # Web server & REST API (Axum)
│   └── mod.rs                # Media routes, async jobs, and live execution streaming
├── client.rs                 # Asynchronous RunPod HTTP client with spinners & polling
├── models.rs                 # Pipeline data models (Flux, Wan, FaceSwap, Vid2Vid, LLM, JobStatus)
├── utils.rs                  # Helpers: Data-URI encoding, media streaming, and preview utils
├── ltx_engine.py             # Local Pod GPU engine for LTX-Video 2.5
├── model_downloader.py       # Background downloader for gated Hugging Face weights
├── txt2img_engine.py         # Local GPU Text-to-Image engine (SDXL Turbo / Lightning)
├── img2img_engine.py         # Local GPU Image-to-Image & Inpainting engine
├── lora_train_engine.py      # High-performance LoRA fine-tuning script
├── tts_engine.py             # Kokoro & Coqui XTTS speech synthesis engine
└── pipeline/
    ├── mod.rs                # Stage trait & Pipeline execution engine
    ├── context.rs            # Shared PipelineContext across stages
    └── stages/
        ├── prompt_enhance.rs     # LLM enhancement stage (Qwen3)
        ├── text_to_image.rs      # Flux Schnell / Dev stage
        ├── image_to_video.rs     # Wan 2.5 / LTX-Video / MiniMax stage
        ├── video_to_video.rs     # Video-to-Video transformation stage
        ├── face_swap.rs          # ReActor / InsightFace Face Swap stage
        ├── interactive_review.rs # Interactive user validation stage
        └── download.rs           # Media download and verification stage
web/                          # Modern Web Studio (HTML5 / Vanilla CSS / ES Modules)
├── index.html                # Visual dashboard, modals, and pipeline tabs
├── style.css                 # Dark mode theme & Glassmorphism design system
└── app.js                    # Interactive controllers, polling, canvas painter & crop tool
Dockerfile                    # Multi-stage build with CUDA, PyTorch, and compiled Rust binary
.github/workflows/docker.yml  # Automated CI/CD build and publish to GHCR
```

---

## 📄 License

MIT License. Designed for high performance and modularity on RunPod.
