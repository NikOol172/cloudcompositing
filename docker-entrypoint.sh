#!/usr/bin/env bash
set -e

# Port configurable via variable d'environnement (RunPod HTTP Proxy)
PORT="${PORT:-3000}"

echo "========================================================"
echo "⚡ Welcome to CloudCompositing.com & Media Manager"
echo "🌐 HTTP Listening Port : ${PORT}"
echo "📁 Working Directory   : $(pwd)"
echo "========================================================"

# RunPod persistent volume support (/workspace)
# If /workspace exists, symlink web assets so files and datasets persist across restarts.
if [ -d "/workspace" ]; then
    cd /workspace
    [ ! -e "web" ] && ln -s /app/web web
    [ ! -e "src" ] && ln -s /app/src src
    [ ! -e "start_comfyui.sh" ] && ln -s /app/start_comfyui.sh start_comfyui.sh
else
    cd /app
fi

# Define ComfyUI directory based on environment
if [ -d "/workspace" ]; then
    COMFY_DIR="/workspace/ComfyUI"
else
    COMFY_DIR="/app/ComfyUI"
fi

# Clone ComfyUI if it does not exist
if [ ! -d "$COMFY_DIR" ]; then
    echo "📦 Downloading ComfyUI to $COMFY_DIR..."
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
fi

# Install ComfyUI dependencies (it's safe to run this multiple times, pip will use cache)
echo "📦 Installing ComfyUI dependencies..."
pip install --no-cache-dir -r "$COMFY_DIR/requirements.txt"

# Télécharger les modèles nécessaires si manquants
echo "⬇️ Checking and downloading required ComfyUI models..."
export COMFYUI_DIR="$COMFY_DIR"
python3 /app/src/setup_comfyui_models.py

# Start ComfyUI in the background
echo "🚀 Starting ComfyUI server in the background (port 8188)..."
pushd "$COMFY_DIR" > /dev/null
# Use --listen 127.0.0.1 to keep it internal, or 0.0.0.0 if external access is needed.
python3 main.py --listen 127.0.0.1 --port 8188 &
popd > /dev/null

# Wait a few seconds for ComfyUI to initialize
sleep 3

# Execute Rust server in headless mode
exec /app/runpod-pipeline serve --port "$PORT"
