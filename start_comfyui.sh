#!/bin/bash
cd /workspace/ComfyUI || { echo "ComfyUI not found at /workspace/ComfyUI"; exit 1; }

# start comfyui if not running
if ! pgrep -f "main.py" > /dev/null; then
    echo "Starting ComfyUI..."
    nohup python3 main.py --listen 127.0.0.1 --port 8188 > comfyui.log 2>&1 < /dev/null &
fi

# download cloudflared if not exists
if ! command -v cloudflared > /dev/null; then
    echo "Installing cloudflared..."
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
    dpkg -i cloudflared-linux-amd64.deb > /dev/null 2>&1
fi

# start cloudflared if not running
if ! pgrep -f "cloudflared tunnel" > /dev/null; then
    echo "Starting cloudflared tunnel..."
    nohup cloudflared tunnel --url http://127.0.0.1:8188 > cloudflared.log 2>&1 < /dev/null &
    sleep 5
fi

# Get URL
URL=$(grep -o 'https://.*\.trycloudflare\.com' cloudflared.log | tail -n 1)

if [ -z "$URL" ]; then
    echo "Error: Failed to get trycloudflare.com URL"
    exit 1
fi

echo "$URL"

