#!/usr/bin/env bash
set -e

# Port configurable via variable d'environnement (RunPod HTTP Proxy)
PORT="${PORT:-3000}"

echo "========================================================"
echo "⚡ Welcome to RunPod Studio & Media Manager"
echo "🌐 HTTP Listening Port : ${PORT}"
echo "📁 Working Directory   : $(pwd)"
echo "========================================================"

# RunPod persistent volume support (/workspace)
# If /workspace exists, symlink web assets so files and datasets persist across restarts.
if [ -d "/workspace" ]; then
    cd /workspace
    [ ! -e "web" ] && ln -s /app/web web
    [ ! -e "src" ] && ln -s /app/src src
else
    cd /app
fi

# Execute Rust server in headless mode
exec /app/runpod-pipeline serve --port "$PORT"
