#!/usr/bin/env bash
set -e

# Port configurable via variable d'environnement (RunPod HTTP Proxy)
PORT="${PORT:-3000}"

echo "========================================================"
echo "⚡ Bienvenue sur RunPod Studio & Media Manager"
echo "🌐 Port d'écoute HTTP : ${PORT}"
echo "📁 Répertoire de travail : $(pwd)"
echo "========================================================"

# Support du volume persistant /workspace de RunPod
# Si /workspace existe (point de montage standard RunPod), on s'y place
# pour que tous les médias générés et datasets soient persistés.
if [ -d "/workspace" ]; then
    cd /workspace
    [ ! -e "web" ] && ln -s /app/web web
    [ ! -e "src" ] && ln -s /app/src src
else
    cd /app
fi

# Exécution du serveur Rust sans tenter d'ouvrir de navigateur graphique (headless)
exec /app/runpod-pipeline serve --port "$PORT" --open false
