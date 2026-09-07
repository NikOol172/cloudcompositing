# ==============================================================================
# Étape 1 : Compilation du binaire Rust (RunPod Pipeline & Web Server)
# ==============================================================================
FROM rust:latest AS builder

WORKDIR /usr/src/app

# Installation de la suite de compilation C / C++ requise pour ring et extensions natives
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libssl-dev \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copie des fichiers de configuration Rust
COPY Cargo.toml Cargo.lock ./

# Copie du code source
COPY src/ ./src/

# Compilation en mode release optimisé
RUN cargo build --release --bin runpod-pipeline

# ==============================================================================
# Étape 2 : Image d'Exécution avec CUDA, PyTorch et Moteurs IA
# ==============================================================================
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

LABEL maintainer="NikOol172"
LABEL description="RunPod Studio - Interface Web moderne & Orchestrateur IA (Video, Image, LoRA, TTS)"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PORT=3000

WORKDIR /app

# Dépendances système requises pour OpenCV, TTS audio et compilations pip (insightface, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python requises pour les moteurs IA locaux
COPY requirements-local-gpu.txt /app/
RUN pip install --no-cache-dir -r requirements-local-gpu.txt

# Copie du binaire Rust compilé depuis l'étape précédente
COPY --from=builder /usr/src/app/target/release/runpod-pipeline /app/runpod-pipeline

# Copie des ressources web (interface Studio moderne) et scripts IA
COPY web/ /app/web/
COPY src/ /app/src/
COPY docker-entrypoint.sh /app/docker-entrypoint.sh

RUN chmod +x /app/docker-entrypoint.sh /app/runpod-pipeline

# Exposition du port Web Studio par défaut (3000)
EXPOSE 3000

# Script de démarrage
ENTRYPOINT ["/app/docker-entrypoint.sh"]
