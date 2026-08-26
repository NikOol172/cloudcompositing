# RunPod Pipeline (Rust) 🦀⚡

Application Rust modulaire et asynchrone pour orchestrer des pipelines d'IA générative sur **RunPod Serverless** et **Public Endpoints** (Text-to-Image, Image-to-Video, Text-to-Video complet, enrichissement LLM de prompts).

---

## 🌟 Fonctionnalités

- **Interface Web Studio & Gestionnaire de Médias (`serve` / `ui`)** :
  - **Tableau de bord visuel interactif** complet pour lancer toutes les générations (Text-to-Video, Image-to-Video, Face Swap, Video-to-Video, Text-to-Image, Prompt Enhancer).
  - **Gestionnaire de médias intégré** : Galerie d'images et vidéos, lecteur vidéo intégré, import par glisser-déposer (Drag & Drop), téléchargement, suppression et réutilisation de médias comme source en 1 clic.
  - **Suivi des jobs en temps réel** avec logs d'exécution et prévisualisation directe.
- **Multi-modèles Vidéo de pointe (`wan-2-5`, `ltx-2-5`, `minimax-h3`)** :
  - **Wan 2.5** (`wan-2-5`) : Moteur de référence haute fluidité et cohérence temporelle.
  - **LTX-Video 2.5** (`ltx-2-5`) : Architecture DiT ultra-rapide conçue pour des temps d'inférence records.
  - **MiniMax-H3 / Hailuo 3** (`minimax-h3`) : Génération cinématique 2K haute fidélité avec audio synchronisé.
  - Sélectionnable directement dans l'interface Web Studio ou via le flag CLI `--model <wan-2-5|ltx-2-5|minimax-h3>`.
- **Image-to-Video (`img2vid`)** :
  - Accepte des **fichiers locaux** (`.png`, `.jpg`, `.webp`) automatiquement encodés en Data-URI (base64) ou des **URLs distantes**.
  - Génère des vidéos de qualité via le moteur de votre choix (**Wan 2.5**, **LTX-Video 2.5**, **MiniMax-H3**).
- **Video-to-Video (`vid2vid`)** :
  - Restyler, transformer ou modifier une vidéo existante (`.mp4`, `.webm` ou URL) avec contrôle de la force de transformation (`--strength`).
- **Face Swap (`faceswap`)** :
  - Remplacer un visage sur une **image** ou une **vidéo** à partir d'une photo source avec restauration automatique haute fidélité (*CodeFormer / GFPGAN*).
- **Text-to-Video complet (`txt2vid`)** :
  - Enchaîne optionnellement l'enrichissement de prompt (Qwen3), la génération d'image haute fidélité (**Flux 1 Schnell / Dev**) et l'animation vidéo (**Wan 2.5**, **LTX-Video 2.5**, **MiniMax-H3**).
  - Mode interactif (`-i / --interactive`) permettant d'ouvrir et de valider l'image avant de lancer le rendu vidéo.
- **Image-to-Image (`img2img`)** :
  - Transformer, restyler ou générer une nouvelle scène à partir d'une image de référence (ex: transformer une photo de chaussettes en joueuse de pickleball en pleine action) avec contrôle précis du degré de transformation (`--strength` / denoise).
  - Prise en charge sur le **Cloud RunPod (Flux / SDXL)** et en **Local GPU (Diffusers / RTX 2080)**.
- **Text-to-Image (`txt2img`)** :
  - Génération rapide d'images avec personnalisation des dimensions, du seed et des steps.
- **Enrichissement de Prompt (`enhance`)** :
  - Optimisation des descriptions textuelles via LLM (Qwen3-32B).
- **Architecture par Étapes (Stages)** :
  - Conception modulaire facilitant l'ajout de nouveaux modèles, filtres d'upscaling ou nœuds ComfyUI.
- **Interface Terminal Moderne** :
  - Spinners animés de statut en temps réel (`IN_QUEUE`, `IN_PROGRESS`, `COMPLETED`), barres de progression de téléchargement et logs colorés.
  - Utilisation de `rustls` (zéro dépendance OpenSSL système requise).

---

## 🚀 Installation & Compilation

Assurez-vous d'avoir Rust installé (1.75+) :

```bash
# Compiler en mode debug
cargo build

# Ou compiler la version optimisée (Release)
cargo build --release
```

L'exécutable se trouvera dans `target/release/runpod-pipeline`.

---

## 🔑 Configuration

Définissez votre clé API RunPod soit dans votre environnement :

```bash
export RUNPOD_API_KEY="votre_cle_api_runpod"
```

Soit via un fichier `.env` à la racine du projet :

```env
RUNPOD_API_KEY=votre_cle_api_runpod
```

Soit en passant le flag global `--api-key <VOTRE_CLE>` lors de l'exécution.

---

## 📖 Guide d'Utilisation
 
### 🌟 0. Interface Web Studio & Galerie Médias (`serve` / `ui`)

Lancer le tableau de bord interactif dans votre navigateur :

```bash
cargo run -- serve
# Ou spécifier un port personnalisé :
cargo run -- serve --port 3000
```

Accessible immédiatement sur **`http://localhost:3000`** avec :
- **Studio de génération visuel** pour tous les modèles.
- **Galerie de médias** avec filtres, prévisualisation, lecteur vidéo intégré et téléversement Drag & Drop.
- **Suivi des jobs RunPod en temps réel**.

---

### 1. Image-to-Video (`img2vid`)

Animer une image locale ou distante en vidéo avec le modèle souhaité (`wan-2-5`, `ltx-2-5`, `minimax-h3`) :

```bash
# Avec Wan 2.5 (par défaut)
cargo run -- img2vid \
  --image ./text-to-video/output_image.png \
  --prompt "Slow cinematic camera pan, soft light reflections" \
  --duration 5 \
  --resolution 720p \
  --output mon_animation.mp4

# Avec MiniMax-H3 (Hailuo 3)
cargo run -- img2vid \
  --model minimax-h3 \
  --image ./mon_image.png \
  --prompt "Dynamic cinematic movement with ambient sound, 4k detail" \
  --output hailuo_video.mp4

# Avec LTX-Video 2.5 (génération ultra-rapide)
cargo run -- img2vid \
  --model ltx-2-5 \
  --image ./mon_image.png \
  --output ltx_video.mp4
```

### 2. Video-to-Video (`vid2vid`)

Modifier, restyler ou transformer une vidéo existante :

```bash
cargo run -- vid2vid \
  --model minimax-h3 \
  --video ./avion_focus_bas.mp4 \
  --prompt "A luxury cinematic commercial shot of black silky compression flight socks, golden hour warm lighting from airplane window, subtle gentle movement, 8k uhd, photorealistic" \
  --strength 0.65 \
  --resolution 720p \
  --output ./avion_modifie.mp4
```

### 3. Face Swap (`faceswap`)

Remplacer un visage sur une image ou une vidéo :

```bash
# Face Swap sur une image
cargo run -- faceswap \
  --source ./mon_portrait.jpg \
  --target ./senior_taxi_aeroport.png \
  --output ./senior_mon_visage.png

# Face Swap sur une vidéo complète
cargo run -- faceswap \
  --source ./mon_portrait.jpg \
  --target ./senior_taxi_aeroport.mp4 \
  --output ./senior_video_mon_visage.mp4
```

### 4. Pipeline complet Text-to-Video (`txt2vid`)

Générer l'image puis l'animer automatiquement avec le moteur vidéo de votre choix :

```bash
cargo run -- txt2vid \
  --model minimax-h3 \
  --prompt "An elegant woman walking briskly toward modern airport terminal, pink compression socks" \
  --resolution 720p \
  --duration 5 \
  --interactive \
  --image-output ./mon_image.png \
  --video-output ./ma_video.mp4
```

### 5. Image-to-Image (`img2img`)

Générer une nouvelle scène ou transformer une image de référence :

```bash
# Exemple : Générer une joueuse de pickleball à partir d'une photo de bas de compression
cargo run -- img2img \
  --image ./bas_compression.png \
  --prompt "An athletic female pickleball player wearing the pink compression socks, holding a paddle on outdoor sunny court, 8k uhd photorealistic" \
  --strength 0.75 \
  --output ./pickleball_player.png

# Exemple avec masque Inpainting (modifier uniquement une zone précise)
cargo run -- img2img \
  --image ./photo_originale.png \
  --mask ./masque_zone.png \
  --prompt "A luxury sporty watch on wrist, realistic details" \
  --strength 0.80 \
  --output ./photo_retouchee.png
```

### 6. Text-to-Image (`txt2img`)

Générer une image avec **RunPod Flux (Cloud)** ou **en Local (GPU RTX 2080 / Diffusers)** :

```bash
# Mode Cloud (RunPod Flux Schnell)
cargo run -- txt2img \
  --prompt "Portrait photograph of an astronaut looking at the stars" \
  --width 1024 \
  --height 1024 \
  --steps 4 \
  --output astro.png

# Mode Local GPU (ex: RTX 2080 avec SDXL Turbo en < 2 secondes)
cargo run -- txt2img \
  --prompt "Portrait photograph of an astronaut looking at the stars, 8k uhd" \
  --local \
  --local-model "stabilityai/sdxl-turbo" \
  --width 512 \
  --height 512 \
  --steps 2 \
  --output astro_local.png
```

---

## 💻 Exécution Locale & Déploiement sur Machine GPU (ex: RTX 2080)

Si vous clonez ce projet sur une machine dotée d'une carte graphique Nvidia (comme une RTX 2080 avec 8 Go de VRAM) :

1. **Installer les dépendances Python requises :**
   ```bash
   pip install -r requirements-local-gpu.txt
   ```
   *(Pour PyTorch avec CUDA : `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`)*

2. **Accélération FaceSwap automatique :**
   - Le moteur InsightFace détecte automatiquement la présence de CUDA via `onnxruntime-gpu` et exécute le swap sur votre GPU, ce qui est quasi-instantané.

3. **Génération d'images locale gratuite :**
   - Vous pouvez exécuter `txt2img` ou `txt2vid` avec `--local` (ou basculer le sélecteur dans l'interface Web sur `Local GPU`).
   - Modèles recommandés pour 8 Go VRAM : `stabilityai/sdxl-turbo` (2 steps), `ByteDance/SDXL-Lightning` (4-8 steps) ou `runwayml/stable-diffusion-v1-5`.

---

### 6. Tester l'enrichissement de Prompt (`enhance`)

```bash
cargo run -- enhance \
  --prompt "A cozy coffee shop in winter"
```

---

## 🏗️ Structure du Code

```
src/
├── main.rs                   # Point d'entrée CLI et orchestration
├── cli.rs                    # Définition des sous-commandes Clap
├── server/                   # Serveur Web & API REST RunPod Studio
│   └── mod.rs                # Routes API média, streaming et exécution asynchrone
├── client.rs                 # Client HTTP asynchrone RunPod avec polling et spinners
├── models.rs                 # Modèles de données (Flux, Wan, FaceSwap, Vid2Vid, LLM, JobStatus)
├── utils.rs                  # Helpers d'encodage Data-URI, téléchargement streaming et preview
└── pipeline/
    ├── mod.rs                # Trait Stage & Moteur de Pipeline
    ├── context.rs            # PipelineContext partagé entre les étapes
    └── stages/
        ├── prompt_enhance.rs     # Étape LLM (Qwen)
        ├── text_to_image.rs      # Étape Flux Schnell / Dev
        ├── image_to_video.rs     # Étape Wan 2.5 (Img2Vid)
        ├── video_to_video.rs     # Étape Wan 2.5 / ComfyUI (Vid2Vid)
        ├── face_swap.rs          # Étape Face Swap (ReActor / InsightFace)
        ├── interactive_review.rs # Étape de validation utilisateur / preview
        └── download.rs           # Étape de téléchargement de médias avec progression
web/                          # Interface Web Studio (HTML5 / Vanilla CSS / ES Modules)
├── index.html                # Tableau de bord Studio & Galerie
├── style.css                 # Thème sombre moderne & Glassmorphism
└── app.js                    # Logique interactive, upload & polling
```
