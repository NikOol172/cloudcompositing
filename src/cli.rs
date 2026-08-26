use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

/// Application Rust d'orchestration de pipelines RunPod (Text-to-Image, Image-to-Video, Text-to-Video).
#[derive(Parser, Debug)]
#[command(name = "runpod-pipeline", author, version, about, long_about = None)]
pub struct Cli {
    /// Clé API RunPod (par défaut lue depuis la variable d'environnement RUNPOD_API_KEY ou .env).
    #[arg(long, env = "RUNPOD_API_KEY", hide_env_values = true)]
    pub api_key: Option<String>,

    /// URL de base de l'API RunPod.
    #[arg(long, default_value = "https://api.runpod.ai/v2")]
    pub base_url: String,

    /// Active les logs détaillés (mode verbeux).
    #[arg(short, long, global = true)]
    pub verbose: bool,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Générer une vidéo à partir d'une image existante (fichier local ou URL).
    #[command(name = "img2vid", alias = "img2-vid")]
    Img2Vid(Img2VidArgs),

    /// Pipeline complet Text-to-Video (Prompt -> Flux -> [Validation] -> Wan 2.5).
    #[command(name = "txt2vid", alias = "txt2-vid")]
    Txt2Vid(Txt2VidArgs),

    /// Générer une image à partir d'un prompt textuel (Flux Schnell / Dev).
    #[command(name = "txt2img", alias = "txt2-img")]
    Txt2Img(Txt2ImgArgs),

    /// Transformer ou adapter une image existante via Image-to-Image (Flux / SDXL).
    #[command(name = "img2img", alias = "img2-img")]
    Img2Img(Img2ImgArgs),

    /// Modifier ou restyler une vidéo existante via Video-to-Video.
    #[command(name = "vid2vid", alias = "vid2-vid")]
    Vid2Vid(Vid2VidArgs),

    /// Remplacer un visage sur une image ou vidéo (Face Swap).
    #[command(name = "faceswap", alias = "face-swap")]
    FaceSwap(FaceSwapArgs),

    /// Lancer l'interface web RunPod Studio & Media Manager.
    #[command(name = "serve", alias = "ui")]
    Serve(ServeArgs),

    /// Tester l'enrichissement d'un prompt via le LLM (Qwen3).
    #[command(name = "enhance")]
    Enhance(EnhanceArgs),
}

#[derive(Args, Debug)]
pub struct Img2VidArgs {
    /// Chemin vers une image locale (ex: ./image.png) ou URL HTTP(S).
    #[arg(short, long)]
    pub image: String,

    /// Prompt descriptif de l'animation vidéo souhaitée.
    #[arg(short, long, default_value = "")]
    pub prompt: String,

    /// Modèle vidéo à utiliser (wan-2-5, ltx-2-5, minimax-h3).
    #[arg(short, long, default_value = "wan-2-5")]
    pub model: String,

    /// Endpoint RunPod personnalisé pour la génération vidéo (surcharge optionnelle).
    #[arg(long)]
    pub endpoint: Option<String>,

    /// Durée de la vidéo en secondes.
    #[arg(short, long, default_value_t = 5)]
    pub duration: u32,

    /// Résolution vidéo (ex: 480p, 720p).
    #[arg(short, long, default_value = "720p")]
    pub resolution: String,

    /// Graine aléatoire (seed) optionnelle.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Chemin de sortie du fichier vidéo MP4 téléchargé.
    #[arg(short, long, default_value = "output_video.mp4")]
    pub output: PathBuf,
}

#[derive(Args, Debug)]
pub struct Txt2VidArgs {
    /// Idée ou description textuelle pour la génération.
    #[arg(short, long)]
    pub prompt: String,

    /// Modèle vidéo à utiliser (wan-2-5, ltx-2-5, minimax-h3).
    #[arg(short, long, default_value = "wan-2-5")]
    pub model: String,

    /// Activer l'enrichissement du prompt via le LLM (Qwen3).
    #[arg(long, default_value_t = false)]
    pub enhance_prompt: bool,

    /// Mode interactif : prévisualise l'image et demande confirmation avant la vidéo.
    #[arg(short, long, default_value_t = false)]
    pub interactive: bool,

    /// Endpoint pour la génération de l'image (Flux).
    #[arg(long, default_value = "black-forest-labs-flux-1-schnell")]
    pub flux_endpoint: String,

    /// Endpoint RunPod personnalisé pour la génération de la vidéo (surcharge optionnelle).
    #[arg(long)]
    pub video_endpoint: Option<String>,

    /// Endpoint pour la génération de la vidéo (Wan) [rétrocompatibilité].
    #[arg(long, default_value = "wan-2-5")]
    pub wan_endpoint: String,

    /// Largeur de l'image générée.
    #[arg(long, default_value_t = 768)]
    pub width: u32,

    /// Hauteur de l'image générée.
    #[arg(long, default_value_t = 1344)]
    pub height: u32,

    /// Nombre de pas d'inférence pour l'image.
    #[arg(long, default_value_t = 4)]
    pub steps: u32,

    /// Durée de la vidéo finale en secondes.
    #[arg(long, default_value_t = 5)]
    pub duration: u32,

    /// Résolution de la vidéo.
    #[arg(long, default_value = "720p")]
    pub resolution: String,

    /// Générer l'image en local via GPU/diffusers plutôt que via RunPod.
    #[arg(long, default_value_t = false)]
    pub local: bool,

    /// Modèle HuggingFace pour la génération locale (ex: stabilityai/sdxl-turbo, ByteDance/SDXL-Lightning).
    #[arg(long)]
    pub local_model: Option<String>,

    /// Chemin de sauvegarde de l'image intermédiaire.
    #[arg(long, default_value = "output_image.png")]
    pub image_output: PathBuf,

    /// Chemin de sauvegarde de la vidéo finale.
    #[arg(long, default_value = "output_video.mp4")]
    pub video_output: PathBuf,
}

#[derive(Args, Debug)]
pub struct Txt2ImgArgs {
    /// Prompt textuel pour l'image.
    #[arg(short, long)]
    pub prompt: String,

    /// Activer l'enrichissement du prompt via LLM.
    #[arg(long, default_value_t = false)]
    pub enhance_prompt: bool,

    /// Générer l'image en local via GPU/diffusers plutôt que via RunPod.
    #[arg(long, default_value_t = false)]
    pub local: bool,

    /// Modèle HuggingFace pour la génération locale (ex: stabilityai/sdxl-turbo, ByteDance/SDXL-Lightning, runwayml/stable-diffusion-v1-5).
    #[arg(long)]
    pub local_model: Option<String>,

    /// Endpoint RunPod pour la génération d'image (si mode cloud).
    #[arg(long, default_value = "black-forest-labs-flux-1-schnell")]
    pub endpoint: String,

    /// Largeur de l'image.
    #[arg(long, default_value_t = 768)]
    pub width: u32,

    /// Hauteur de l'image.
    #[arg(long, default_value_t = 1344)]
    pub height: u32,

    /// Nombre de pas d'inférence.
    #[arg(long, default_value_t = 4)]
    pub steps: u32,

    /// Graine aléatoire optionnelle.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Chemin de sortie du fichier image PNG.
    #[arg(short, long, default_value = "output_image.png")]
    pub output: PathBuf,
}

#[derive(Args, Debug)]
pub struct Img2ImgArgs {
    /// Chemin vers l'image source (fichier local ou URL).
    #[arg(short, long)]
    pub image: String,

    /// Prompt descriptif de la transformation ou de la nouvelle scène souhaitée.
    #[arg(short, long)]
    pub prompt: String,

    /// Masque optionnel pour l'inpainting (fichier image local noir & blanc ou URL).
    #[arg(short, long)]
    pub mask: Option<String>,

    /// Force de transformation / denoise (entre 0.05 et 1.0, défaut: 0.70).
    #[arg(long, default_value_t = 0.70)]
    pub strength: f32,

    /// Générer l'image en local via la carte graphique (PyTorch / Diffusers) au lieu du Cloud.
    #[arg(long, default_value_t = false)]
    pub local: bool,

    /// Modèle HuggingFace pour la génération locale (ex: stabilityai/sdxl-turbo, ByteDance/SDXL-Lightning).
    #[arg(long)]
    pub local_model: Option<String>,

    /// Endpoint RunPod pour la génération Image-to-Image (si mode cloud).
    #[arg(long, default_value = "black-forest-labs-flux-1-schnell")]
    pub endpoint: String,

    /// Largeur de l'image (0 = automatique selon l'image source).
    #[arg(long, default_value_t = 0)]
    pub width: u32,

    /// Hauteur de l'image (0 = automatique selon l'image source).
    #[arg(long, default_value_t = 0)]
    pub height: u32,

    /// Nombre de pas d'inférence (steps).
    #[arg(long, default_value_t = 4)]
    pub steps: u32,

    /// Graine aléatoire optionnelle.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Chemin de sortie du fichier image résultat.
    #[arg(short, long, default_value = "output_img2img.png")]
    pub output: PathBuf,
}

#[derive(Args, Debug)]
pub struct EnhanceArgs {
    /// Prompt simple à enrichir.
    #[arg(short, long)]
    pub prompt: String,

    /// Endpoint RunPod LLM.
    #[arg(long, default_value = "qwen3-32b-awq")]
    pub endpoint: String,

    /// Modèle LLM.
    #[arg(long, default_value = "Qwen/Qwen3-32B-AWQ")]
    pub model: String,
}

#[derive(Args, Debug)]
pub struct Vid2VidArgs {
    /// Chemin vers une vidéo locale (ex: ./video.mp4) ou URL HTTP(S).
    #[arg(long)]
    pub video: String,

    /// Prompt descriptif de la modification ou du nouveau style souhaité.
    #[arg(short, long)]
    pub prompt: String,

    /// Modèle vidéo à utiliser (wan-2-5, ltx-2-5, minimax-h3).
    #[arg(short, long, default_value = "wan-2-5")]
    pub model: String,

    /// Endpoint RunPod pour le traitement Video-to-Video (surcharge optionnelle).
    #[arg(long)]
    pub endpoint: Option<String>,

    /// Durée de la vidéo en secondes.
    #[arg(short, long, default_value_t = 5)]
    pub duration: u32,

    /// Force de transformation / denoising (entre 0.1 et 1.0).
    #[arg(long, default_value_t = 0.65)]
    pub strength: f32,

    /// Résolution vidéo (ex: 480p, 720p).
    #[arg(short, long, default_value = "720p")]
    pub resolution: String,

    /// Graine aléatoire (seed) optionnelle.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Prompt négatif optionnel.
    #[arg(long)]
    pub negative_prompt: Option<String>,

    /// Chemin de sortie du fichier vidéo MP4 modifié.
    #[arg(short, long, default_value = "output_vid2vid.mp4")]
    pub output: PathBuf,
}

#[derive(Args, Debug)]
pub struct FaceSwapArgs {
    /// Photo source contenant le visage à insérer (fichier local ou URL).
    #[arg(short, long)]
    pub source: String,

    /// Média cible (image ou vidéo) sur lequel appliquer le visage.
    #[arg(short, long)]
    pub target: String,

    /// Endpoint RunPod pour le Face Swap (ex: faceswap / reactor / comfyui).
    #[arg(long, default_value = "faceswap")]
    pub endpoint: String,

    /// Activer la restauration haute définition du visage (GFPGAN / CodeFormer).
    #[arg(long, default_value_t = true)]
    pub restore_face: bool,

    /// Index du visage cible à remplacer (par défaut 0 pour le visage principal).
    #[arg(long, default_value_t = 0)]
    pub face_index: u32,

    /// Chemin de sortie pour sauvegarder le résultat (image ou vidéo).
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct ServeArgs {
    /// Port d'écoute du serveur web local.
    #[arg(short, long, default_value_t = 3000)]
    pub port: u16,

    /// Ouvrir automatiquement le navigateur par défaut.
    #[arg(long, default_value_t = true)]
    pub open: bool,
}
