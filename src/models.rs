use serde::{Deserialize, Serialize};

/// Représentation des différents états possibles d'un job RunPod Serverless.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum JobState {
    InQueue,
    InProgress,
    Completed,
    Failed,
    Cancelled,
    TimedOut,
    #[serde(other)]
    Unknown,
}

#[allow(dead_code)]
impl JobState {
    pub fn is_finished(&self) -> bool {
        matches!(
            self,
            JobState::Completed | JobState::Failed | JobState::Cancelled | JobState::TimedOut
        )
    }

    pub fn is_success(&self) -> bool {
        matches!(self, JobState::Completed)
    }
}

/// Réponse lors de la soumission asynchrone d'un job via `/run`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRunResponse {
    pub id: String,
    pub status: Option<JobState>,
}

/// Réponse retournée par l'endpoint `/status/{id}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobStatusResponse {
    pub id: String,
    pub status: JobState,
    pub output: Option<serde_json::Value>,
    pub error: Option<serde_json::Value>,
    #[serde(rename = "executionTime")]
    pub execution_time: Option<u64>,
    #[serde(rename = "delayTime")]
    pub delay_time: Option<u64>,
}

/// Enveloppe générique pour les entrées RunPod `{"input": ...}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunpodPayload<T> {
    pub input: T,
}

/// Paramètres d'entrée pour la génération Text-to-Image (Flux 1 Schnell / Dev / SDXL).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluxInput {
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_inference_steps: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guidance_scale: Option<f32>,
}

/// Paramètres d'entrée pour la génération Image-to-Image (Flux / SDXL / ComfyUI).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Img2ImgInput {
    pub prompt: String,
    pub image: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask_image: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strength: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_inference_steps: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guidance_scale: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
}

impl Default for Img2ImgInput {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            image: String::new(),
            mask_image: None,
            strength: Some(0.70),
            width: Some(768),
            height: Some(1344),
            num_inference_steps: Some(4),
            guidance_scale: None,
            seed: None,
            negative_prompt: None,
        }
    }
}

/// Modèles vidéo pris en charge (Wan 2.5, LTX-Video 2.5, MiniMax-H3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum VideoModel {
    #[serde(rename = "wan-2-5", alias = "wan", alias = "wan2.5", alias = "wan-2.5")]
    Wan2_5,
    #[serde(rename = "ltx-2-5", alias = "ltx", alias = "ltx-video", alias = "ltx-video-2-5", alias = "ltx2.5")]
    Ltx2_5,
    #[serde(rename = "minimax-h3", alias = "minimax", alias = "hailuo", alias = "hailuo-3", alias = "minimax_h3")]
    MiniMaxH3,
}

impl Default for VideoModel {
    fn default() -> Self {
        Self::Wan2_5
    }
}

impl VideoModel {
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Wan2_5 => "Wan 2.5",
            Self::Ltx2_5 => "LTX-Video 2.5",
            Self::MiniMaxH3 => "MiniMax-H3 (Hailuo 3)",
        }
    }

    pub fn default_endpoint(&self) -> &'static str {
        match self {
            Self::Wan2_5 => "wan-2-5",
            Self::Ltx2_5 => "ltx-video-2-5",
            Self::MiniMaxH3 => "minimax-h3",
        }
    }
}

impl std::str::FromStr for VideoModel {
    type Err = std::convert::Infallible;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.to_lowercase();
        if lower.contains("ltx") {
            Ok(Self::Ltx2_5)
        } else if lower.contains("minimax") || lower.contains("hailuo") || lower.contains("h3") {
            Ok(Self::MiniMaxH3)
        } else {
            Ok(Self::Wan2_5)
        }
    }
}

impl std::fmt::Display for VideoModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Wan2_5 => write!(f, "wan-2-5"),
            Self::Ltx2_5 => write!(f, "ltx-2-5"),
            Self::MiniMaxH3 => write!(f, "minimax-h3"),
        }
    }
}

/// Paramètres d'entrée pour la génération Image-to-Video / Text-to-Video (Wan 2.5, LTX-Video 2.5, MiniMax-H3).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WanInput {
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolution: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

pub type VideoGenInput = WanInput;

impl Default for WanInput {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            image: None,
            duration: Some(5),
            resolution: Some("720p".to_string()),
            seed: None,
            negative_prompt: None,
            model: None,
        }
    }
}

/// Paramètres d'entrée pour la modification Video-to-Video (Wan 2.5 / ComfyUI).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vid2VidInput {
    pub prompt: String,
    pub image: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strength: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolution: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
}

impl Default for Vid2VidInput {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            image: String::new(),
            video: None,
            duration: Some(5),
            strength: Some(0.65),
            resolution: Some("720p".to_string()),
            seed: None,
            negative_prompt: None,
        }
    }
}

/// Paramètres d'entrée pour le remplacement de visage (Face Swap / ReActor / InsightFace).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceSwapInput {
    /// Image du visage source (URL ou Data-URI).
    pub source_image: String,
    /// Image du visage source (alias pour certains serveurs).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_face: Option<String>,
    /// Image cible où insérer le visage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_image: Option<String>,
    /// Vidéo cible où insérer le visage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_video: Option<String>,
    /// Cible générique (image ou vidéo).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    /// Index du visage cible à remplacer (par défaut 0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub face_index: Option<u32>,
    /// Activer la restauration du visage (GFPGAN / CodeFormer).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub restore_face: Option<bool>,
}

impl Default for FaceSwapInput {
    fn default() -> Self {
        Self {
            source_image: String::new(),
            source_face: None,
            target_image: None,
            target_video: None,
            target: None,
            face_index: Some(0),
            restore_face: Some(true),
        }
    }
}

/// Messages pour les requêtes de chat LLM (compatible OpenAI / Qwen).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Requête de chat completion (Qwen3 / LLM endpoints).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

/// Choix retourné par l'endpoint de chat LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub message: ChatMessage,
    pub index: Option<u32>,
    pub finish_reason: Option<String>,
}

/// Réponse retournée par l'endpoint de chat LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: Option<String>,
    pub choices: Vec<ChatChoice>,
}

/// Helper robuste pour extraire une URL de média (image ou vidéo) à partir des formats variés de sortie RunPod.
pub fn extract_media_url(val: &serde_json::Value) -> Option<String> {
    match val {
        serde_json::Value::String(s) => {
            if s.starts_with("http://")
                || s.starts_with("https://")
                || s.starts_with("data:image/")
                || s.starts_with("data:video/")
            {
                Some(s.clone())
            } else {
                None
            }
        }
        serde_json::Value::Array(arr) => arr.first().and_then(extract_media_url),
        serde_json::Value::Object(map) => {
            // Clés courantes retournées par les endpoints RunPod (Flux, Wan, Stable Diffusion, ComfyUI, etc.)
            let candidate_keys = [
                "video_url",
                "video",
                "result",
                "image",
                "images",
                "output",
                "url",
                "message",
                "file",
            ];

            for key in candidate_keys {
                if let Some(field) = map.get(key) {
                    if let Some(url) = extract_media_url(field) {
                        return Some(url);
                    }
                }
            }
            None
        }
        _ => None,
    }
}


/// Informations de compte RunPod (récupérées via GraphQL).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunpodAccountInfo {
    pub id: Option<String>,
    pub email: Option<String>,
    pub client_balance: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::str::FromStr;

    #[test]
    fn test_extract_media_url_formats() {
        let v1 = json!("https://example.com/video.mp4");
        assert_eq!(extract_media_url(&v1), Some("https://example.com/video.mp4".to_string()));

        let v2 = json!({"result": "https://example.com/image.png"});
        assert_eq!(extract_media_url(&v2), Some("https://example.com/image.png".to_string()));

        let v3 = json!({"images": ["https://example.com/flux1.png", "https://example.com/flux2.png"]});
        assert_eq!(extract_media_url(&v3), Some("https://example.com/flux1.png".to_string()));

        let v4 = json!({"output": {"video_url": "https://example.com/wan.mp4"}});
        assert_eq!(extract_media_url(&v4), Some("https://example.com/wan.mp4".to_string()));
    }

    #[test]
    fn test_video_model_parsing_and_defaults() {
        assert_eq!(VideoModel::from_str("wan-2-5").unwrap(), VideoModel::Wan2_5);
        assert_eq!(VideoModel::from_str("wan").unwrap(), VideoModel::Wan2_5);
        assert_eq!(VideoModel::from_str("ltx-2-5").unwrap(), VideoModel::Ltx2_5);
        assert_eq!(VideoModel::from_str("ltx").unwrap(), VideoModel::Ltx2_5);
        assert_eq!(VideoModel::from_str("minimax-h3").unwrap(), VideoModel::MiniMaxH3);
        assert_eq!(VideoModel::from_str("hailuo-3").unwrap(), VideoModel::MiniMaxH3);
        assert_eq!(VideoModel::from_str("hailuo").unwrap(), VideoModel::MiniMaxH3);

        assert_eq!(VideoModel::Wan2_5.default_endpoint(), "wan-2-5");
        assert_eq!(VideoModel::Ltx2_5.default_endpoint(), "ltx-video-2-5");
        assert_eq!(VideoModel::MiniMaxH3.default_endpoint(), "minimax-h3");
    }

    #[test]
    fn test_img2img_input_serialization() {
        let input = Img2ImgInput {
            prompt: "A pickleball player with pink compression socks".to_string(),
            image: "https://example.com/socks.png".to_string(),
            mask_image: Some("https://example.com/mask.png".to_string()),
            strength: Some(0.75),
            width: Some(768),
            height: Some(1344),
            num_inference_steps: Some(4),
            guidance_scale: None,
            seed: Some(42),
            negative_prompt: None,
        };

        let serialized = serde_json::to_string(&input).unwrap();
        assert!(serialized.contains("pickleball player"));
        assert!(serialized.contains("0.75"));
        assert!(serialized.contains("mask.png"));
    }
}
