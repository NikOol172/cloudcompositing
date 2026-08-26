use crate::client::RunpodClient;
use crate::models::{extract_media_url, Vid2VidInput, VideoModel};
use crate::pipeline::{PipelineContext, Stage};
use crate::utils::resolve_video_input;
use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use console::style;
use std::time::Duration;

pub const DEFAULT_VID2VID_ENDPOINT: &str = "wan-2-5";

/// Étape de transformation / restyling d'une vidéo existante (Video-to-Video via Wan 2.5, LTX-Video 2.5 ou MiniMax-H3).
pub struct VideoToVideoStage {
    pub model: VideoModel,
    pub endpoint_id: String,
    pub video_source_override: Option<String>,
    pub prompt_override: Option<String>,
    pub duration: u32,
    pub strength: f32,
    pub resolution: String,
    pub seed: Option<u64>,
    pub negative_prompt: Option<String>,
    pub timeout: Duration,
}

impl Default for VideoToVideoStage {
    fn default() -> Self {
        Self {
            model: VideoModel::Wan2_5,
            endpoint_id: DEFAULT_VID2VID_ENDPOINT.to_string(),
            video_source_override: None,
            prompt_override: None,
            duration: 5,
            strength: 0.65,
            resolution: "720p".to_string(),
            seed: None,
            negative_prompt: None,
            timeout: Duration::from_secs(900),
        }
    }
}

impl VideoToVideoStage {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_model(mut self, model: VideoModel) -> Self {
        self.model = model;
        if self.endpoint_id == DEFAULT_VID2VID_ENDPOINT {
            self.endpoint_id = model.default_endpoint().to_string();
        }
        self
    }

    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint_id = endpoint.into();
        self
    }

    pub fn with_video(mut self, video: impl Into<String>) -> Self {
        self.video_source_override = Some(video.into());
        self
    }

    pub fn with_duration(mut self, duration: u32) -> Self {
        self.duration = duration;
        self
    }

    pub fn with_strength(mut self, strength: f32) -> Self {
        self.strength = strength;
        self
    }

    pub fn with_resolution(mut self, resolution: impl Into<String>) -> Self {
        self.resolution = resolution.into();
        self
    }

    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt_override = Some(prompt.into());
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn with_negative_prompt(mut self, neg_prompt: impl Into<String>) -> Self {
        self.negative_prompt = Some(neg_prompt.into());
        self
    }
}

#[async_trait]
impl Stage for VideoToVideoStage {
    fn name(&self) -> &str {
        match self.model {
            VideoModel::Wan2_5 => "Modification Vidéo (Video-to-Video / Wan 2.5)",
            VideoModel::Ltx2_5 => "Modification Vidéo (Video-to-Video / LTX-Video 2.5)",
            VideoModel::MiniMaxH3 => "Modification Vidéo (Video-to-Video / MiniMax-H3)",
        }
    }

    async fn execute(&self, ctx: &mut PipelineContext, client: &RunpodClient) -> Result<()> {
        // Détermination de la vidéo source (URL ou fichier local)
        let raw_video_input = if let Some(ref override_vid) = self.video_source_override {
            override_vid.clone()
        } else if let Some(ref url) = ctx.video_url {
            url.clone()
        } else if let Some(ref path) = ctx.downloaded_video_path {
            path.to_string_lossy().to_string()
        } else {
            bail!("Aucune vidéo source fournie pour l'étape Video-to-Video (ni dans le contexte, ni en paramètre).");
        };

        println!("  Préparation de la vidéo source...");
        let resolved_video = resolve_video_input(&raw_video_input)
            .await
            .with_context(|| format!("Impossible de préparer la vidéo source '{}'", raw_video_input))?;

        if resolved_video.starts_with("data:") {
            println!("  Vidéo locale encodée avec succès en Data-URI (base64).");
        } else {
            println!("  Vidéo distante référencée : {}", style(&resolved_video).dim());
        }

        let prompt = self
            .prompt_override
            .as_deref()
            .unwrap_or_else(|| ctx.effective_prompt());

        let input = Vid2VidInput {
            prompt: prompt.to_string(),
            image: resolved_video.clone(),
            video: Some(resolved_video),
            duration: Some(self.duration),
            strength: Some(self.strength),
            resolution: Some(self.resolution.clone()),
            seed: self.seed,
            negative_prompt: self.negative_prompt.clone(),
        };

        println!(
            "  Modèle : {}, Durée {}s, Force (strength) {:.2}, Résolution {}, Prompt: {}",
            style(self.model.display_name()).bold().cyan(),
            self.duration,
            self.strength,
            self.resolution,
            style(prompt).italic()
        );

        let status = client
            .run_and_poll(
                &self.endpoint_id,
                &input,
                Duration::from_secs(5),
                self.timeout,
                &format!("Modification Vidéo ({})", self.model.display_name()),
            )
            .await?;

        let output_val = status
            .output
            .ok_or_else(|| anyhow::anyhow!("Aucun champ 'output' retourné pour la vidéo"))?;

        let video_url = extract_media_url(&output_val)
            .ok_or_else(|| anyhow::anyhow!("Impossible d'extraire l'URL de la vidéo produite : {:?}", output_val))?;

        println!(
            "  {} Vidéo transformée : {}",
            style("✓").green().bold(),
            style(&video_url).cyan().underlined()
        );

        ctx.video_url = Some(video_url);

        Ok(())
    }
}
