use crate::client::RunpodClient;
use crate::models::{extract_media_url, VideoModel, WanInput};
use crate::pipeline::{PipelineContext, Stage};
use crate::utils::resolve_image_input;
use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use console::style;
use std::time::Duration;

pub const DEFAULT_WAN_ENDPOINT: &str = "wan-2-5";

/// Étape de génération de vidéo à partir d'une image (Image-to-Video via Wan 2.5, LTX-Video 2.5 ou MiniMax-H3).
pub struct ImageToVideoStage {
    pub model: VideoModel,
    pub endpoint_id: String,
    pub image_source_override: Option<String>,
    pub prompt_override: Option<String>,
    pub duration: u32,
    pub resolution: String,
    pub seed: Option<u64>,
    pub negative_prompt: Option<String>,
    pub timeout: Duration,
}

impl Default for ImageToVideoStage {
    fn default() -> Self {
        Self {
            model: VideoModel::Wan2_5,
            endpoint_id: DEFAULT_WAN_ENDPOINT.to_string(),
            image_source_override: None,
            prompt_override: None,
            duration: 5,
            resolution: "720p".to_string(),
            seed: None,
            negative_prompt: None,
            timeout: Duration::from_secs(600),
        }
    }
}

impl ImageToVideoStage {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_model(mut self, model: VideoModel) -> Self {
        self.model = model;
        if self.endpoint_id == DEFAULT_WAN_ENDPOINT {
            self.endpoint_id = model.default_endpoint().to_string();
        }
        self
    }

    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint_id = endpoint.into();
        self
    }

    pub fn with_image(mut self, image: impl Into<String>) -> Self {
        self.image_source_override = Some(image.into());
        self
    }

    pub fn with_duration(mut self, duration: u32) -> Self {
        self.duration = duration;
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
}

#[async_trait]
impl Stage for ImageToVideoStage {
    fn name(&self) -> &str {
        match self.model {
            VideoModel::Wan2_5 => "Génération Vidéo (Image-to-Video / Wan 2.5)",
            VideoModel::Ltx2_5 => "Génération Vidéo (Image-to-Video / LTX-Video 2.5)",
            VideoModel::MiniMaxH3 => "Génération Vidéo (Image-to-Video / MiniMax-H3)",
        }
    }

    async fn execute(&self, ctx: &mut PipelineContext, client: &RunpodClient) -> Result<()> {
        // Détermination de l'image source (URL ou image locale)
        let raw_image_input = if let Some(ref override_img) = self.image_source_override {
            override_img.clone()
        } else if let Some(ref url) = ctx.image_url {
            url.clone()
        } else if let Some(ref path) = ctx.local_image_path {
            path.to_string_lossy().to_string()
        } else {
            bail!("Aucune image source fournie pour l'étape Image-to-Video (ni dans le contexte, ni en paramètre).");
        };

        println!("  Préparation de l'image source...");
        let resolved_image = resolve_image_input(&raw_image_input)
            .await
            .with_context(|| format!("Impossible de préparer l'image source '{}'", raw_image_input))?;

        if resolved_image.starts_with("data:") {
            println!("  Image locale encodée avec succès en Data-URI (base64).");
        } else {
            println!("  Image distante référencée : {}", style(&resolved_image).dim());
        }

        let prompt = self
            .prompt_override
            .as_deref()
            .unwrap_or_else(|| ctx.effective_prompt());

        let input = WanInput {
            prompt: prompt.to_string(),
            image: Some(resolved_image),
            duration: Some(self.duration),
            resolution: Some(self.resolution.clone()),
            seed: self.seed,
            negative_prompt: self.negative_prompt.clone(),
            model: Some(self.model.to_string()),
        };

        println!(
            "  Modèle : {}, Durée {}s, Résolution {}, Prompt: {}",
            style(self.model.display_name()).bold().cyan(),
            self.duration,
            self.resolution,
            style(prompt).italic()
        );

        let status = client
            .run_and_poll(
                &self.endpoint_id,
                &input,
                Duration::from_secs(5),
                self.timeout,
                &format!("Animation Vidéo {}", self.model.display_name()),
            )
            .await?;

        let output_val = status
            .output
            .ok_or_else(|| anyhow::anyhow!("Aucun champ 'output' retourné pour la vidéo"))?;

        let video_url = extract_media_url(&output_val)
            .ok_or_else(|| anyhow::anyhow!("Impossible d'extraire l'URL de la vidéo de: {:?}", output_val))?;

        println!(
            "  {} Vidéo produite : {}",
            style("✓").green().bold(),
            style(&video_url).cyan().underlined()
        );

        ctx.video_url = Some(video_url);

        Ok(())
    }
}
