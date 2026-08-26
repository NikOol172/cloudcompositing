use crate::client::RunpodClient;
use crate::models::{extract_media_url, FluxInput};
use crate::pipeline::{PipelineContext, Stage};
use anyhow::{Context, Result};
use async_trait::async_trait;
use console::style;
use rand::Rng;
use std::time::Duration;
use tokio::io::AsyncBufReadExt;

pub const DEFAULT_FLUX_ENDPOINT: &str = "black-forest-labs-flux-1-schnell";
pub const DEFAULT_LOCAL_MODEL: &str = "stabilityai/sdxl-turbo";

/// Étape de génération d'image à partir de texte (Text-to-Image via RunPod Flux ou Diffusers Local).
pub struct TextToImageStage {
    pub endpoint_id: String,
    pub width: u32,
    pub height: u32,
    pub num_inference_steps: u32,
    pub seed: Option<u64>,
    pub guidance_scale: Option<f32>,
    pub timeout: Duration,
    pub local: bool,
    pub local_model: Option<String>,
}

impl Default for TextToImageStage {
    fn default() -> Self {
        Self {
            endpoint_id: DEFAULT_FLUX_ENDPOINT.to_string(),
            width: 768,
            height: 1344,
            num_inference_steps: 4,
            seed: None,
            guidance_scale: None,
            timeout: Duration::from_secs(300),
            local: false,
            local_model: None,
        }
    }
}

impl TextToImageStage {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint_id = endpoint.into();
        self
    }

    pub fn with_dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    pub fn with_steps(mut self, steps: u32) -> Self {
        self.num_inference_steps = steps;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn with_local(mut self, local: bool) -> Self {
        self.local = local;
        self
    }

    pub fn with_local_model(mut self, model: impl Into<String>) -> Self {
        self.local_model = Some(model.into());
        self
    }
}

#[async_trait]
impl Stage for TextToImageStage {
    fn name(&self) -> &str {
        if self.local {
            "Génération d'Image Locale (GPU / Diffusers)"
        } else {
            "Génération d'Image Cloud (Text-to-Image / Flux)"
        }
    }

    async fn execute(&self, ctx: &mut PipelineContext, client: &RunpodClient) -> Result<()> {
        let prompt = ctx.effective_prompt().to_string();
        let seed = self
            .seed
            .unwrap_or_else(|| rand::thread_rng().gen_range(1..999_999_999));

        if self.local {
            let model = self
                .local_model
                .as_deref()
                .unwrap_or(DEFAULT_LOCAL_MODEL);

            let temp_out = std::env::temp_dir().join(format!("local_t2i_{}.png", rand::random::<u32>()));
            let temp_out_str = temp_out.to_string_lossy().to_string();

            println!(
                "  Mode: {}, Modèle: {}, {}x{}, {} steps, Seed: {}",
                style("Local GPU/CPU").magenta().bold(),
                style(model).cyan(),
                self.width,
                self.height,
                self.num_inference_steps,
                seed
            );
            println!("  Lancement du moteur local diffusers...");

            let mut cmd = tokio::process::Command::new("python3");
            cmd.args([
                "src/txt2img_engine.py",
                "--prompt",
                &prompt,
                "--output",
                &temp_out_str,
                "--model",
                model,
                "--width",
                &self.width.to_string(),
                "--height",
                &self.height.to_string(),
                "--steps",
                &self.num_inference_steps.to_string(),
                "--seed",
                &seed.to_string(),
            ]);

            if let Some(cfg) = self.guidance_scale {
                cmd.args(["--guidance-scale", &cfg.to_string()]);
            }

            if let Ok(home) = std::env::var("HOME") {
                let full_paths = format!(
                    "{}/.local/lib/python3.10/site-packages:/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages:/usr/lib/python3.10",
                    home
                );
                let current_pypath = std::env::var("PYTHONPATH").unwrap_or_default();
                let new_pypath = if current_pypath.is_empty() {
                    full_paths
                } else {
                    format!("{}:{}", current_pypath, full_paths)
                };
                cmd.env("PYTHONPATH", new_pypath);
            }

            cmd.stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped());

            let mut child = cmd
                .spawn()
                .context("Impossible de démarrer le script txt2img_engine.py")?;

            let stdout = child.stdout.take().unwrap();
            let stderr = child.stderr.take().unwrap();

            let mut stdout_reader = tokio::io::BufReader::new(stdout).lines();
            let mut stderr_reader = tokio::io::BufReader::new(stderr).lines();

            loop {
                tokio::select! {
                    line = stdout_reader.next_line() => {
                        match line {
                            Ok(Some(l)) => println!("  {}", l),
                            _ => break,
                        }
                    }
                    line = stderr_reader.next_line() => {
                        match line {
                            Ok(Some(l)) => eprintln!("  [stderr] {}", l),
                            _ => {}
                        }
                    }
                }
            }

            let status = child.wait().await?;
            if !status.success() {
                anyhow::bail!("Échec de l'exécution du moteur Text-to-Image local.");
            }

            println!(
                "  {} Image locale produite avec succès : {}",
                style("✓").green().bold(),
                style(&temp_out_str).cyan().underlined()
            );

            ctx.image_url = Some(temp_out_str);
            ctx.set_meta("t2i_seed", seed);
            ctx.set_meta("t2i_engine", "local_diffusers");
            return Ok(());
        }

        // RunPod Cloud generation
        let input = FluxInput {
            prompt: prompt.clone(),
            width: Some(self.width),
            height: Some(self.height),
            num_inference_steps: Some(self.num_inference_steps),
            seed: Some(seed),
            guidance_scale: self.guidance_scale,
        };

        println!(
            "  Mode: {}, {}x{}, {} steps, Seed: {}",
            style("RunPod Cloud").blue().bold(),
            self.width,
            self.height,
            self.num_inference_steps,
            seed
        );

        let status = client
            .run_and_poll(
                &self.endpoint_id,
                &input,
                Duration::from_secs(3),
                self.timeout,
                "Génération Flux",
            )
            .await?;

        let output_val = status
            .output
            .ok_or_else(|| anyhow::anyhow!("Aucun champ 'output' dans la réponse RunPod"))?;

        let image_url = extract_media_url(&output_val)
            .ok_or_else(|| anyhow::anyhow!("Impossible d'extraire l'URL de l'image de: {:?}", output_val))?;

        println!(
            "  {} Image produite : {}",
            style("✓").green().bold(),
            style(&image_url).cyan().underlined()
        );

        ctx.image_url = Some(image_url);
        ctx.set_meta("flux_seed", seed);
        ctx.set_meta("t2i_engine", "runpod_flux");

        Ok(())
    }
}
