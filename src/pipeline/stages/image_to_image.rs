use crate::client::RunpodClient;
use crate::models::{extract_media_url, Img2ImgInput};
use crate::pipeline::{PipelineContext, Stage};
use crate::utils::resolve_image_input;
use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use console::style;
use rand::Rng;
use std::time::Duration;
use tokio::io::AsyncBufReadExt;

pub const DEFAULT_IMG2IMG_ENDPOINT: &str = "black-forest-labs-flux-1-schnell";
pub const DEFAULT_LOCAL_MODEL: &str = "stabilityai/sdxl-turbo";

/// Étape de génération Image-to-Image & Inpainting (via RunPod Flux / SDXL Cloud ou Diffusers Local).
pub struct ImageToImageStage {
    pub endpoint_id: String,
    pub image_source_override: Option<String>,
    pub mask_source_override: Option<String>,
    pub prompt_override: Option<String>,
    pub strength: f32,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub num_inference_steps: u32,
    pub seed: Option<u64>,
    pub guidance_scale: Option<f32>,
    pub timeout: Duration,
    pub local: bool,
    pub local_model: Option<String>,
    pub lora: Option<String>,
    pub lora_scale: Option<f32>,
    pub controlnet_image: Option<String>,
    pub controlnet_type: Option<String>,
    pub controlnet_scale: Option<f32>,
}

impl Default for ImageToImageStage {
    fn default() -> Self {
        Self {
            endpoint_id: DEFAULT_IMG2IMG_ENDPOINT.to_string(),
            image_source_override: None,
            mask_source_override: None,
            prompt_override: None,
            strength: 0.70,
            width: Some(768),
            height: Some(1344),
            num_inference_steps: 4,
            seed: None,
            guidance_scale: None,
            timeout: Duration::from_secs(300),
            local: false,
            local_model: None,
            lora: None,
            lora_scale: None,
            controlnet_image: None,
            controlnet_type: None,
            controlnet_scale: None,
        }
    }
}

impl ImageToImageStage {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint_id = endpoint.into();
        self
    }

    pub fn with_image(mut self, image: impl Into<String>) -> Self {
        self.image_source_override = Some(image.into());
        self
    }

    pub fn with_mask(mut self, mask: impl Into<String>) -> Self {
        self.mask_source_override = Some(mask.into());
        self
    }

    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt_override = Some(prompt.into());
        self
    }

    pub fn with_strength(mut self, strength: f32) -> Self {
        self.strength = strength;
        self
    }

    pub fn with_dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = Some(width);
        self.height = Some(height);
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

    pub fn with_lora(mut self, lora: Option<String>, scale: Option<f32>) -> Self {
        self.lora = lora;
        self.lora_scale = scale;
        self
    }

    pub fn with_controlnet(
        mut self,
        image: Option<String>,
        c_type: Option<String>,
        scale: Option<f32>,
    ) -> Self {
        self.controlnet_image = image;
        self.controlnet_type = c_type;
        self.controlnet_scale = scale;
        self
    }
}

#[async_trait]
impl Stage for ImageToImageStage {
    fn name(&self) -> &str {
        if self.mask_source_override.is_some() {
            if self.local {
                "Inpainting Masqué Local (GPU / Diffusers)"
            } else {
                "Inpainting Masqué Cloud (Flux / SDXL)"
            }
        } else if self.local {
            "Génération Image-to-Image Locale (GPU / Diffusers)"
        } else {
            "Génération Image-to-Image Cloud (Flux / SDXL)"
        }
    }

    async fn execute(&self, ctx: &mut PipelineContext, client: &RunpodClient) -> Result<()> {
        let raw_image_input = if let Some(ref img) = self.image_source_override {
            img.clone()
        } else if let Some(ref path) = ctx.local_image_path {
            path.to_string_lossy().to_string()
        } else if let Some(ref img_path) = ctx.downloaded_image_path {
            img_path.to_string_lossy().to_string()
        } else if let Some(ref url) = ctx.image_url {
            url.clone()
        } else {
            bail!("Aucune image source fournie pour l'étape Image-to-Image.");
        };

        let prompt = if let Some(ref p) = self.prompt_override {
            p.clone()
        } else {
            ctx.effective_prompt().to_string()
        };

        let seed = self
            .seed
            .unwrap_or_else(|| rand::thread_rng().gen_range(1..999_999_999));

        if self.local {
            let model = self
                .local_model
                .as_deref()
                .unwrap_or(DEFAULT_LOCAL_MODEL);

            let temp_out = std::env::temp_dir().join(format!("local_i2i_{}.png", rand::random::<u32>()));
            let temp_out_str = temp_out.to_string_lossy().to_string();

            let resolved_image = resolve_image_input(&raw_image_input).await?;
            let source_file_path = if resolved_image.starts_with("data:image/") {
                // Decode base64 to temp file for python script
                let temp_in = std::env::temp_dir().join(format!("local_i2i_in_{}.png", rand::random::<u32>()));
                if let Some(comma_pos) = resolved_image.find(',') {
                    let base64_data = &resolved_image[comma_pos + 1..];
                    use base64::Engine;
                    let bytes = base64::engine::general_purpose::STANDARD.decode(base64_data)?;
                    tokio::fs::write(&temp_in, bytes).await?;
                }
                temp_in.to_string_lossy().to_string()
            } else {
                resolved_image
            };

            let mut mask_file_path: Option<String> = None;
            if let Some(ref raw_mask) = self.mask_source_override {
                let resolved_mask = resolve_image_input(raw_mask).await?;
                let temp_mask_path = if resolved_mask.starts_with("data:image/") {
                    let temp_m = std::env::temp_dir().join(format!("local_mask_{}.png", rand::random::<u32>()));
                    if let Some(comma_pos) = resolved_mask.find(',') {
                        let base64_data = &resolved_mask[comma_pos + 1..];
                        use base64::Engine;
                        let bytes = base64::engine::general_purpose::STANDARD.decode(base64_data)?;
                        tokio::fs::write(&temp_m, bytes).await?;
                    }
                    temp_m.to_string_lossy().to_string()
                } else {
                    resolved_mask
                };
                mask_file_path = Some(temp_mask_path);
            }

            println!(
                "  Mode: {}, Modèle: {}, Masque: {}, Strength: {}, Steps: {}, Seed: {}",
                style("Local GPU/CPU").magenta().bold(),
                style(model).cyan(),
                if mask_file_path.is_some() { style("Activé").green().bold() } else { style("Aucun").dim() },
                self.strength,
                self.num_inference_steps,
                seed
            );
            println!("  Lancement du moteur local diffusers img2img/inpaint...");

            let mut cmd = tokio::process::Command::new(crate::utils::get_python_binary());
            crate::utils::configure_python_command(&mut cmd);
            cmd.args([
                "src/img2img_engine.py",
                "--image",
                &source_file_path,
                "--prompt",
                &prompt,
                "--output",
                &temp_out_str,
                "--model",
                model,
                "--strength",
                &self.strength.to_string(),
                "--steps",
                &self.num_inference_steps.to_string(),
                "--seed",
                &seed.to_string(),
            ]);

            if let Some(ref mf) = mask_file_path {
                cmd.args(["--mask", mf]);
            }

            if let Some(w) = self.width {
                cmd.args(["--width", &w.to_string()]);
            }
            if let Some(h) = self.height {
                cmd.args(["--height", &h.to_string()]);
            }
            if let Some(cfg) = self.guidance_scale {
                cmd.args(["--guidance-scale", &cfg.to_string()]);
            }

            if let Some(ref lora) = self.lora {
                cmd.args(["--lora", lora]);
                if let Some(scale) = self.lora_scale {
                    cmd.args(["--lora-scale", &scale.to_string()]);
                }
            }

            if let Some(ref cn_type) = self.controlnet_type {
                cmd.args(["--controlnet-type", cn_type]);
                if let Some(ref cn_img) = self.controlnet_image {
                    cmd.args(["--controlnet-image", cn_img]);
                }
                if let Some(scale) = self.controlnet_scale {
                    cmd.args(["--controlnet-scale", &scale.to_string()]);
                }
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

            cmd.stdout(std::process::Stdio::piped());
            cmd.stderr(std::process::Stdio::piped());

            let mut child = cmd.spawn().context("Impossible de lancer 'src/img2img_engine.py'.")?;

            if let Some(stdout) = child.stdout.take() {
                let mut reader = tokio::io::BufReader::new(stdout).lines();
                while let Ok(Some(line)) = reader.next_line().await {
                    if line.starts_with("[PROGRESS]") {
                        println!("  {}", style(&line).yellow());
                    } else if line.starts_with("[INFO]") {
                        println!("  {}", style(&line).dim());
                    } else if line.starts_with("[SUCCESS]") {
                        println!("  {}", style(&line).green().bold());
                    } else {
                        println!("  {}", line);
                    }
                }
            }

            let status = child.wait().await?;
            if !status.success() {
                bail!("Échec de la génération locale Image-to-Image (code {})", status);
            }

            ctx.image_url = Some(temp_out_str);
            ctx.set_meta("i2i_seed", seed);
            ctx.set_meta("i2i_engine", "local_diffusers");
            return Ok(());
        }

        // Mode Cloud RunPod
        let resolved_image = resolve_image_input(&raw_image_input).await?;
        let resolved_mask = if let Some(ref m) = self.mask_source_override {
            Some(resolve_image_input(m).await?)
        } else {
            None
        };

        println!(
            "  Image Source: {}, Masque: {}, Strength: {}, Prompt: '{}'",
            style(&raw_image_input).cyan(),
            if resolved_mask.is_some() { style("Activé").green() } else { style("Non").dim() },
            self.strength,
            style(&prompt).italic()
        );

        let input_payload = Img2ImgInput {
            prompt,
            image: resolved_image,
            mask_image: resolved_mask,
            strength: Some(self.strength),
            width: self.width,
            height: self.height,
            num_inference_steps: Some(self.num_inference_steps),
            guidance_scale: self.guidance_scale,
            seed: Some(seed),
            negative_prompt: None,
        };

        let status = client
            .run_and_poll(
                &self.endpoint_id,
                &input_payload,
                Duration::from_secs(3),
                self.timeout,
                "Génération Cloud Image-to-Image",
            )
            .await?;

        let output_val = status.output.context("Aucun output retourné par le job Image-to-Image")?;
        let image_url = extract_media_url(&output_val).context(
            format!("Impossible d'extraire l'URL de l'image depuis la réponse: {:?}", output_val)
        )?;

        println!("  Image-to-Image URL générée: {}", style(&image_url).green().underlined());

        ctx.image_url = Some(image_url);

        Ok(())
    }
}
