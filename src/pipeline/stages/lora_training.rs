use crate::client::RunpodClient;
use crate::pipeline::{PipelineContext, Stage};
use anyhow::{Context, Result};
use async_trait::async_trait;
use console::style;
use std::path::PathBuf;
use tokio::io::AsyncBufReadExt;

/// Étape d'Entraînement LoRA (Fine-Tuning de Modèle Personnalisé)
pub struct LoraTrainingStage {
    pub dataset_dir: PathBuf,
    pub instance_prompt: String,
    pub output_name: String,
    pub base_model: String,
    pub train_steps: u32,
    pub learning_rate: f32,
    pub lora_rank: u32,
    pub resolution: u32,
    pub gradient_accumulation_steps: u32,
    pub mixed_precision: String,
    pub device: String,
}

impl Default for LoraTrainingStage {
    fn default() -> Self {
        Self {
            dataset_dir: PathBuf::from("datasets/default"),
            instance_prompt: "a photo of sks person".to_string(),
            output_name: "custom_lora.safetensors".to_string(),
            base_model: "runwayml/stable-diffusion-v1-5".to_string(),
            train_steps: 500,
            learning_rate: 1e-4,
            lora_rank: 8,
            resolution: 512,
            gradient_accumulation_steps: 4,
            mixed_precision: "no".to_string(),
            device: "cuda".to_string(),
        }
    }
}

impl LoraTrainingStage {
    pub fn new(dataset_dir: impl Into<PathBuf>, output_name: impl Into<String>) -> Self {
        Self {
            dataset_dir: dataset_dir.into(),
            output_name: output_name.into(),
            ..Default::default()
        }
    }

    pub fn with_instance_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.instance_prompt = prompt.into();
        self
    }

    pub fn with_base_model(mut self, base_model: impl Into<String>) -> Self {
        self.base_model = base_model.into();
        self
    }

    pub fn with_train_steps(mut self, steps: u32) -> Self {
        self.train_steps = steps;
        self
    }

    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn with_lora_rank(mut self, rank: u32) -> Self {
        self.lora_rank = rank;
        self
    }

    pub fn with_resolution(mut self, res: u32) -> Self {
        self.resolution = res;
        self
    }
}

#[async_trait]
impl Stage for LoraTrainingStage {
    fn name(&self) -> &str {
        "Entraînement LoRA (Fine-Tuning)"
    }

    async fn execute(&self, ctx: &mut PipelineContext, _client: &RunpodClient) -> Result<()> {
        if !self.dataset_dir.exists() {
            anyhow::bail!("Le dossier du dataset '{}' n'existe pas.", self.dataset_dir.display());
        }

        let loras_dir = std::path::Path::new("loras");
        let _ = std::fs::create_dir_all(loras_dir);

        let final_filename = if self.output_name.ends_with(".safetensors") {
            self.output_name.clone()
        } else {
            format!("{}.safetensors", self.output_name)
        };
        let final_output = loras_dir.join(&final_filename);

        println!(
            "  LoRA Config : Nom: {}, Base: {}, Steps: {}, LR: {}, Rank: {}, Res: {}px",
            style(&final_filename).cyan().bold(),
            style(&self.base_model).yellow(),
            style(self.train_steps).green().bold(),
            self.learning_rate,
            self.lora_rank,
            self.resolution
        );
        println!("  Lancement du moteur d'entraînement LoRA...");

        let mut cmd = tokio::process::Command::new(crate::utils::get_python_binary());
        crate::utils::configure_python_command(&mut cmd);
        cmd.args([
            "src/lora_train_engine.py",
            "--dataset-dir", self.dataset_dir.to_str().unwrap(),
            "--instance-prompt", &self.instance_prompt,
            "--output-name", &final_filename,
            "--output-dir", "loras",
            "--base-model", &self.base_model,
            "--resolution", &self.resolution.to_string(),
            "--train-steps", &self.train_steps.to_string(),
            "--learning-rate", &self.learning_rate.to_string(),
            "--lora-rank", &self.lora_rank.to_string(),
            "--gradient-accumulation-steps", &self.gradient_accumulation_steps.to_string(),
            "--mixed-precision", &self.mixed_precision,
            "--device", &self.device,
        ]);

        cmd.stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        let mut child = cmd
            .spawn()
            .context("Impossible de démarrer le script lora_train_engine.py")?;

        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();

        let mut stdout_reader = tokio::io::BufReader::new(stdout).lines();
        let mut stderr_reader = tokio::io::BufReader::new(stderr).lines();

        let mut stdout_done = false;
        let mut stderr_done = false;

        while !stdout_done || !stderr_done {
            tokio::select! {
                line = stdout_reader.next_line(), if !stdout_done => {
                    match line {
                        Ok(Some(l)) => {
                            if l.starts_with("[PROGRESS]") {
                                println!("  {}", style(&l).cyan());
                            } else {
                                println!("  {}", l);
                            }
                        }
                        _ => stdout_done = true,
                    }
                }
                line = stderr_reader.next_line(), if !stderr_done => {
                    match line {
                        Ok(Some(l)) => eprintln!("  {}", style(l).dim()),
                        _ => stderr_done = true,
                    }
                }
            }
        }

        let status = child.wait().await?;
        if !status.success() {
            anyhow::bail!("Le script lora_train_engine.py s'est terminé avec une erreur (code {:?})", status.code());
        }

        if !final_output.exists() {
            anyhow::bail!("Le fichier LoRA généré n'a pas été trouvé à l'emplacement : {}", final_output.display());
        }

        ctx.set_meta("lora_path", final_output.to_string_lossy().to_string());
        println!("  LoRA entraîné avec succès : {}", style(final_output.display()).green().bold());
        Ok(())
    }
}
