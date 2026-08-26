use crate::client::RunpodClient;
use crate::pipeline::{PipelineContext, Stage};
use anyhow::{Context, Result};
use async_trait::async_trait;
use console::style;
use std::time::Duration;

pub const DEFAULT_FACESWAP_ENDPOINT: &str = "1peyap2qc3tx31";

/// Étape de remplacement de visage (Face Swap sur Image ou Vidéo).
pub struct FaceSwapStage {
    pub endpoint_id: String,
    pub source_face_input: String,
    pub target_media_input: String,
    pub restore_face: bool,
    pub face_index: u32,
    pub timeout: Duration,
}

impl Default for FaceSwapStage {
    fn default() -> Self {
        Self {
            endpoint_id: DEFAULT_FACESWAP_ENDPOINT.to_string(),
            source_face_input: String::new(),
            target_media_input: String::new(),
            restore_face: true,
            face_index: 0,
            timeout: Duration::from_secs(600),
        }
    }
}

impl FaceSwapStage {
    pub fn new(source: impl Into<String>, target: impl Into<String>) -> Self {
        Self {
            source_face_input: source.into(),
            target_media_input: target.into(),
            ..Default::default()
        }
    }

    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint_id = endpoint.into();
        self
    }

    pub fn with_restore_face(mut self, restore: bool) -> Self {
        self.restore_face = restore;
        self
    }

    pub fn with_face_index(mut self, index: u32) -> Self {
        self.face_index = index;
        self
    }
}

use tokio::io::AsyncBufReadExt;

#[async_trait]
impl Stage for FaceSwapStage {
    fn name(&self) -> &str {
        "Remplacement de Visage (Face Swap)"
    }

    async fn execute(&self, ctx: &mut PipelineContext, _client: &RunpodClient) -> Result<()> {
        let is_video = self.target_media_input.to_lowercase().ends_with(".mp4")
            || self.target_media_input.to_lowercase().ends_with(".webm")
            || self.target_media_input.to_lowercase().ends_with(".mov");

        let target_type = if is_video { "Vidéo" } else { "Image" };
        let out_ext = if is_video { "mp4" } else { "png" };
        let temp_out = std::env::temp_dir().join(format!("swapped_{}.{}", rand::random::<u32>(), out_ext));

        println!(
            "  Paramètres : Type cible: {}, Restauration visage: {}, Index: {}",
            style(target_type).cyan().bold(),
            if self.restore_face { style("Activée").green() } else { style("Désactivée").dim() },
            self.face_index
        );

        println!("  Lancement du moteur de remplacement de visage InsightFace...");

        let mut cmd = tokio::process::Command::new("python3");
        cmd.args([
            "src/faceswap_engine.py",
            "--source", &self.source_face_input,
            "--target", &self.target_media_input,
            "--output", temp_out.to_str().unwrap(),
            "--face-index", &self.face_index.to_string(),
        ]);

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
            .context("Impossible de démarrer le script faceswap_engine.py")?;

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
            anyhow::bail!("Échec de l'exécution du moteur de Face Swap.");
        }

        let result_file_str = temp_out.to_string_lossy().to_string();
        println!(
            "  {} Remplacement de visage terminé avec succès : {}",
            style("✓").green().bold(),
            style(&result_file_str).cyan().underlined()
        );

        if is_video {
            ctx.video_url = Some(result_file_str);
        } else {
            ctx.image_url = Some(result_file_str);
        }

        Ok(())
    }
}
