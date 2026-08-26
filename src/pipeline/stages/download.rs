use crate::client::RunpodClient;
use crate::pipeline::{PipelineContext, Stage};
use crate::utils::download_file;
use anyhow::Result;
use async_trait::async_trait;
use console::style;
use std::path::PathBuf;

/// Étape de téléchargement des fichiers médias générés (images ou vidéos).
pub struct DownloadStage {
    pub target_image_path: Option<PathBuf>,
    pub target_video_path: Option<PathBuf>,
    pub show_progress: bool,
}

impl Default for DownloadStage {
    fn default() -> Self {
        Self {
            target_image_path: None,
            target_video_path: None,
            show_progress: true,
        }
    }
}

impl DownloadStage {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn download_image(mut self, path: impl Into<PathBuf>) -> Self {
        self.target_image_path = Some(path.into());
        self
    }

    pub fn download_video(mut self, path: impl Into<PathBuf>) -> Self {
        self.target_video_path = Some(path.into());
        self
    }
}

#[async_trait]
impl Stage for DownloadStage {
    fn name(&self) -> &str {
        "Téléchargement des Médias Produits"
    }

    async fn execute(&self, ctx: &mut PipelineContext, _client: &RunpodClient) -> Result<()> {
        let mut downloaded_anything = false;

        // Téléchargement de l'image si une URL et un chemin cible existent
        if let (Some(ref url), Some(ref path)) = (&ctx.image_url, &self.target_image_path) {
            let final_path = crate::utils::get_unique_incremental_path(path);
            println!(
                "  Téléchargement de l'image vers {}",
                style(final_path.display()).cyan()
            );
            download_file(url, &final_path, self.show_progress).await?;
            println!(
                "  {} Image sauvegardée avec succès : {}",
                style("✓").green().bold(),
                style(final_path.display()).bold()
            );
            ctx.downloaded_image_path = Some(final_path);
            downloaded_anything = true;
        }

        // Téléchargement de la vidéo si une URL et un chemin cible existent
        if let (Some(ref url), Some(ref path)) = (&ctx.video_url, &self.target_video_path) {
            let final_path = crate::utils::get_unique_incremental_path(path);
            println!(
                "  Téléchargement de la vidéo vers {}",
                style(final_path.display()).cyan()
            );
            download_file(url, &final_path, self.show_progress).await?;
            println!(
                "  {} Vidéo sauvegardée avec succès : {}",
                style("✓").green().bold(),
                style(final_path.display()).bold()
            );
            ctx.downloaded_video_path = Some(final_path);
            downloaded_anything = true;
        }

        if !downloaded_anything {
            println!("  ℹ Aucun fichier média à télécharger ou aucun chemin cible spécifié.");
        }

        Ok(())
    }
}
