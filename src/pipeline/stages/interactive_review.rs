use crate::client::RunpodClient;
use crate::pipeline::{PipelineContext, Stage};
use crate::utils::open_file_preview;
use anyhow::{bail, Result};
use async_trait::async_trait;
use console::style;
use std::io::{self, Write};

/// Étape de validation interactive : permet à l'utilisateur de prévisualiser l'image et d'autoriser la génération vidéo.
pub struct InteractiveReviewStage {
    pub auto_open_preview: bool,
    pub prompt_message: String,
}

impl Default for InteractiveReviewStage {
    fn default() -> Self {
        Self {
            auto_open_preview: true,
            prompt_message: "Appuyez sur [ENTRÉE] pour lancer la génération vidéo (ou 'q' pour quitter) : ".to_string(),
        }
    }
}

impl InteractiveReviewStage {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_auto_open(mut self, auto_open: bool) -> Self {
        self.auto_open_preview = auto_open;
        self
    }
}

#[async_trait]
impl Stage for InteractiveReviewStage {
    fn name(&self) -> &str {
        "Validation Interactive Utilisateur"
    }

    async fn execute(&self, ctx: &mut PipelineContext, _client: &RunpodClient) -> Result<()> {
        println!("\n{}", style("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━").yellow());
        println!(
            " {}",
            style("🔍 Étape de validation de l'image intermédiaire").bold().yellow()
        );

        if let Some(ref path) = ctx.downloaded_image_path {
            println!(" Image disponible localement : {}", style(path.display()).cyan());
            if self.auto_open_preview {
                println!(" Tentative d'ouverture dans la visionneuse par défaut...");
                let _ = open_file_preview(path);
            }
        } else if let Some(ref url) = ctx.image_url {
            println!(" Image générée : {}", style(url).cyan());
        }

        println!(" Prompt actuel : {}", style(ctx.effective_prompt()).italic());
        println!("{}\n", style("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━").yellow());

        print!("{}", style(&self.prompt_message).bold());
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let trimmed = input.trim().to_lowercase();

        if trimmed == "q" || trimmed == "quit" || trimmed == "n" || trimmed == "no" {
            bail!("Génération interrompue par l'utilisateur.");
        }

        Ok(())
    }
}
