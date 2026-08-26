pub mod context;
pub mod stages;

pub use context::PipelineContext;

use crate::client::RunpodClient;
use anyhow::{Context, Result};
use async_trait::async_trait;
use console::style;
use std::time::Instant;


/// Trait représentant une étape individuelle dans le pipeline de génération.
#[async_trait]
pub trait Stage: Send + Sync {
    /// Nom descriptif de l'étape.
    fn name(&self) -> &str;

    /// Exécute la logique de l'étape en accédant/mettant à jour le contexte.
    async fn execute(&self, ctx: &mut PipelineContext, client: &RunpodClient) -> Result<()>;
}

/// Structure d'orchestration d'une séquence d'étapes de pipeline.
pub struct Pipeline {
    name: String,
    stages: Vec<Box<dyn Stage>>,
}

impl Pipeline {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            stages: Vec::new(),
        }
    }

    /// Ajoute une étape à la fin du pipeline.
    pub fn add_stage<S: Stage + 'static>(mut self, stage: S) -> Self {
        self.stages.push(Box::new(stage));
        self
    }

    /// Exécute l'ensemble des étapes du pipeline séquentiellement.
    pub async fn run(&self, ctx: &mut PipelineContext, client: &RunpodClient) -> Result<()> {
        let total_stages = self.stages.len();
        println!(
            "\n{}",
            style(format!("🚀 Lancement du Pipeline : {}", self.name))
                .cyan()
                .bold()
        );
        println!(
            "{}",
            style(format!("Total d'étapes à exécuter : {}\n", total_stages)).dim()
        );

        let overall_start = Instant::now();

        for (idx, stage) in self.stages.iter().enumerate() {
            let stage_num = idx + 1;
            println!(
                "{}",
                style(format!("▶ Étape [{}/{}] : {}", stage_num, total_stages, stage.name()))
                    .yellow()
                    .bold()
            );

            let stage_start = Instant::now();
            stage
                .execute(ctx, client)
                .await
                .with_context(|| format!("Échec de l'étape [{}] : {}", stage_num, stage.name()))?;

            let stage_duration = stage_start.elapsed();
            println!(
                "  {} Étape '{}' validée en {:.2}s\n",
                style("✓").green().bold(),
                stage.name(),
                stage_duration.as_secs_f64()
            );
        }

        let total_duration = overall_start.elapsed();
        println!(
            "{}",
            style(format!(
                "✨ Pipeline '{}' terminé avec succès en {:.2}s !\n",
                self.name,
                total_duration.as_secs_f64()
            ))
            .green()
            .bold()
        );

        Ok(())
    }
}
