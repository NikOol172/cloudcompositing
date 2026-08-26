use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;

/// Contexte partagé transmis tout au long de l'exécution d'un pipeline.
#[derive(Debug, Clone, Default)]
pub struct PipelineContext {
    /// Prompt initial fourni par l'utilisateur.
    pub initial_prompt: String,
    /// Prompt enrichi par le LLM (si l'étape d'enrichissement a été activée).
    pub enhanced_prompt: Option<String>,
    /// URL de l'image (obtenue par génération ou spécifiée en entrée).
    pub image_url: Option<String>,
    /// Chemin vers une image locale fournie en entrée.
    pub local_image_path: Option<PathBuf>,
    /// Chemin du fichier image téléchargé sur le disque.
    pub downloaded_image_path: Option<PathBuf>,
    /// URL de la vidéo générée.
    pub video_url: Option<String>,
    /// Chemin du fichier vidéo téléchargé sur le disque.
    pub downloaded_video_path: Option<PathBuf>,
    /// Données et métadonnées arbitraires stockées par les différentes étapes.
    pub metadata: HashMap<String, Value>,
}

#[allow(dead_code)]
impl PipelineContext {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            initial_prompt: prompt.into(),
            ..Default::default()
        }
    }

    /// Renvoie le prompt effectif : le prompt enrichi s'il existe, sinon le prompt initial.
    pub fn effective_prompt(&self) -> &str {
        self.enhanced_prompt
            .as_deref()
            .unwrap_or(&self.initial_prompt)
    }

    /// Définit une métadonnée arbitraire.
    pub fn set_meta<T: Into<Value>>(&mut self, key: impl Into<String>, value: T) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Récupère une métadonnée.
    pub fn get_meta(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }
}
