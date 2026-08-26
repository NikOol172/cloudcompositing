use crate::client::RunpodClient;
use crate::models::{ChatCompletionRequest, ChatMessage};
use crate::pipeline::{PipelineContext, Stage};
use anyhow::Result;
use async_trait::async_trait;
use console::style;
use tracing::warn;

pub const DEFAULT_QWEN_ENDPOINT: &str = "qwen3-32b-awq";
pub const DEFAULT_QWEN_MODEL: &str = "Qwen/Qwen3-32B-AWQ";

/// Étape d'enrichissement de prompt par un LLM (Qwen ou compatible OpenAI).
pub struct PromptEnhanceStage {
    pub endpoint_id: String,
    pub model: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub allow_fallback: bool,
}

impl Default for PromptEnhanceStage {
    fn default() -> Self {
        Self {
            endpoint_id: DEFAULT_QWEN_ENDPOINT.to_string(),
            model: DEFAULT_QWEN_MODEL.to_string(),
            max_tokens: 200,
            temperature: 0.7,
            allow_fallback: true,
        }
    }
}

impl PromptEnhanceStage {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_endpoint(mut self, endpoint: impl Into<String>, model: impl Into<String>) -> Self {
        self.endpoint_id = endpoint.into();
        self.model = model.into();
        self
    }
}

#[async_trait]
impl Stage for PromptEnhanceStage {
    fn name(&self) -> &str {
        "Enrichissement du prompt via LLM (Qwen)"
    }

    async fn execute(&self, ctx: &mut PipelineContext, client: &RunpodClient) -> Result<()> {
        let base_prompt = &ctx.initial_prompt;
        println!(
            "  Prompt initial : {}",
            style(base_prompt).italic().cyan()
        );

        let system_prompt = "You are an expert at writing prompts for AI image and video generation. \
            Transform the user's simple idea into a detailed, vivid visual description. \
            Include details about lighting, style, camera angle, composition, and atmosphere. \
            Keep the description concise and under 100 words. \
            Output ONLY the enhanced prompt, nothing else. Do not include thinking, commentary, or markdown tags.";

        let request = ChatCompletionRequest {
            model: self.model.clone(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: system_prompt.to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: base_prompt.clone(),
                },
            ],
            max_tokens: Some(self.max_tokens),
            temperature: Some(self.temperature),
        };

        match client.chat_completion(&self.endpoint_id, &request).await {
            Ok(resp) => {
                if let Some(choice) = resp.choices.first() {
                    let mut content = choice.message.content.trim().to_string();

                    // Nettoyage des balises de raisonnement <think>...</think>
                    if let Some(start) = content.find("<think>") {
                        if let Some(end) = content.find("</think>") {
                            content.replace_range(start..end + 8, "");
                        } else {
                            content.truncate(start);
                        }
                    }
                    let cleaned = content.trim().to_string();

                    println!(
                        "  Prompt enrichi (LLM) : {}",
                        style(&cleaned).bold().green()
                    );
                    ctx.enhanced_prompt = Some(cleaned);
                } else {
                    let fallback_prompt = smart_enhance_prompt(base_prompt);
                    warn!("Réponse LLM vide, application de l'optimisation visuelle par défaut.");
                    ctx.enhanced_prompt = Some(fallback_prompt);
                }
            }
            Err(e) => {
                if self.allow_fallback {
                    let enhanced_fallback = smart_enhance_prompt(base_prompt);
                    println!(
                        "  {} LLM non disponible ({}), application de l'optimisation visuelle intelligente.",
                        style("ℹ️").cyan(),
                        e
                    );
                    println!(
                        "  Prompt optimisé : {}",
                        style(&enhanced_fallback).bold().green()
                    );
                    ctx.enhanced_prompt = Some(enhanced_fallback);
                } else {
                    return Err(e);
                }
            }
        }

        Ok(())
    }
}

/// Optimiseur intelligent de prompt visuel (fallback rapide et haute fidélité pour la génération d'images et vidéos).
pub fn smart_enhance_prompt(input: &str) -> String {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return input.to_string();
    }

    let lower = trimmed.to_lowercase();
    let mut enhancements = Vec::new();

    if !lower.contains("cinematic") && !lower.contains("camera") && !lower.contains("shot") && !lower.contains("pan") {
        enhancements.push("smooth cinematic camera tracking");
    }

    if !lower.contains("light") && !lower.contains("sun") && !lower.contains("glow") && !lower.contains("illumination") {
        enhancements.push("soft natural ambient lighting with subtle reflections");
    }

    if !lower.contains("photorealistic") && !lower.contains("realistic") && !lower.contains("hyperrealistic") {
        enhancements.push("photorealistic");
    }

    if !lower.contains("8k") && !lower.contains("4k") && !lower.contains("uhd") {
        enhancements.push("8k uhd");
    }

    if !lower.contains("quality") && !lower.contains("commercial") && !lower.contains("lifestyle") {
        enhancements.push("commercial aesthetic, master craftsmanship");
    }

    if enhancements.is_empty() {
        trimmed.to_string()
    } else {
        format!("{}, {}", trimmed.trim_end_matches('.'), enhancements.join(", "))
    }
}
