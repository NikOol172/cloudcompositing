use crate::models::{
    ChatCompletionRequest, ChatCompletionResponse, JobRunResponse, JobState, JobStatusResponse,
    RunpodAccountInfo, RunpodPayload,
};
use anyhow::{bail, Context, Result};
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tracing::{debug, info};

#[allow(dead_code)]
pub const DEFAULT_RUNPOD_BASE_URL: &str = "https://api.runpod.ai/v2";

/// Client HTTP asynchrone pour interagir avec les endpoints Serverless et Publics de RunPod.
pub type ProgressCallback = std::sync::Arc<dyn Fn(&str, &str, &str, f32) + Send + Sync>;

#[derive(Clone)]
pub struct RunpodClient {
    http: reqwest::Client,
    #[allow(dead_code)]
    api_key: String,
    base_url: String,
    progress_callback: Option<ProgressCallback>,
}

#[allow(dead_code)]
impl RunpodClient {
    /// Initialise un nouveau client RunPod avec la clé API fournie.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_base_url(api_key, DEFAULT_RUNPOD_BASE_URL)
    }

    /// Initialise un client en chargeant la clé `RUNPOD_API_KEY` depuis l'environnement ou un fichier `.env`.
    pub fn from_env() -> Result<Self> {
        let _ = dotenvy::dotenv();
        let api_key = std::env::var("RUNPOD_API_KEY")
            .context("La variable d'environnement 'RUNPOD_API_KEY' n'est pas définie.")?;
        Ok(Self::new(api_key))
    }

    /// Initialise un client avec une URL de base personnalisée.
    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        let api_key = api_key.into();
        let mut headers = HeaderMap::new();
        if let Ok(mut auth_val) = HeaderValue::from_str(&format!("Bearer {}", api_key)) {
            auth_val.set_sensitive(true);
            headers.insert(AUTHORIZATION, auth_val);
        }
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let http = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(Duration::from_secs(60))
            .build()
            .expect("Échec de création du client HTTP reqwest");

        Self {
            http,
            api_key,
            base_url: base_url.into().trim_end_matches('/').to_string(),
            progress_callback: None,
        }
    }

    pub fn with_progress_callback<F: Fn(&str, &str, &str, f32) + Send + Sync + 'static>(
        mut self,
        callback: F,
    ) -> Self {
        self.progress_callback = Some(std::sync::Arc::new(callback));
        self
    }

    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Soumet un job asynchrone sur l'endpoint RunPod spécifié (`POST /v2/{endpoint_id}/run`).
    pub async fn run_job<I: Serialize>(
        &self,
        endpoint_id: &str,
        input: &I,
    ) -> Result<JobRunResponse> {
        let url = format!("{}/{}/run", self.base_url, endpoint_id);
        debug!("Soumission du job vers {}", url);

        let payload = RunpodPayload { input };
        let response = self
            .http
            .post(&url)
            .json(&payload)
            .send()
            .await
            .with_context(|| format!("Erreur réseau lors de l'appel à {}", url))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "<illisible>".to_string());
            bail!(
                "Échec de soumission RunPod (HTTP {}): {}",
                status,
                text
            );
        }

        let run_resp: JobRunResponse = response
            .json()
            .await
            .with_context(|| "Impossible de décoder la réponse JSON de soumission de job")?;

        Ok(run_resp)
    }

    /// Récupère le statut actuel d'un job (`GET /v2/{endpoint_id}/status/{job_id}`).
    pub async fn get_job_status(
        &self,
        endpoint_id: &str,
        job_id: &str,
    ) -> Result<JobStatusResponse> {
        let url = format!("{}/{}/status/{}", self.base_url, endpoint_id, job_id);
        let response = self
            .http
            .get(&url)
            .send()
            .await
            .with_context(|| format!("Erreur réseau lors de la vérification du statut sur {}", url))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "<illisible>".to_string());
            bail!("Erreur statut RunPod (HTTP {}): {}", status, text);
        }

        let status_resp: JobStatusResponse = response
            .json()
            .await
            .with_context(|| "Impossible de désérialiser la réponse de statut de job")?;

        Ok(status_resp)
    }

    /// Interroge périodiquement (polling) le statut du job jusqu'à complétion ou échec, avec affichage animé.
    pub async fn poll_job(
        &self,
        endpoint_id: &str,
        job_id: &str,
        poll_interval: Duration,
        timeout: Duration,
        task_label: &str,
    ) -> Result<JobStatusResponse> {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
                .template("{spinner:.cyan} {msg}")
                .unwrap(),
        );
        pb.enable_steady_tick(Duration::from_millis(80));

        let start_time = Instant::now();
        info!("Attente de complétion du job {} sur '{}'", job_id, endpoint_id);

        loop {
            let elapsed = start_time.elapsed();
            if elapsed > timeout {
                pb.finish_and_clear();
                bail!(
                    "Le job '{}' a dépassé le délai maximum d'attente (timeout de {:.0}s).",
                    job_id,
                    timeout.as_secs_f64()
                );
            }

            match self.get_job_status(endpoint_id, job_id).await {
                Ok(status) => {
                    let raw_status = match &status.status {
                        JobState::InQueue => "IN_QUEUE",
                        JobState::InProgress => "IN_PROGRESS",
                        JobState::Completed => "COMPLETED",
                        JobState::Failed => "FAILED",
                        JobState::Cancelled => "CANCELLED",
                        JobState::TimedOut => "TIMED_OUT",
                        JobState::Unknown => "UNKNOWN",
                    };

                    if let Some(ref cb) = self.progress_callback {
                        cb(job_id, endpoint_id, raw_status, elapsed.as_secs_f32());
                    }

                    let status_str = match &status.status {
                        JobState::InQueue => style("EN ATTENTE (IN_QUEUE)").yellow().to_string(),
                        JobState::InProgress => style("EN COURS (IN_PROGRESS)").blue().to_string(),
                        JobState::Completed => style("TERMINÉ (COMPLETED)").green().bold().to_string(),
                        JobState::Failed => style("ÉCHEC (FAILED)").red().bold().to_string(),
                        JobState::Cancelled => style("ANNULÉ (CANCELLED)").magenta().to_string(),
                        JobState::TimedOut => style("EXPIRÉ (TIMED_OUT)").red().to_string(),
                        JobState::Unknown => style("STATUT INCONNU").dim().to_string(),
                    };

                    pb.set_message(format!(
                        "{} | Job: {} | {} ({:.1}s)",
                        task_label,
                        style(job_id).cyan(),
                        status_str,
                        elapsed.as_secs_f32()
                    ));

                    if status.status == JobState::Completed {
                        pb.finish_with_message(format!(
                            "{} | Job: {} | {} en {:.1}s",
                            task_label,
                            style(job_id).cyan(),
                            style("TERMINÉ AVEC SUCCÈS").green().bold(),
                            elapsed.as_secs_f32()
                        ));
                        return Ok(status);
                    } else if status.status == JobState::Failed || status.status == JobState::Cancelled {
                        pb.finish_with_message(format!(
                            "{} | Job: {} | {}",
                            task_label,
                            style(job_id).cyan(),
                            style("ÉCHEC").red().bold()
                        ));
                        let err_detail = status
                            .error
                            .as_ref()
                            .map(|e| e.to_string())
                            .unwrap_or_else(|| "Erreur non spécifiée par RunPod".to_string());
                        bail!("Le job RunPod a échoué: {}", err_detail);
                    }
                }
                Err(e) => {
                    debug!("Erreur temporaire lors du polling: {}", e);
                    pb.set_message(format!(
                        "{} | Job: {} | Erreur réseau temporaire, nouvel essai... ({:.1}s)",
                        task_label,
                        job_id,
                        elapsed.as_secs_f32()
                    ));
                }
            }

            tokio::time::sleep(poll_interval).await;
        }
    }

    /// Soumet un job et attend sa complétion.
    pub async fn run_and_poll<I: Serialize>(
        &self,
        endpoint_id: &str,
        input: &I,
        poll_interval: Duration,
        timeout: Duration,
        task_label: &str,
    ) -> Result<JobStatusResponse> {
        let submission = self.run_job(endpoint_id, input).await?;
        println!(
            "  {} Job soumis : {}",
            style("✓").green().bold(),
            style(&submission.id).cyan().bold()
        );

        if let Some(ref cb) = self.progress_callback {
            cb(&submission.id, endpoint_id, "IN_QUEUE", 0.0);
        }

        self.poll_job(endpoint_id, &submission.id, poll_interval, timeout, task_label)
            .await
    }

    /// Exécute une requête de chat LLM (ex: Qwen3-32B) via l'interface OpenAI compatible (`POST /v2/{endpoint_id}/openai/v1/chat/completions`).
    pub async fn chat_completion(
        &self,
        endpoint_id: &str,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        let url = format!("{}/{}/openai/v1/chat/completions", self.base_url, endpoint_id);
        debug!("Appel Chat Completion vers {}", url);

        let response = self
            .http
            .post(&url)
            .json(request)
            .timeout(Duration::from_secs(45))
            .send()
            .await
            .with_context(|| format!("Erreur lors de l'appel LLM sur {}", url))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "<illisible>".to_string());
            bail!("Échec LLM RunPod (HTTP {}): {}", status, text);
        }

        let chat_resp: ChatCompletionResponse = response
            .json()
            .await
            .with_context(|| "Impossible de désérialiser la réponse LLM")?;

        Ok(chat_resp)
    }

    /// Récupère les informations de compte et le solde (clientBalance) via l'API GraphQL de RunPod.
    pub async fn get_account_info(&self) -> Result<RunpodAccountInfo> {
        let url = "https://api.runpod.io/graphql";
        let query = serde_json::json!({
            "query": "query { myself { id email clientBalance } }"
        });

        let response = self
            .http
            .post(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&query)
            .timeout(Duration::from_secs(15))
            .send()
            .await
            .with_context(|| "Erreur réseau lors de la requête GraphQL RunPod")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            bail!("Erreur GraphQL RunPod (HTTP {}): {}", status, text);
        }

        #[derive(Deserialize)]
        struct Myself {
            id: Option<String>,
            email: Option<String>,
            #[serde(rename = "clientBalance")]
            client_balance: Option<f64>,
        }

        #[derive(Deserialize)]
        struct Data {
            myself: Option<Myself>,
        }

        #[derive(Deserialize)]
        struct GraphQLResponse {
            data: Option<Data>,
            errors: Option<serde_json::Value>,
        }

        let result: GraphQLResponse = response
            .json()
            .await
            .with_context(|| "Impossible de décoder la réponse GraphQL")?;

        if let Some(data) = result.data {
            if let Some(m) = data.myself {
                return Ok(RunpodAccountInfo {
                    id: m.id,
                    email: m.email,
                    client_balance: m.client_balance,
                });
            }
        }

        if let Some(errs) = result.errors {
            bail!("Erreur retournée par RunPod: {}", errs);
        }

        bail!("Impossible de récupérer les informations de compte RunPod (clé invalide ou erreur GraphQL)");
    }
}
