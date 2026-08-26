use crate::client::RunpodClient;
use crate::pipeline::stages::{
    DownloadStage, FaceSwapStage, ImageToImageStage, ImageToVideoStage, PromptEnhanceStage, TextToImageStage,
    VideoToVideoStage,
};
use crate::pipeline::{Pipeline, PipelineContext};
use axum::{
    extract::{Multipart, Path as AxumPath, State},
    http::StatusCode,
    response::{Html, IntoResponse, Json, Response},
    routing::{delete, get, post},
    Router,
};
use console::style;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, SystemTime};
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::ServeDir;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaItem {
    pub name: String,
    pub size_bytes: u64,
    pub is_video: bool,
    pub modified: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobLog {
    pub level: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerJob {
    pub id: String,
    pub pipeline_type: String,
    pub status: String,
    pub prompt: Option<String>,
    pub source: Option<String>,
    pub result_file: Option<String>,
    pub elapsed_seconds: u64,
    #[serde(default)]
    pub created_at: u64,
    pub logs: Vec<JobLog>,
    #[serde(default)]
    pub runpod_job_id: Option<String>,
    #[serde(default)]
    pub runpod_status: Option<String>,
    #[serde(default)]
    pub runpod_endpoint: Option<String>,
    #[serde(default)]
    pub stage_info: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptEntry {
    pub id: String,
    pub text: String,
    pub pipeline_type: String,
    pub created_at: u64,
    #[serde(default)]
    pub is_favorite: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerSettings {
    pub api_key: String,
    pub base_url: String,
    pub wan_endpoint: String,
    #[serde(default = "default_ltx_endpoint")]
    pub ltx_endpoint: String,
    #[serde(default = "default_minimax_endpoint")]
    pub minimax_endpoint: String,
    pub flux_endpoint: String,
    pub faceswap_endpoint: String,
    pub llm_endpoint: String,
    #[serde(default = "default_local_model")]
    pub local_model: String,
}

fn default_local_model() -> String {
    "stabilityai/sdxl-turbo".to_string()
}

fn default_ltx_endpoint() -> String {
    std::env::var("RUNPOD_LTX_ENDPOINT").unwrap_or_else(|_| "ltx-video-2-5".to_string())
}

fn default_minimax_endpoint() -> String {
    std::env::var("RUNPOD_MINIMAX_ENDPOINT").unwrap_or_else(|_| "minimax-h3".to_string())
}

impl ServerSettings {
    pub fn resolve_video_endpoint(&self, model: &crate::models::VideoModel) -> String {
        match model {
            crate::models::VideoModel::Wan2_5 => self.wan_endpoint.clone(),
            crate::models::VideoModel::Ltx2_5 => self.ltx_endpoint.clone(),
            crate::models::VideoModel::MiniMaxH3 => self.minimax_endpoint.clone(),
        }
    }
}

impl Default for ServerSettings {
    fn default() -> Self {
        Self {
            api_key: std::env::var("RUNPOD_API_KEY").unwrap_or_default(),
            base_url: "https://api.runpod.ai/v2".to_string(),
            wan_endpoint: std::env::var("RUNPOD_WAN_ENDPOINT").unwrap_or_else(|_| "wan-2-5".to_string()),
            ltx_endpoint: default_ltx_endpoint(),
            minimax_endpoint: default_minimax_endpoint(),
            flux_endpoint: "black-forest-labs-flux-1-schnell".to_string(),
            faceswap_endpoint: "1peyap2qc3tx31".to_string(),
            llm_endpoint: "qwen3-32b-awq".to_string(),
            local_model: "stabilityai/sdxl-turbo".to_string(),
        }
    }
}

#[derive(Clone)]
pub struct AppState {
    pub jobs: Arc<Mutex<Vec<ServerJob>>>,
    pub prompts: Arc<Mutex<Vec<PromptEntry>>>,
    pub settings: Arc<Mutex<ServerSettings>>,
    pub workspace_dir: PathBuf,
}

pub async fn start_server(port: u16, open_browser: bool) -> anyhow::Result<()> {
    let workspace_dir = std::env::current_dir()?;
    let settings = ServerSettings::default();

    // Charger l'historique des prompts existant s'il existe
    let prompts_file = workspace_dir.join("prompts_history.json");
    let initial_prompts: Vec<PromptEntry> = if prompts_file.exists() {
        tokio::fs::read_to_string(&prompts_file)
            .await
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    let state = AppState {
        jobs: Arc::new(Mutex::new(Vec::new())),
        prompts: Arc::new(Mutex::new(initial_prompts)),
        settings: Arc::new(Mutex::new(settings)),
        workspace_dir,
    };

    let app = Router::new()
        .route("/api/media", get(list_media))
        .route("/api/media/:filename", get(serve_media).delete(delete_media))
        .route("/api/upload", post(upload_media))
        .route("/api/jobs", get(list_jobs).delete(clear_jobs))
        .route("/api/prompts", get(list_prompts).post(save_prompt))
        .route("/api/prompts/:id", delete(delete_prompt))
        .route("/api/prompts/:id/favorite", post(toggle_favorite_prompt))
        .route("/api/settings", get(get_settings).post(save_settings))
        .route("/api/balance", get(get_balance))
        .route("/api/crop", post(crop_media))
        .route("/api/generate/txt2vid", post(generate_txt2vid))
        .route("/api/generate/img2vid", post(generate_img2vid))
        .route("/api/generate/faceswap", post(generate_faceswap))
        .route("/api/generate/vid2vid", post(generate_vid2vid))
        .route("/api/generate/txt2img", post(generate_txt2img))
        .route("/api/generate/img2img", post(generate_img2img))
        .route("/api/generate/enhance", post(generate_enhance))
        .fallback_service(ServeDir::new("web").fallback(get(serve_embedded_fallback)))
        .layer(axum::extract::DefaultBodyLimit::max(500 * 1024 * 1024))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port);
    let url = format!("http://localhost:{}", port);

    println!("\n{}", style("🚀 RunPod Studio & Media Manager démarré !").bold().magenta());
    println!("  • Interface Web : {}", style(&url).cyan().underlined().bold());
    println!("  • Écoute locale : {}", style(&addr).dim());
    println!("  • Appuyez sur {} pour quitter.\n", style("Ctrl+C").yellow().bold());

    if open_browser {
        let _ = open::that_detached(&url);
    }

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn serve_embedded_fallback() -> impl IntoResponse {
    Html("<!DOCTYPE html><html><body><h1>RunPod Studio</h1><p>Dossier web introuvable.</p></body></html>")
}

// ----------------------------------------------------
// Media Handlers
// ----------------------------------------------------
async fn list_media(State(state): State<AppState>) -> Json<Vec<MediaItem>> {
    let mut items = Vec::new();
    if let Ok(mut entries) = tokio::fs::read_dir(&state.workspace_dir).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    let ext_lower = ext.to_lowercase();
                    let is_img = matches!(ext_lower.as_str(), "png" | "jpg" | "jpeg" | "webp");
                    let is_vid = matches!(ext_lower.as_str(), "mp4" | "webm" | "mov" | "mkv");

                    if is_img || is_vid {
                        if let Ok(meta) = entry.metadata().await {
                            let modified = meta
                                .modified()
                                .unwrap_or(SystemTime::UNIX_EPOCH)
                                .duration_since(SystemTime::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs();

                            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                                items.push(MediaItem {
                                    name: name.to_string(),
                                    size_bytes: meta.len(),
                                    is_video: is_vid,
                                    modified,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    // Sort newest first
    items.sort_by(|a, b| b.modified.cmp(&a.modified));
    Json(items)
}

async fn serve_media(
    State(state): State<AppState>,
    AxumPath(filename): AxumPath<String>,
) -> Result<Response, StatusCode> {
    let sanitized_name = Path::new(&filename)
        .file_name()
        .ok_or(StatusCode::BAD_REQUEST)?
        .to_str()
        .ok_or(StatusCode::BAD_REQUEST)?;

    let file_path = state.workspace_dir.join(sanitized_name);
    if !file_path.exists() {
        return Err(StatusCode::NOT_FOUND);
    }

    let mime_type = mime_guess::from_path(&file_path)
        .first_or_octet_stream()
        .to_string();

    let bytes = tokio::fs::read(&file_path)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let mut response = Response::new(bytes.into());
    response.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        mime_type.parse().unwrap(),
    );
    Ok(response)
}

async fn delete_media(
    State(state): State<AppState>,
    AxumPath(filename): AxumPath<String>,
) -> Result<StatusCode, StatusCode> {
    let sanitized_name = Path::new(&filename)
        .file_name()
        .ok_or(StatusCode::BAD_REQUEST)?
        .to_str()
        .ok_or(StatusCode::BAD_REQUEST)?;

    let file_path = state.workspace_dir.join(sanitized_name);
    if file_path.exists() {
        tokio::fs::remove_file(file_path)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        Ok(StatusCode::OK)
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

async fn upload_media(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<serde_json::Value>, StatusCode> {
    while let Ok(Some(field)) = multipart.next_field().await {
        if let Some(filename) = field.file_name() {
            let sanitized_name = Path::new(filename)
                .file_name()
                .ok_or(StatusCode::BAD_REQUEST)?
                .to_string_lossy()
                .to_string();

            let target_path = state.workspace_dir.join(&sanitized_name);
            let data = field.bytes().await.map_err(|_| StatusCode::BAD_REQUEST)?;

            tokio::fs::write(&target_path, data)
                .await
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

            // Normalisation automatique des vidéos vers H.264 universel pour compatibilité totale navigateur et AI
            let is_vid = sanitized_name.to_lowercase().ends_with(".mp4")
                || sanitized_name.to_lowercase().ends_with(".mov")
                || sanitized_name.to_lowercase().ends_with(".mkv")
                || sanitized_name.to_lowercase().ends_with(".webm");

            if is_vid {
                let temp_input = state.workspace_dir.join(format!("raw_{}", sanitized_name));
                if tokio::fs::rename(&target_path, &temp_input).await.is_ok() {
                    let ffmpeg_cmd = if Path::new("/home/nhou/.local/bin/ffmpeg").exists() {
                        "/home/nhou/.local/bin/ffmpeg"
                    } else {
                        "ffmpeg"
                    };

                    let mut cmd = tokio::process::Command::new(ffmpeg_cmd);
                    cmd.arg("-y")
                        .arg("-i")
                        .arg(&temp_input)
                        .arg("-c:v")
                        .arg("libx264")
                        .arg("-pix_fmt")
                        .arg("yuv420p")
                        .arg("-preset")
                        .arg("veryfast")
                        .arg("-crf")
                        .arg("19")
                        .arg("-c:a")
                        .arg("aac")
                        .arg("-b:a")
                        .arg("128k")
                        .arg(&target_path);

                    if let Ok(output) = cmd.output().await {
                        if output.status.success() {
                            let _ = tokio::fs::remove_file(&temp_input).await;
                        } else {
                            let _ = tokio::fs::rename(&temp_input, &target_path).await;
                        }
                    } else {
                        let _ = tokio::fs::rename(&temp_input, &target_path).await;
                    }
                }
            }

            return Ok(Json(serde_json::json!({
                "status": "success",
                "filename": sanitized_name
            })));
        }
    }
    Err(StatusCode::BAD_REQUEST)
}

#[derive(Debug, Deserialize)]
pub struct CropPayload {
    pub source: String,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub target_width: Option<u32>,
    pub target_height: Option<u32>,
    pub output: Option<String>,
}

async fn crop_media(
    State(state): State<AppState>,
    Json(payload): Json<CropPayload>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let sanitized_source = Path::new(&payload.source)
        .file_name()
        .ok_or(StatusCode::BAD_REQUEST)?
        .to_string_lossy()
        .to_string();

    let source_path = state.workspace_dir.join(&sanitized_source);
    if !source_path.exists() {
        return Err(StatusCode::NOT_FOUND);
    }

    let is_video = sanitized_source.to_lowercase().ends_with(".mp4")
        || sanitized_source.to_lowercase().ends_with(".mov")
        || sanitized_source.to_lowercase().ends_with(".webm")
        || sanitized_source.to_lowercase().ends_with(".mkv");

    let stem = Path::new(&sanitized_source)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("media");
    let ext = Path::new(&sanitized_source)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or(if is_video { "mp4" } else { "png" });

    let raw_output = payload.output.unwrap_or_else(|| {
        format!("{}_crop.{}", stem, ext)
    });

    let unique_path = crate::utils::get_unique_incremental_path(&state.workspace_dir.join(&raw_output));
    let output_filename = unique_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    let ffmpeg_cmd = if let Ok(custom_path) = std::env::var("FFMPEG_PATH") {
        custom_path
    } else if Path::new("/home/nhou/.local/bin/ffmpeg").exists() {
        "/home/nhou/.local/bin/ffmpeg".to_string()
    } else {
        "ffmpeg".to_string()
    };

    // Ensure width and height are even
    let crop_w = (payload.width / 2) * 2;
    let crop_h = (payload.height / 2) * 2;
    let crop_x = payload.x;
    let crop_y = payload.y;

    let mut vf_filter = format!("crop={}:{}:{}:{}", crop_w, crop_h, crop_x, crop_y);
    if let (Some(tw), Some(th)) = (payload.target_width, payload.target_height) {
        if tw > 0 && th > 0 {
            let tw_even = (tw / 2) * 2;
            let th_even = (th / 2) * 2;
            vf_filter.push_str(&format!(",scale={}:{}", tw_even, th_even));
        }
    }

    if is_video {
        vf_filter.push_str(",format=yuv420p");
    }

    let mut cmd = tokio::process::Command::new(&ffmpeg_cmd);
    cmd.arg("-y")
        .arg("-i")
        .arg(&source_path)
        .arg("-vf")
        .arg(&vf_filter);

    if is_video {
        cmd.arg("-c:v")
            .arg("libx264")
            .arg("-preset")
            .arg("fast")
            .arg("-crf")
            .arg("18")
            .arg("-c:a")
            .arg("aac")
            .arg("-b:a")
            .arg("128k");
    }

    cmd.arg(&unique_path);

    let output = cmd.output().await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        eprintln!("Erreur FFmpeg crop: {}", err);
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    Ok(Json(serde_json::json!({
        "status": "success",
        "filename": output_filename,
        "is_video": is_video
    })))
}

// ----------------------------------------------------
// Jobs Handlers & Live Progress Tracking
// ----------------------------------------------------
fn create_client_with_tracking(state: AppState, job_id: String, api_key: String, base_url: String) -> RunpodClient {
    RunpodClient::with_base_url(api_key, base_url).with_progress_callback(move |runpod_id, endpoint, status_str, _elapsed| {
        let state = state.clone();
        let jid = job_id.clone();
        let r_id = runpod_id.to_string();
        let ep = endpoint.to_string();
        let st = status_str.to_string();

        tokio::spawn(async move {
            let mut jobs = state.jobs.lock().await;
            if let Some(j) = jobs.iter_mut().find(|job| job.id == jid) {
                j.runpod_job_id = Some(r_id.clone());
                j.runpod_endpoint = Some(ep.clone());
                
                let is_new_status = j.runpod_status.as_deref() != Some(&st);
                j.runpod_status = Some(st.clone());

                if is_new_status {
                    let msg = match st.as_str() {
                        "IN_QUEUE" => format!("⏳ Job RunPod ({}) en file d'attente...", r_id),
                        "IN_PROGRESS" => format!("⚡ GPU actif : traitement en cours sur '{}' ({})", ep, r_id),
                        "COMPLETED" => format!("✓ Traitement GPU RunPod terminé avec succès ({})", r_id),
                        "FAILED" => format!("❌ Échec du worker RunPod ({})", r_id),
                        _ => format!("ℹ️ Statut RunPod : {}", st),
                    };
                    j.logs.push(JobLog {
                        level: if st == "COMPLETED" { "success".to_string() } else if st == "FAILED" { "error".to_string() } else { "info".to_string() },
                        message: msg,
                    });
                }
            }
        });
    })
}

#[allow(dead_code)]
async fn append_job_log(state: &AppState, job_id: &str, level: &str, message: impl Into<String>) {
    let mut jobs = state.jobs.lock().await;
    if let Some(j) = jobs.iter_mut().find(|job| job.id == job_id) {
        j.logs.push(JobLog {
            level: level.to_string(),
            message: message.into(),
        });
    }
}

async fn list_jobs(State(state): State<AppState>) -> Json<Vec<ServerJob>> {
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut jobs = state.jobs.lock().await;
    for j in jobs.iter_mut() {
        if j.status == "RUNNING" || j.status == "IN_PROGRESS" {
            j.elapsed_seconds = now.saturating_sub(j.created_at);
        }
    }
    Json(jobs.clone())
}

async fn clear_jobs(State(state): State<AppState>) -> StatusCode {
    let mut jobs = state.jobs.lock().await;
    jobs.retain(|j| j.status == "RUNNING" || j.status == "IN_PROGRESS");
    StatusCode::OK
}

// ----------------------------------------------------
// Prompts History Handlers & Helpers
// ----------------------------------------------------
async fn persist_prompts(workspace_dir: &Path, prompts: &[PromptEntry]) {
    let prompts_file = workspace_dir.join("prompts_history.json");
    if let Ok(json_str) = serde_json::to_string_pretty(prompts) {
        let _ = tokio::fs::write(&prompts_file, json_str).await;
    }
}

pub async fn record_prompt_auto(state: &AppState, text: &str, pipeline_type: &str) {
    let text = text.trim();
    if text.is_empty() {
        return;
    }

    let mut prompts = state.prompts.lock().await;
    // Si déjà présent récemment, le remonter en haut
    if let Some(pos) = prompts.iter().position(|p| p.text.eq_ignore_ascii_case(text)) {
        let mut existing = prompts.remove(pos);
        existing.created_at = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        prompts.insert(0, existing);
    } else {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        prompts.insert(
            0,
            PromptEntry {
                id: format!("p_{}", rand::random::<u32>()),
                text: text.to_string(),
                pipeline_type: pipeline_type.to_string(),
                created_at: now,
                is_favorite: false,
            },
        );
    }

    // Conserver un maximum de 200 prompts dans l'historique
    if prompts.len() > 200 {
        prompts.truncate(200);
    }

    persist_prompts(&state.workspace_dir, &prompts).await;
}

async fn list_prompts(State(state): State<AppState>) -> Json<Vec<PromptEntry>> {
    let prompts = state.prompts.lock().await;
    let mut sorted = prompts.clone();
    // Tri : favoris en premier, puis les plus récents
    sorted.sort_by(|a, b| {
        b.is_favorite
            .cmp(&a.is_favorite)
            .then_with(|| b.created_at.cmp(&a.created_at))
    });
    Json(sorted)
}

#[derive(Deserialize)]
pub struct SavePromptPayload {
    pub text: String,
    pub pipeline_type: Option<String>,
}

async fn save_prompt(
    State(state): State<AppState>,
    Json(payload): Json<SavePromptPayload>,
) -> Result<Json<PromptEntry>, StatusCode> {
    let text = payload.text.trim();
    if text.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    let mut prompts = state.prompts.lock().await;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let entry = PromptEntry {
        id: format!("p_{}", rand::random::<u32>()),
        text: text.to_string(),
        pipeline_type: payload.pipeline_type.unwrap_or_else(|| "custom".to_string()),
        created_at: now,
        is_favorite: false,
    };

    prompts.insert(0, entry.clone());
    persist_prompts(&state.workspace_dir, &prompts).await;

    Ok(Json(entry))
}

async fn toggle_favorite_prompt(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> StatusCode {
    let mut prompts = state.prompts.lock().await;
    if let Some(p) = prompts.iter_mut().find(|p| p.id == id) {
        p.is_favorite = !p.is_favorite;
        persist_prompts(&state.workspace_dir, &prompts).await;
        StatusCode::OK
    } else {
        StatusCode::NOT_FOUND
    }
}

async fn delete_prompt(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> StatusCode {
    let mut prompts = state.prompts.lock().await;
    let original_len = prompts.len();
    prompts.retain(|p| p.id != id);
    if prompts.len() != original_len {
        persist_prompts(&state.workspace_dir, &prompts).await;
        StatusCode::OK
    } else {
        StatusCode::NOT_FOUND
    }
}

// ----------------------------------------------------
// Settings Handlers
// ----------------------------------------------------
async fn get_settings(State(state): State<AppState>) -> Json<ServerSettings> {
    let settings = state.settings.lock().await;
    Json(settings.clone())
}

async fn save_settings(
    State(state): State<AppState>,
    Json(new_settings): Json<ServerSettings>,
) -> StatusCode {
    let mut settings = state.settings.lock().await;
    *settings = new_settings;
    StatusCode::OK
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceResponse {
    pub success: bool,
    pub id: Option<String>,
    pub email: Option<String>,
    pub client_balance: Option<f64>,
    pub error: Option<String>,
}

async fn get_balance(State(state): State<AppState>) -> Json<BalanceResponse> {
    let settings = state.settings.lock().await;
    let api_key = settings.api_key.clone();
    drop(settings);

    if api_key.trim().is_empty() {
        return Json(BalanceResponse {
            success: false,
            id: None,
            email: None,
            client_balance: None,
            error: Some("Clé API RunPod non configurée".to_string()),
        });
    }

    let client = RunpodClient::new(api_key);
    match client.get_account_info().await {
        Ok(info) => Json(BalanceResponse {
            success: true,
            id: info.id,
            email: info.email,
            client_balance: info.client_balance,
            error: None,
        }),
        Err(e) => Json(BalanceResponse {
            success: false,
            id: None,
            email: None,
            client_balance: None,
            error: Some(e.to_string()),
        }),
    }
}

// ----------------------------------------------------
// Pipeline Execution Handlers
// ----------------------------------------------------

#[derive(Deserialize)]
pub struct Txt2VidPayload {
    pub prompt: String,
    pub model: Option<String>,
    pub resolution: Option<String>,
    pub duration: Option<u32>,
    pub enhance_prompt: Option<bool>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub output: Option<String>,
    pub local: Option<bool>,
    pub local_model: Option<String>,
}

async fn generate_txt2vid(
    State(state): State<AppState>,
    Json(payload): Json<Txt2VidPayload>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let job_id = format!("t2v_{}", rand::random::<u32>());
    let raw_output = payload.output.unwrap_or_else(|| format!("{}.mp4", job_id));
    let unique_path = crate::utils::get_unique_incremental_path(&state.workspace_dir.join(&raw_output));
    let output_file = unique_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    let prompt_clone = payload.prompt.clone();
    record_prompt_auto(&state, &prompt_clone, "txt2vid").await;

    let video_model = payload
        .model
        .as_deref()
        .and_then(|m| std::str::FromStr::from_str(m).ok())
        .unwrap_or(crate::models::VideoModel::Wan2_5);

    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let job = ServerJob {
        id: job_id.clone(),
        pipeline_type: "txt2vid".to_string(),
        status: "RUNNING".to_string(),
        prompt: Some(prompt_clone.clone()),
        source: None,
        result_file: Some(output_file.clone()),
        elapsed_seconds: 0,
        created_at: now,
        logs: vec![JobLog {
            level: "info".to_string(),
            message: format!("Initialisation Text-to-Video [Flux + {}] : '{}'", video_model.display_name(), prompt_clone),
        }],
        runpod_job_id: None,
        runpod_status: None,
        runpod_endpoint: None,
        stage_info: None,
    };

    {
        let mut jobs = state.jobs.lock().await;
        jobs.insert(0, job);
    }

    let state_clone = state.clone();
    let job_id_clone = job_id.clone();
    let output_file_clone = output_file.clone();

    tokio::spawn(async move {
        let start = Instant::now();
        let (api_key, base_url, flux_ep, video_ep) = {
            let s = state_clone.settings.lock().await;
            (
                s.api_key.clone(),
                s.base_url.clone(),
                s.flux_endpoint.clone(),
                s.resolve_video_endpoint(&video_model),
            )
        };

        let client = create_client_with_tracking(state_clone.clone(), job_id_clone.clone(), api_key, base_url);
        let mut ctx = PipelineContext::new(payload.prompt);

        let mut pipeline = Pipeline::new("Text-to-Video");
        if payload.enhance_prompt.unwrap_or(false) {
            pipeline = pipeline.add_stage(PromptEnhanceStage::new());
        }

        let is_local = payload.local.unwrap_or(false);
        let mut t2i_stage = TextToImageStage::new()
            .with_endpoint(flux_ep)
            .with_dimensions(payload.width.unwrap_or(if is_local { 512 } else { 768 }), payload.height.unwrap_or(if is_local { 512 } else { 1344 }))
            .with_steps(if is_local { 2 } else { 4 })
            .with_local(is_local);

        if let Some(ref m) = payload.local_model {
            t2i_stage = t2i_stage.with_local_model(m.clone());
        }

        pipeline = pipeline
            .add_stage(t2i_stage)
            .add_stage(
                ImageToVideoStage::new()
                    .with_model(video_model)
                    .with_endpoint(video_ep)
                    .with_duration(payload.duration.unwrap_or(5))
                    .with_resolution(payload.resolution.unwrap_or_else(|| "720p".to_string())),
            )
            .add_stage(
                DownloadStage::new()
                    .download_video(state_clone.workspace_dir.join(&output_file_clone)),
            );

        let res = pipeline.run(&mut ctx, &client).await;
        let elapsed = start.elapsed().as_secs();

        let mut jobs = state_clone.jobs.lock().await;
        if let Some(j) = jobs.iter_mut().find(|job| job.id == job_id_clone) {
            j.elapsed_seconds = elapsed;
            match res {
                Ok(_) => {
                    j.status = "COMPLETED".to_string();
                    j.logs.push(JobLog {
                        level: "success".to_string(),
                        message: format!("Vidéo générée avec succès en {}s !", elapsed),
                    });
                }
                Err(e) => {
                    j.status = "FAILED".to_string();
                    j.logs.push(JobLog {
                        level: "error".to_string(),
                        message: format!("Erreur lors de la génération : {:#}", e),
                    });
                }
            }
        }
    });

    Ok(Json(serde_json::json!({ "id": job_id })))
}

#[derive(Deserialize)]
pub struct Img2VidPayload {
    pub image: String,
    pub model: Option<String>,
    pub prompt: Option<String>,
    pub resolution: Option<String>,
    pub duration: Option<u32>,
    pub output: Option<String>,
}

async fn generate_img2vid(
    State(state): State<AppState>,
    Json(payload): Json<Img2VidPayload>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let job_id = format!("i2v_{}", rand::random::<u32>());
    let raw_output = payload.output.unwrap_or_else(|| format!("{}.mp4", job_id));
    let unique_path = crate::utils::get_unique_incremental_path(&state.workspace_dir.join(&raw_output));
    let output_file = unique_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let image_src = payload.image.clone();

    if let Some(ref p) = payload.prompt {
        record_prompt_auto(&state, p, "img2vid").await;
    }

    let video_model = payload
        .model
        .as_deref()
        .and_then(|m| std::str::FromStr::from_str(m).ok())
        .unwrap_or(crate::models::VideoModel::Wan2_5);

    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let job = ServerJob {
        id: job_id.clone(),
        pipeline_type: "img2vid".to_string(),
        status: "RUNNING".to_string(),
        prompt: payload.prompt.clone(),
        source: Some(image_src.clone()),
        result_file: Some(output_file.clone()),
        elapsed_seconds: 0,
        created_at: now,
        logs: vec![JobLog {
            level: "info".to_string(),
            message: format!("Animation Image-to-Video [{}] depuis '{}'", video_model.display_name(), image_src),
        }],
        runpod_job_id: None,
        runpod_status: None,
        runpod_endpoint: None,
        stage_info: None,
    };

    {
        let mut jobs = state.jobs.lock().await;
        jobs.insert(0, job);
    }

    let state_clone = state.clone();
    let job_id_clone = job_id.clone();
    let output_file_clone = output_file.clone();

    tokio::spawn(async move {
        let start = Instant::now();
        let (api_key, base_url, video_ep) = {
            let s = state_clone.settings.lock().await;
            (
                s.api_key.clone(),
                s.base_url.clone(),
                s.resolve_video_endpoint(&video_model),
            )
        };

        let client = create_client_with_tracking(state_clone.clone(), job_id_clone.clone(), api_key, base_url);
        let mut ctx = PipelineContext::new(payload.prompt.clone().unwrap_or_default());

        let pipeline = Pipeline::new("Image-to-Video")
            .add_stage(
                ImageToVideoStage::new()
                    .with_model(video_model)
                    .with_endpoint(video_ep)
                    .with_image(payload.image)
                    .with_duration(payload.duration.unwrap_or(5))
                    .with_resolution(payload.resolution.unwrap_or_else(|| "720p".to_string()))
                    .with_prompt(payload.prompt.unwrap_or_default()),
            )
            .add_stage(
                DownloadStage::new()
                    .download_video(state_clone.workspace_dir.join(&output_file_clone)),
            );

        let res = pipeline.run(&mut ctx, &client).await;
        let elapsed = start.elapsed().as_secs();

        let mut jobs = state_clone.jobs.lock().await;
        if let Some(j) = jobs.iter_mut().find(|job| job.id == job_id_clone) {
            j.elapsed_seconds = elapsed;
            match res {
                Ok(_) => {
                    j.status = "COMPLETED".to_string();
                    j.logs.push(JobLog {
                        level: "success".to_string(),
                        message: format!("Animation terminée avec succès en {}s !", elapsed),
                    });
                }
                Err(e) => {
                    j.status = "FAILED".to_string();
                    j.logs.push(JobLog {
                        level: "error".to_string(),
                        message: format!("Erreur : {:#}", e),
                    });
                }
            }
        }
    });

    Ok(Json(serde_json::json!({ "id": job_id })))
}

#[derive(Deserialize)]
pub struct FaceSwapPayload {
    pub source: String,
    pub target: String,
    pub restore_face: Option<bool>,
    pub face_index: Option<u32>,
    pub output: Option<String>,
}

async fn generate_faceswap(
    State(state): State<AppState>,
    Json(payload): Json<FaceSwapPayload>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let job_id = format!("fs_{}", rand::random::<u32>());
    let target_lower = payload.target.to_lowercase();
    let is_video = target_lower.ends_with(".mp4")
        || target_lower.ends_with(".webm")
        || target_lower.ends_with(".mov")
        || target_lower.ends_with(".mkv");
    let raw_output = payload.output.unwrap_or_else(|| {
        if is_video {
            format!("{}.mp4", job_id)
        } else {
            format!("{}.png", job_id)
        }
    });
    let unique_path = crate::utils::get_unique_incremental_path(&state.workspace_dir.join(&raw_output));
    let output_file = unique_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let job = ServerJob {
        id: job_id.clone(),
        pipeline_type: "faceswap".to_string(),
        status: "RUNNING".to_string(),
        prompt: None,
        source: Some(format!("{} ➔ {}", payload.source, payload.target)),
        result_file: Some(output_file.clone()),
        elapsed_seconds: 0,
        created_at: now,
        logs: vec![JobLog {
            level: "info".to_string(),
            message: format!("Face Swap de '{}' sur '{}'", payload.source, payload.target),
        }],
        runpod_job_id: None,
        runpod_status: None,
        runpod_endpoint: None,
        stage_info: None,
    };

    {
        let mut jobs = state.jobs.lock().await;
        jobs.insert(0, job);
    }

    let state_clone = state.clone();
    let job_id_clone = job_id.clone();
    let output_file_clone = output_file.clone();

    tokio::spawn(async move {
        let start = Instant::now();
        let (api_key, base_url, fs_ep) = {
            let s = state_clone.settings.lock().await;
            (
                s.api_key.clone(),
                s.base_url.clone(),
                s.faceswap_endpoint.clone(),
            )
        };

        let client = create_client_with_tracking(state_clone.clone(), job_id_clone.clone(), api_key, base_url);
        let mut ctx = PipelineContext::default();

        let stage = FaceSwapStage::new(payload.source, payload.target)
            .with_endpoint(fs_ep)
            .with_restore_face(payload.restore_face.unwrap_or(true))
            .with_face_index(payload.face_index.unwrap_or(0));

        let download = if is_video {
            DownloadStage::new().download_video(state_clone.workspace_dir.join(&output_file_clone))
        } else {
            DownloadStage::new().download_image(state_clone.workspace_dir.join(&output_file_clone))
        };

        let pipeline = Pipeline::new("Face-Swap")
            .add_stage(stage)
            .add_stage(download);

        let res = pipeline.run(&mut ctx, &client).await;
        let elapsed = start.elapsed().as_secs();

        let mut jobs = state_clone.jobs.lock().await;
        if let Some(j) = jobs.iter_mut().find(|job| job.id == job_id_clone) {
            j.elapsed_seconds = elapsed;
            match res {
                Ok(_) => {
                    j.status = "COMPLETED".to_string();
                    j.logs.push(JobLog {
                        level: "success".to_string(),
                        message: format!("Face Swap terminé avec succès en {}s !", elapsed),
                    });
                }
                Err(e) => {
                    j.status = "FAILED".to_string();
                    j.logs.push(JobLog {
                        level: "error".to_string(),
                        message: format!("Erreur : {:#}", e),
                    });
                }
            }
        }
    });

    Ok(Json(serde_json::json!({ "id": job_id })))
}

#[derive(Deserialize)]
pub struct Vid2VidPayload {
    pub video: String,
    pub model: Option<String>,
    pub prompt: String,
    pub strength: Option<f32>,
    pub resolution: Option<String>,
    pub output: Option<String>,
}

async fn generate_vid2vid(
    State(state): State<AppState>,
    Json(payload): Json<Vid2VidPayload>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let job_id = format!("v2v_{}", rand::random::<u32>());
    let raw_output = payload.output.unwrap_or_else(|| format!("{}.mp4", job_id));
    let unique_path = crate::utils::get_unique_incremental_path(&state.workspace_dir.join(&raw_output));
    let output_file = unique_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    record_prompt_auto(&state, &payload.prompt, "vid2vid").await;

    let video_model = payload
        .model
        .as_deref()
        .and_then(|m| std::str::FromStr::from_str(m).ok())
        .unwrap_or(crate::models::VideoModel::Wan2_5);

    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let job = ServerJob {
        id: job_id.clone(),
        pipeline_type: "vid2vid".to_string(),
        status: "RUNNING".to_string(),
        prompt: Some(payload.prompt.clone()),
        source: Some(payload.video.clone()),
        result_file: Some(output_file.clone()),
        elapsed_seconds: 0,
        created_at: now,
        logs: vec![JobLog {
            level: "info".to_string(),
            message: format!("Restyling Vid2Vid [{}] sur '{}'", video_model.display_name(), payload.video),
        }],
        runpod_job_id: None,
        runpod_status: None,
        runpod_endpoint: None,
        stage_info: None,
    };

    {
        let mut jobs = state.jobs.lock().await;
        jobs.insert(0, job);
    }

    let state_clone = state.clone();
    let job_id_clone = job_id.clone();
    let output_file_clone = output_file.clone();

    tokio::spawn(async move {
        let start = Instant::now();
        let (api_key, base_url, video_ep) = {
            let s = state_clone.settings.lock().await;
            (
                s.api_key.clone(),
                s.base_url.clone(),
                s.resolve_video_endpoint(&video_model),
            )
        };

        let client = create_client_with_tracking(state_clone.clone(), job_id_clone.clone(), api_key, base_url);
        let mut ctx = PipelineContext::new(payload.prompt.clone());

        let stage = VideoToVideoStage::new()
            .with_model(video_model)
            .with_endpoint(video_ep)
            .with_video(payload.video)
            .with_strength(payload.strength.unwrap_or(0.65))
            .with_resolution(payload.resolution.unwrap_or_else(|| "720p".to_string()))
            .with_prompt(payload.prompt);

        let pipeline = Pipeline::new("Video-to-Video")
            .add_stage(stage)
            .add_stage(
                DownloadStage::new()
                    .download_video(state_clone.workspace_dir.join(&output_file_clone)),
            );

        let res = pipeline.run(&mut ctx, &client).await;
        let elapsed = start.elapsed().as_secs();

        let mut jobs = state_clone.jobs.lock().await;
        if let Some(j) = jobs.iter_mut().find(|job| job.id == job_id_clone) {
            j.elapsed_seconds = elapsed;
            match res {
                Ok(_) => {
                    j.status = "COMPLETED".to_string();
                    j.logs.push(JobLog {
                        level: "success".to_string(),
                        message: format!("Transformation Vid2Vid terminée en {}s !", elapsed),
                    });
                }
                Err(e) => {
                    j.status = "FAILED".to_string();
                    j.logs.push(JobLog {
                        level: "error".to_string(),
                        message: format!("Erreur : {:#}", e),
                    });
                }
            }
        }
    });

    Ok(Json(serde_json::json!({ "id": job_id })))
}

#[derive(Deserialize)]
pub struct Txt2ImgPayload {
    pub prompt: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub steps: Option<u32>,
    pub output: Option<String>,
    pub local: Option<bool>,
    pub local_model: Option<String>,
}

async fn generate_txt2img(
    State(state): State<AppState>,
    Json(payload): Json<Txt2ImgPayload>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let job_id = format!("t2i_{}", rand::random::<u32>());
    let raw_output = payload.output.unwrap_or_else(|| format!("{}.png", job_id));
    let unique_path = crate::utils::get_unique_incremental_path(&state.workspace_dir.join(&raw_output));
    let output_file = unique_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    record_prompt_auto(&state, &payload.prompt, "txt2img").await;

    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let job = ServerJob {
        id: job_id.clone(),
        pipeline_type: "txt2img".to_string(),
        status: "RUNNING".to_string(),
        prompt: Some(payload.prompt.clone()),
        source: None,
        result_file: Some(output_file.clone()),
        elapsed_seconds: 0,
        created_at: now,
        logs: vec![JobLog {
            level: "info".to_string(),
            message: format!("Génération Flux: '{}'", payload.prompt),
        }],
        runpod_job_id: None,
        runpod_status: None,
        runpod_endpoint: None,
        stage_info: None,
    };

    {
        let mut jobs = state.jobs.lock().await;
        jobs.insert(0, job);
    }

    let state_clone = state.clone();
    let job_id_clone = job_id.clone();
    let output_file_clone = output_file.clone();

    tokio::spawn(async move {
        let start = Instant::now();
        let (api_key, base_url, flux_ep) = {
            let s = state_clone.settings.lock().await;
            (
                s.api_key.clone(),
                s.base_url.clone(),
                s.flux_endpoint.clone(),
            )
        };

        let client = create_client_with_tracking(state_clone.clone(), job_id_clone.clone(), api_key, base_url);
        let mut ctx = PipelineContext::new(payload.prompt);

        let is_local = payload.local.unwrap_or(false);
        let mut stage = TextToImageStage::new()
            .with_endpoint(flux_ep)
            .with_dimensions(payload.width.unwrap_or(if is_local { 512 } else { 1024 }), payload.height.unwrap_or(if is_local { 512 } else { 1024 }))
            .with_steps(payload.steps.unwrap_or(if is_local { 2 } else { 4 }))
            .with_local(is_local);

        if let Some(ref m) = payload.local_model {
            stage = stage.with_local_model(m.clone());
        }

        let pipeline = Pipeline::new("Text-to-Image")
            .add_stage(stage)
            .add_stage(
                DownloadStage::new()
                    .download_image(state_clone.workspace_dir.join(&output_file_clone)),
            );

        let res = pipeline.run(&mut ctx, &client).await;
        let elapsed = start.elapsed().as_secs();

        let mut jobs = state_clone.jobs.lock().await;
        if let Some(j) = jobs.iter_mut().find(|job| job.id == job_id_clone) {
            j.elapsed_seconds = elapsed;
            match res {
                Ok(_) => {
                    j.status = "COMPLETED".to_string();
                    j.logs.push(JobLog {
                        level: "success".to_string(),
                        message: format!("Image générée avec succès en {}s !", elapsed),
                    });
                }
                Err(e) => {
                    j.status = "FAILED".to_string();
                    j.logs.push(JobLog {
                        level: "error".to_string(),
                        message: format!("Erreur : {:#}", e),
                    });
                }
            }
        }
    });

    Ok(Json(serde_json::json!({ "id": job_id })))
}

#[derive(Deserialize)]
pub struct Img2ImgPayload {
    pub image: String,
    pub mask: Option<String>,
    pub prompt: String,
    pub strength: Option<f32>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub steps: Option<u32>,
    pub output: Option<String>,
    pub local: Option<bool>,
    pub local_model: Option<String>,
}

async fn generate_img2img(
    State(state): State<AppState>,
    Json(payload): Json<Img2ImgPayload>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let job_id = format!("i2i_{}", rand::random::<u32>());
    let raw_output = payload.output.unwrap_or_else(|| format!("{}.png", job_id));
    let unique_path = crate::utils::get_unique_incremental_path(&state.workspace_dir.join(&raw_output));
    let output_file = unique_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    record_prompt_auto(&state, &payload.prompt, "img2img").await;

    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let is_inpaint = payload.mask.is_some();
    let job = ServerJob {
        id: job_id.clone(),
        pipeline_type: if is_inpaint { "inpaint".to_string() } else { "img2img".to_string() },
        status: "RUNNING".to_string(),
        prompt: Some(payload.prompt.clone()),
        source: Some(payload.image.clone()),
        result_file: Some(output_file.clone()),
        elapsed_seconds: 0,
        created_at: now,
        logs: vec![JobLog {
            level: "info".to_string(),
            message: if is_inpaint {
                format!("Inpainting (Masqué) depuis '{}' : '{}'", payload.image, payload.prompt)
            } else {
                format!("Génération Image-to-Image depuis '{}' : '{}'", payload.image, payload.prompt)
            },
        }],
        runpod_job_id: None,
        runpod_status: None,
        runpod_endpoint: None,
        stage_info: None,
    };

    {
        let mut jobs = state.jobs.lock().await;
        jobs.insert(0, job);
    }

    let state_clone = state.clone();
    let job_id_clone = job_id.clone();
    let output_file_clone = output_file.clone();

    tokio::spawn(async move {
        let start = Instant::now();
        let (api_key, base_url, flux_ep) = {
            let s = state_clone.settings.lock().await;
            (
                s.api_key.clone(),
                s.base_url.clone(),
                s.flux_endpoint.clone(),
            )
        };

        let client = create_client_with_tracking(state_clone.clone(), job_id_clone.clone(), api_key, base_url);
        let mut ctx = PipelineContext::new(payload.prompt.clone());

        let is_local = payload.local.unwrap_or(false);
        let mut stage = ImageToImageStage::new()
            .with_endpoint(flux_ep)
            .with_image(payload.image)
            .with_strength(payload.strength.unwrap_or(0.70))
            .with_steps(payload.steps.unwrap_or(if is_local { 4 } else { 4 }))
            .with_local(is_local);

        if let Some(mask) = payload.mask {
            stage = stage.with_mask(mask);
        }

        if let (Some(w), Some(h)) = (payload.width, payload.height) {
            stage = stage.with_dimensions(w, h);
        }

        if let Some(ref m) = payload.local_model {
            stage = stage.with_local_model(m.clone());
        }

        let pipeline = Pipeline::new("Image-to-Image")
            .add_stage(stage)
            .add_stage(
                DownloadStage::new()
                    .download_image(state_clone.workspace_dir.join(&output_file_clone)),
            );

        let res = pipeline.run(&mut ctx, &client).await;
        let elapsed = start.elapsed().as_secs();

        let mut jobs = state_clone.jobs.lock().await;
        if let Some(j) = jobs.iter_mut().find(|job| job.id == job_id_clone) {
            j.elapsed_seconds = elapsed;
            match res {
                Ok(_) => {
                    j.status = "COMPLETED".to_string();
                    j.logs.push(JobLog {
                        level: "success".to_string(),
                        message: format!("Image-to-Image générée avec succès en {}s !", elapsed),
                    });
                }
                Err(e) => {
                    j.status = "FAILED".to_string();
                    j.logs.push(JobLog {
                        level: "error".to_string(),
                        message: format!("Erreur : {:#}", e),
                    });
                }
            }
        }
    });

    Ok(Json(serde_json::json!({ "id": job_id })))
}

#[derive(Deserialize)]
pub struct EnhancePayload {
    pub prompt: String,
}

async fn generate_enhance(
    State(state): State<AppState>,
    Json(payload): Json<EnhancePayload>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let (api_key, base_url, llm_ep) = {
        let s = state.settings.lock().await;
        (s.api_key.clone(), s.base_url.clone(), s.llm_endpoint.clone())
    };

    let original_prompt = payload.prompt.clone();
    record_prompt_auto(&state, &original_prompt, "original").await;

    let client = RunpodClient::with_base_url(api_key, base_url);
    let mut ctx = PipelineContext::new(payload.prompt);

    let stage = PromptEnhanceStage::new().with_endpoint(llm_ep, "Qwen/Qwen3-32B-AWQ");
    let pipeline = Pipeline::new("Prompt-Enhance").add_stage(stage);

    let _ = pipeline.run(&mut ctx, &client).await;

    let enhanced = ctx
        .enhanced_prompt
        .filter(|p| !p.trim().is_empty())
        .unwrap_or_else(|| {
            crate::pipeline::stages::prompt_enhance::smart_enhance_prompt(&original_prompt)
        });

    record_prompt_auto(&state, &enhanced, "enhanced").await;

    Ok(Json(serde_json::json!({
        "status": "success",
        "enhanced_prompt": enhanced
    })))
}
