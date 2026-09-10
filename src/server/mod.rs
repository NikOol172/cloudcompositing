use crate::client::RunpodClient;
use crate::pipeline::stages::{
    DownloadStage, FaceSwapStage, ImageToImageStage, ImageToVideoStage, PromptEnhanceStage, TextToImageStage,
    TextToSpeechStage, VideoToVideoStage,
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
use tokio::io::AsyncBufReadExt;
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::ServeDir;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaItem {
    pub name: String,
    pub size_bytes: u64,
    pub is_video: bool,
    #[serde(default)]
    pub is_audio: bool,
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
    #[serde(default)]
    pub hf_token: String,
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
        let hf_tok = std::env::var("HF_TOKEN")
            .or_else(|_| std::env::var("HUGGINGFACE_HUB_TOKEN"))
            .unwrap_or_default();

        Self {
            api_key: std::env::var("RUNPOD_API_KEY").unwrap_or_default(),
            hf_token: hf_tok,
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

    let settings_file = workspace_dir.join(".settings.json");
    let mut settings = if settings_file.exists() {
        tokio::fs::read_to_string(&settings_file)
            .await
            .ok()
            .and_then(|s| serde_json::from_str::<ServerSettings>(&s).ok())
            .unwrap_or_default()
    } else {
        ServerSettings::default()
    };

    if settings.api_key.trim().is_empty() {
        if let Ok(key) = std::env::var("RUNPOD_API_KEY") {
            settings.api_key = key;
        }
    }
    if settings.hf_token.trim().is_empty() {
        let hf_tok = std::env::var("HF_TOKEN")
            .or_else(|_| std::env::var("HUGGINGFACE_HUB_TOKEN"))
            .unwrap_or_default();
        if !hf_tok.is_empty() {
            settings.hf_token = hf_tok;
        }
    }

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
        .route("/api/system/info", get(get_system_info))
        .route("/api/models/status", get(get_models_status))
        .route("/api/models/download", post(download_model))
        .route("/api/crop", post(crop_media))
        .route("/api/generate/txt2vid", post(generate_txt2vid))
        .route("/api/generate/img2vid", post(generate_img2vid))
        .route("/api/generate/faceswap", post(generate_faceswap))
        .route("/api/generate/vid2vid", post(generate_vid2vid))
        .route("/api/generate/txt2img", post(generate_txt2img))
        .route("/api/generate/img2img", post(generate_img2img))
        .route("/api/generate/enhance", post(generate_enhance))
        .route("/api/generate/tts", post(generate_tts))
        .route("/api/loras", get(list_loras))
        .route("/api/lora/datasets", get(list_lora_datasets))
        .route("/api/lora/datasets/:name/upload", post(upload_dataset_images))
        .route("/api/lora/datasets/:name/image/:image", get(serve_dataset_image))
        .route("/api/lora/autocaption", post(autocaption_dataset))
        .route("/api/lora/caption", post(save_dataset_caption))
        .route("/api/lora/train", post(start_lora_training))
        .route("/api/network", get(get_network_info))
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

    println!("\n{}", style("🚀 RunPod Studio & Media Manager ready!").bold().magenta());
    println!("  • Local Web UI   : {}", style(&url).cyan().underlined().bold());
    if let Some(lan_ip) = crate::utils::get_local_lan_ip() {
        println!("  • Mobile / Wi-Fi : {}", style(format!("http://{}:{}", lan_ip, port)).green().underlined().bold());
    }
    println!("  • Network Listen : {}", style(&addr).dim());
    println!("  • Press {} to stop.\n", style("Ctrl+C").yellow().bold());

    if open_browser {
        let _ = open::that_detached(&url);
    }

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn serve_embedded_fallback() -> impl IntoResponse {
    Html("<!DOCTYPE html><html><body><h1>RunPod Studio</h1><p>Web folder not found.</p></body></html>")
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
                    let is_aud = matches!(ext_lower.as_str(), "wav" | "mp3" | "ogg" | "flac" | "m4a" | "aac");

                    if is_img || is_vid || is_aud {
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
                                    is_audio: is_aud,
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

#[derive(Serialize)]
pub struct LoraItem {
    pub filename: String,
    pub path: String,
    pub size_mb: f64,
}

async fn list_loras(State(state): State<AppState>) -> Json<Vec<LoraItem>> {
    let loras_dir = state.workspace_dir.join("loras");
    let mut items = Vec::new();

    if let Ok(entries) = std::fs::read_dir(&loras_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
                if ext == "safetensors" || ext == "pt" || ext == "bin" {
                    let filename = entry.file_name().to_string_lossy().to_string();
                    let metadata = entry.metadata().ok();
                    let size_bytes = metadata.map(|m| m.len()).unwrap_or(0);
                    let size_mb = (size_bytes as f64) / (1024.0 * 1024.0);
                    items.push(LoraItem {
                        filename,
                        path: path.to_string_lossy().to_string(),
                        size_mb: (size_mb * 10.0).round() / 10.0,
                    });
                }
            }
        }
    }

    items.sort_by(|a, b| a.filename.to_lowercase().cmp(&b.filename.to_lowercase()));
    Json(items)
}

#[derive(Serialize)]
pub struct NetworkInfo {
    pub lan_ip: Option<String>,
    pub port: u16,
    pub mobile_url: String,
    pub runpod_pod_id: Option<String>,
    pub is_runpod: bool,
}

async fn get_network_info() -> Json<NetworkInfo> {
    let port = 3000;
    let pod_id = std::env::var("RUNPOD_POD_ID")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    let (mobile_url, is_runpod) = if let Some(ref id) = pod_id {
        (format!("https://{}-{}.proxy.runpod.net", id, port), true)
    } else if let Some(ref ip) = crate::utils::get_local_lan_ip() {
        (format!("http://{}:{}", ip, port), false)
    } else {
        (format!("http://localhost:{}", port), false)
    };

    Json(NetworkInfo {
        lan_ip: crate::utils::get_local_lan_ip(),
        port,
        mobile_url,
        runpod_pod_id: pod_id,
        is_runpod,
    })
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
        eprintln!("FFmpeg crop error: {}", err);
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
                        "IN_QUEUE" => format!("⏳ RunPod Job ({}) queued...", r_id),
                        "IN_PROGRESS" => format!("⚡ GPU Active: processing on '{}' ({})", ep, r_id),
                        "COMPLETED" => format!("✓ RunPod GPU processing completed successfully ({})", r_id),
                        "FAILED" => format!("❌ RunPod worker failed ({})", r_id),
                        _ => format!("ℹ️ RunPod Status: {}", st),
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
    if !new_settings.hf_token.trim().is_empty() {
        std::env::set_var("HF_TOKEN", new_settings.hf_token.trim());
        std::env::set_var("HUGGINGFACE_HUB_TOKEN", new_settings.hf_token.trim());
    }
    if !new_settings.api_key.trim().is_empty() {
        std::env::set_var("RUNPOD_API_KEY", new_settings.api_key.trim());
    }

    let settings_file = state.workspace_dir.join(".settings.json");
    if let Ok(data) = serde_json::to_string_pretty(&new_settings) {
        let _ = tokio::fs::write(&settings_file, data).await;
    }

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
            error: Some("RunPod API key not configured".to_string()),
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub has_gpu: bool,
    pub gpu_name: Option<String>,
    pub vram_mb: Option<u64>,
}

async fn get_system_info() -> Json<SystemInfo> {
    // 1. Essayer nvidia-smi (rapide et standard)
    if let Ok(output) = tokio::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        .output()
        .await
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Some(line) = stdout.lines().next() {
                let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if !parts.is_empty() && !parts[0].is_empty() {
                    let name = parts[0].to_string();
                    let vram = parts.get(1).and_then(|v| v.parse::<u64>().ok());
                    return Json(SystemInfo {
                        has_gpu: true,
                        gpu_name: Some(name),
                        vram_mb: vram,
                    });
                }
            }
        }
    }

    // 2. Fallback via Python si disponible
    let py_bin = crate::utils::get_python_binary();
    if let Ok(output) = tokio::process::Command::new(&py_bin)
        .args(["-c", "import torch; print(f'{torch.cuda.get_device_name(0)}|{int(torch.cuda.get_device_properties(0).total_memory/1024/1024)}') if torch.cuda.is_available() else print('NO')"])
        .output()
        .await
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let trimmed = stdout.trim();
            if trimmed != "NO" && trimmed.contains('|') {
                let parts: Vec<&str> = trimmed.split('|').collect();
                let name = parts[0].to_string();
                let vram = parts.get(1).and_then(|v| v.parse::<u64>().ok());
                return Json(SystemInfo {
                    has_gpu: true,
                    gpu_name: Some(name),
                    vram_mb: vram,
                });
            }
        }
    }

    Json(SystemInfo {
        has_gpu: false,
        gpu_name: None,
        vram_mb: None,
    })
}

// ----------------------------------------------------
// Models Status & Local Model Downloader
// ----------------------------------------------------

pub fn check_ltx_installed(workspace_dir: &Path) -> (bool, u64, PathBuf) {
    let mut candidates = vec![
        workspace_dir.join("models").join("ltx-video"),
        workspace_dir.join("models").join("LTX-Video"),
        PathBuf::from("/workspace/models/ltx-video"),
        PathBuf::from("/workspace/models/LTX-Video"),
        workspace_dir.join(".hf_cache").join("hub").join("models--Lightricks--LTX-Video"),
        PathBuf::from("/workspace/.hf_cache/hub/models--Lightricks--LTX-Video"),
    ];

    if let Ok(home) = std::env::var("HOME") {
        candidates.push(PathBuf::from(home).join(".cache/huggingface/hub/models--Lightricks--LTX-Video"));
    }

    for candidate in &candidates {
        if candidate.exists() {
            let model_index = candidate.join("model_index.json");
            let snapshots = candidate.join("snapshots");
            if model_index.exists() || snapshots.exists() || candidate.join("transformer").exists() {
                let size = get_dir_size(candidate);
                if size > 100 * 1024 * 1024 {
                    return (true, size, candidate.clone());
                }
            }
        }
    }

    let default_dest = if Path::new("/workspace").exists() {
        PathBuf::from("/workspace/models/ltx-video")
    } else {
        workspace_dir.join("models").join("ltx-video")
    };

    (false, 0, default_dest)
}

fn get_dir_size(path: &Path) -> u64 {
    let mut total = 0;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_file() {
                if let Ok(meta) = p.metadata() {
                    total += meta.len();
                }
            } else if p.is_dir() {
                total += get_dir_size(&p);
            }
        }
    }
    total
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelItemInfo {
    pub id: String,
    pub name: String,
    pub repo_id: String,
    pub installed: bool,
    pub size_gb: f64,
    pub path: String,
    pub requires_hf_token: bool,
    pub license_url: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsStatusResponse {
    pub models: Vec<ModelItemInfo>,
}

async fn get_models_status(State(state): State<AppState>) -> Json<ModelsStatusResponse> {
    let (ltx_installed, ltx_size, ltx_path) = check_ltx_installed(&state.workspace_dir);
    let ltx_size_gb = (ltx_size as f64) / (1024.0 * 1024.0 * 1024.0);

    let models = vec![
        ModelItemInfo {
            id: "ltx-video".to_string(),
            name: "Lightricks LTX-Video 2.5".to_string(),
            repo_id: "Lightricks/LTX-Video".to_string(),
            installed: ltx_installed,
            size_gb: (ltx_size_gb * 100.0).round() / 100.0,
            path: ltx_path.to_string_lossy().to_string(),
            requires_hf_token: true,
            license_url: "https://huggingface.co/Lightricks/LTX-Video".to_string(),
            description: "High-performance DiT video generation engine for Pod GPU (~11 GB)".to_string(),
        },
    ];

    Json(ModelsStatusResponse { models })
}

#[derive(Debug, Deserialize)]
pub struct DownloadModelPayload {
    pub model_id: Option<String>,
}

async fn download_model(
    State(state): State<AppState>,
    Json(payload): Json<DownloadModelPayload>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let model_id_req = payload.model_id.unwrap_or_else(|| "ltx-video".to_string());

    let (repo_id, target_dir) = if model_id_req.to_lowercase().contains("ltx") {
        let (_, _, path) = check_ltx_installed(&state.workspace_dir);
        ("Lightricks/LTX-Video".to_string(), path)
    } else {
        return Err(StatusCode::BAD_REQUEST);
    };

    // Check if download is already running
    {
        let jobs = state.jobs.lock().await;
        if let Some(existing) = jobs.iter().find(|j| {
            j.pipeline_type == "model_download" && j.status == "RUNNING"
        }) {
            return Ok(Json(serde_json::json!({
                "job_id": existing.id,
                "already_running": true,
                "status": "RUNNING"
            })));
        }
    }

    let job_id = format!("dl_{}", rand::random::<u32>());
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let job = ServerJob {
        id: job_id.clone(),
        pipeline_type: "model_download".to_string(),
        status: "RUNNING".to_string(),
        prompt: Some(format!("Download {}", repo_id)),
        source: None,
        result_file: Some(target_dir.to_string_lossy().to_string()),
        elapsed_seconds: 0,
        created_at: now,
        logs: vec![JobLog {
            level: "info".to_string(),
            message: format!("Starting download of {} to {}", repo_id, target_dir.display()),
        }],
        runpod_job_id: None,
        runpod_status: None,
        runpod_endpoint: None,
        stage_info: Some("Initializing...".to_string()),
    };

    {
        let mut jobs = state.jobs.lock().await;
        jobs.insert(0, job);
    }

    let state_clone = state.clone();
    let job_id_clone = job_id.clone();
    let repo_id_clone = repo_id.clone();
    let target_dir_clone = target_dir.clone();

    tokio::spawn(async move {
        let start = Instant::now();
        let hf_token = {
            let s = state_clone.settings.lock().await;
            s.hf_token.clone()
        };

        let mut cmd = tokio::process::Command::new(crate::utils::get_python_binary());
        crate::utils::configure_python_command(&mut cmd);
        let mut args = vec![
            "src/model_downloader.py".to_string(),
            "--model-id".to_string(),
            repo_id_clone.clone(),
            "--local-dir".to_string(),
            target_dir_clone.to_string_lossy().to_string(),
        ];
        if !hf_token.trim().is_empty() {
            args.push("--token".to_string());
            args.push(hf_token.trim().to_string());
        }
        cmd.args(&args);

        cmd.stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                let mut jobs = state_clone.jobs.lock().await;
                if let Some(j) = jobs.iter_mut().find(|job| job.id == job_id_clone) {
                    j.status = "FAILED".to_string();
                    j.logs.push(JobLog {
                        level: "error".to_string(),
                        message: format!("Impossible de lancer le script model_downloader.py : {}", e),
                    });
                }
                return;
            }
        };

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
                            let line_str: String = l;
                            let mut jobs = state_clone.jobs.lock().await;
                            if let Some(j) = jobs.iter_mut().find(|job| job.id == job_id_clone) {
                                j.elapsed_seconds = start.elapsed().as_secs();
                                if line_str.contains("[STATUS]") || line_str.contains("%|") {
                                    j.stage_info = Some(line_str.clone());
                                }
                                j.logs.push(JobLog {
                                    level: "info".to_string(),
                                    message: line_str,
                                });
                            }
                        }
                        _ => stdout_done = true,
                    }
                }
                line = stderr_reader.next_line(), if !stderr_done => {
                    match line {
                        Ok(Some(l)) => {
                            let line_str: String = l;
                            let mut jobs = state_clone.jobs.lock().await;
                            if let Some(j) = jobs.iter_mut().find(|job| job.id == job_id_clone) {
                                j.logs.push(JobLog {
                                    level: "warn".to_string(),
                                    message: line_str,
                                });
                            }
                        }
                        _ => stderr_done = true,
                    }
                }
            }
        }

        let status = child.wait().await;
        let elapsed = start.elapsed().as_secs();

        let mut jobs = state_clone.jobs.lock().await;
        if let Some(j) = jobs.iter_mut().find(|job| job.id == job_id_clone) {
            j.elapsed_seconds = elapsed;
            match status {
                Ok(s) if s.success() => {
                    j.status = "COMPLETED".to_string();
                    j.stage_info = Some("Download completed successfully!".to_string());
                    j.logs.push(JobLog {
                        level: "success".to_string(),
                        message: format!("🎉 Model {} downloaded successfully in {}s!", repo_id_clone, elapsed),
                    });
                }
                Ok(s) => {
                    j.status = "FAILED".to_string();
                    let code = s.code().unwrap_or(-1);
                    if code == 41 {
                        j.stage_info = Some("Error 401: Invalid Hugging Face Token".to_string());
                        j.logs.push(JobLog {
                            level: "error".to_string(),
                            message: "Invalid or expired Hugging Face access token (401). Please configure your token in Settings.".to_string(),
                        });
                    } else if code == 43 {
                        j.stage_info = Some("Error 403: Hugging Face License required".to_string());
                        j.logs.push(JobLog {
                            level: "error".to_string(),
                            message: format!("Access denied to gated model '{}'. Please accept terms on https://huggingface.co/{} then configure your token.", repo_id_clone, repo_id_clone),
                        });
                    } else {
                        j.stage_info = Some("Download failed".to_string());
                        j.logs.push(JobLog {
                            level: "error".to_string(),
                            message: format!("Download stopped with exit code {}", code),
                        });
                    }
                }
                Err(e) => {
                    j.status = "FAILED".to_string();
                    j.logs.push(JobLog {
                        level: "error".to_string(),
                        message: format!("System error waiting for process: {}", e),
                    });
                }
            }
        }
    });

    Ok(Json(serde_json::json!({
        "job_id": job_id,
        "repo_id": repo_id,
        "target_dir": target_dir.to_string_lossy().to_string(),
        "status": "RUNNING"
    })))
}

fn spawn_local_ltx_job(
    state: AppState,
    job_id: String,
    prompt: String,
    image_src: Option<String>,
    output_file: String,
    resolution: String,
    duration: u32,
) {
    tokio::spawn(async move {
        let start = Instant::now();
        let (installed, _, model_path) = check_ltx_installed(&state.workspace_dir);
        if !installed {
            let mut jobs = state.jobs.lock().await;
            if let Some(j) = jobs.iter_mut().find(|job| job.id == job_id) {
                j.status = "FAILED".to_string();
                j.logs.push(JobLog {
                    level: "error".to_string(),
                    message: "LTX-Video 2.5 model is not yet installed on this Pod GPU (~11 GB). Please download it first in Settings with your authorized Hugging Face token.".to_string(),
                });
            }
            return;
        }

        let mut cmd = tokio::process::Command::new(crate::utils::get_python_binary());
        crate::utils::configure_python_command(&mut cmd);
        let out_file_path = state.workspace_dir.join(&output_file);
        let mut args = vec![
            "src/ltx_engine.py".to_string(),
            "--prompt".to_string(),
            prompt.clone(),
            "--duration".to_string(),
            duration.to_string(),
            "--resolution".to_string(),
            resolution.clone(),
            "--output".to_string(),
            out_file_path.to_string_lossy().to_string(),
            "--model-id".to_string(),
            model_path.to_string_lossy().to_string(),
        ];
        if let Some(ref img) = image_src {
            let img_path = state.workspace_dir.join(img);
            let img_str = if img_path.exists() {
                img_path.to_string_lossy().to_string()
            } else {
                img.clone()
            };
            args.push("--image".to_string());
            args.push(img_str);
        }
        cmd.args(&args);

        cmd.stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                let mut jobs = state.jobs.lock().await;
                if let Some(j) = jobs.iter_mut().find(|job| job.id == job_id) {
                    j.status = "FAILED".to_string();
                    j.logs.push(JobLog {
                        level: "error".to_string(),
                        message: format!("Failed to start ltx_engine.py: {}", e),
                    });
                }
                return;
            }
        };

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
                            let line_str: String = l;
                            let mut jobs = state.jobs.lock().await;
                            if let Some(j) = jobs.iter_mut().find(|job| job.id == job_id) {
                                j.elapsed_seconds = start.elapsed().as_secs();
                                if line_str.starts_with("[STATUS]") || line_str.starts_with("[INFO]") {
                                    j.stage_info = Some(line_str.clone());
                                }
                                j.logs.push(JobLog {
                                    level: "info".to_string(),
                                    message: line_str,
                                });
                            }
                        }
                        _ => stdout_done = true,
                    }
                }
                line = stderr_reader.next_line(), if !stderr_done => {
                    match line {
                        Ok(Some(l)) => {
                            let line_str: String = l;
                            let mut jobs = state.jobs.lock().await;
                            if let Some(j) = jobs.iter_mut().find(|job| job.id == job_id) {
                                j.logs.push(JobLog {
                                    level: "warn".to_string(),
                                    message: line_str,
                                });
                            }
                        }
                        _ => stderr_done = true,
                    }
                }
            }
        }

        let status = child.wait().await;
        let elapsed = start.elapsed().as_secs();

        let mut jobs = state.jobs.lock().await;
        if let Some(j) = jobs.iter_mut().find(|job| job.id == job_id) {
            j.elapsed_seconds = elapsed;
            match status {
                Ok(s) if s.success() => {
                    j.status = "COMPLETED".to_string();
                    j.stage_info = Some("LTX-Video generation completed successfully!".to_string());
                    j.logs.push(JobLog {
                        level: "success".to_string(),
                        message: format!("🎉 LTX-Video 2.5 video generated successfully on Pod GPU in {}s!", elapsed),
                    });
                }
                Ok(s) => {
                    j.status = "FAILED".to_string();
                    j.logs.push(JobLog {
                        level: "error".to_string(),
                        message: format!("LTX-Video engine exited with code {:?}", s.code()),
                    });
                }
                Err(e) => {
                    j.status = "FAILED".to_string();
                    j.logs.push(JobLog {
                        level: "error".to_string(),
                        message: format!("Error waiting for LTX-Video process: {}", e),
                    });
                }
            }
        }
    });
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
            message: format!("Initializing Text-to-Video [Flux + {}]: '{}'", video_model.display_name(), prompt_clone),
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

    if video_model == crate::models::VideoModel::Ltx2_5 {
        spawn_local_ltx_job(
            state,
            job_id.clone(),
            payload.prompt,
            None,
            output_file,
            payload.resolution.unwrap_or_else(|| "720p".to_string()),
            payload.duration.unwrap_or(5),
        );
        return Ok(Json(serde_json::json!({ "id": job_id })));
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
                        message: format!("Video generated successfully in {}s!", elapsed),
                    });
                }
                Err(e) => {
                    j.status = "FAILED".to_string();
                    j.logs.push(JobLog {
                        level: "error".to_string(),
                        message: format!("Generation error: {:#}", e),
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
            message: format!("Initializing Image-to-Video [{}] from '{}'", video_model.display_name(), image_src),
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

    if video_model == crate::models::VideoModel::Ltx2_5 {
        spawn_local_ltx_job(
            state,
            job_id.clone(),
            payload.prompt.unwrap_or_else(|| "Cinematic smooth motion, natural camera movement, high quality".to_string()),
            Some(image_src),
            output_file,
            payload.resolution.unwrap_or_else(|| "720p".to_string()),
            payload.duration.unwrap_or(5),
        );
        return Ok(Json(serde_json::json!({ "id": job_id })));
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
                        message: format!("Animation completed successfully in {}s!", elapsed),
                    });
                }
                Err(e) => {
                    j.status = "FAILED".to_string();
                    j.logs.push(JobLog {
                        level: "error".to_string(),
                        message: format!("Error: {:#}", e),
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
            message: format!("Initializing Face Swap from '{}' onto '{}'", payload.source, payload.target),
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
                        message: format!("Face Swap completed successfully in {}s!", elapsed),
                    });
                }
                Err(e) => {
                    j.status = "FAILED".to_string();
                    j.logs.push(JobLog {
                        level: "error".to_string(),
                        message: format!("Error: {:#}", e),
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
            message: format!("Initializing Vid2Vid [{}] on '{}'", video_model.display_name(), payload.video),
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
                        message: format!("Vid2Vid transformation completed in {}s!", elapsed),
                    });
                }
                Err(e) => {
                    j.status = "FAILED".to_string();
                    j.logs.push(JobLog {
                        level: "error".to_string(),
                        message: format!("Error: {:#}", e),
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
    pub lora: Option<String>,
    pub lora_scale: Option<f32>,
    pub controlnet_image: Option<String>,
    pub controlnet_type: Option<String>,
    pub controlnet_scale: Option<f32>,
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
            message: format!("Initializing Text-to-Image (Flux): '{}'", payload.prompt),
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

        if let Some(ref lora) = payload.lora {
            stage = stage.with_lora(Some(lora.clone()), payload.lora_scale);
        }

        if payload.controlnet_image.is_some() || payload.controlnet_type.is_some() {
            stage = stage.with_controlnet(
                payload.controlnet_image.clone(),
                payload.controlnet_type.clone(),
                payload.controlnet_scale,
            );
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
                        message: format!("Image generated successfully in {}s!", elapsed),
                    });
                }
                Err(e) => {
                    j.status = "FAILED".to_string();
                    j.logs.push(JobLog {
                        level: "error".to_string(),
                        message: format!("Error: {:#}", e),
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
    pub lora: Option<String>,
    pub lora_scale: Option<f32>,
    pub controlnet_image: Option<String>,
    pub controlnet_type: Option<String>,
    pub controlnet_scale: Option<f32>,
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
                format!("Inpainting (Masked) from '{}': '{}'", payload.image, payload.prompt)
            } else {
                format!("Image-to-Image generation from '{}': '{}'", payload.image, payload.prompt)
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

        if let Some(ref lora) = payload.lora {
            stage = stage.with_lora(Some(lora.clone()), payload.lora_scale);
        }

        if payload.controlnet_image.is_some() || payload.controlnet_type.is_some() {
            stage = stage.with_controlnet(
                payload.controlnet_image.clone(),
                payload.controlnet_type.clone(),
                payload.controlnet_scale,
            );
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
                        message: format!("Image-to-Image generated successfully in {}s!", elapsed),
                    });
                }
                Err(e) => {
                    j.status = "FAILED".to_string();
                    j.logs.push(JobLog {
                        level: "error".to_string(),
                        message: format!("Error: {:#}", e),
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

#[derive(Deserialize)]
pub struct TtsPayload {
    pub text: String,
    pub engine: Option<String>,
    pub voice: Option<String>,
    pub language: Option<String>,
    pub speed: Option<f32>,
    pub speaker_wav: Option<String>,
}

async fn generate_tts(
    State(state): State<AppState>,
    Json(payload): Json<TtsPayload>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let job_id = format!("job_{}", rand::random::<u32>());
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let text_clone = payload.text.clone();
    record_prompt_auto(&state, &text_clone, "tts").await;

    let engine = payload.engine.unwrap_or_else(|| "kokoro".to_string());
    let voice = payload.voice.unwrap_or_default();
    let language = payload.language.unwrap_or_else(|| "fr".to_string());
    let speed = payload.speed.unwrap_or(1.0);
    let speaker_wav = payload.speaker_wav;

    let output_file = format!("tts_{}_{}.wav", engine, rand::random::<u16>());
    let output_path = state.workspace_dir.join(&output_file);

    let job = ServerJob {
        id: job_id.clone(),
        pipeline_type: "tts".to_string(),
        status: "RUNNING".to_string(),
        prompt: Some(text_clone.clone()),
        source: speaker_wav.clone(),
        result_file: Some(output_file.clone()),
        elapsed_seconds: 0,
        created_at: now,
        logs: vec![JobLog {
            level: "info".to_string(),
            message: format!("Initializing Text-to-Speech [{}]: '{}'", engine.to_uppercase(), text_clone),
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
    let text_for_run = text_clone.clone();

    tokio::spawn(async move {
        let start = Instant::now();
        let (api_key, base_url) = {
            let s = state_clone.settings.lock().await;
            (s.api_key.clone(), s.base_url.clone())
        };

        let client = create_client_with_tracking(state_clone.clone(), job_id_clone.clone(), api_key, base_url);
        let mut ctx = PipelineContext::new(&text_for_run);

        let tts_stage = TextToSpeechStage::new(&text_for_run)
            .with_engine(&engine)
            .with_voice(&voice)
            .with_language(&language)
            .with_speed(speed)
            .with_speaker_wav(speaker_wav)
            .with_output_path(&output_path);

        let pipeline = Pipeline::new("Text-to-Speech").add_stage(tts_stage);

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
                        message: format!("TTS audio generated successfully in {}s!", elapsed),
                    });
                }
                Err(e) => {
                    j.status = "FAILED".to_string();
                    j.logs.push(JobLog {
                        level: "error".to_string(),
                        message: format!("Error: {:#}", e),
                    });
                }
            }
        }
    });

    Ok(Json(serde_json::json!({ "id": job_id })))
}

// ----------------------------------------------------
// LoRA Dataset & Training Handlers
// ----------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetImageItem {
    pub name: String,
    pub path: String,
    pub caption: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    pub name: String,
    pub image_count: usize,
    pub images: Vec<DatasetImageItem>,
}

#[derive(Debug, Deserialize)]
pub struct AutoCaptionPayload {
    pub dataset_name: String,
    pub trigger: String,
    pub category: Option<String>,
    pub overwrite: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct SaveCaptionPayload {
    pub dataset_name: String,
    pub image_name: String,
    pub caption: String,
}

#[derive(Debug, Deserialize)]
pub struct TrainLoraPayload {
    pub dataset_name: String,
    pub output_name: String,
    pub instance_prompt: String,
    pub base_model: Option<String>,
    pub train_steps: Option<u32>,
    pub learning_rate: Option<f32>,
    pub lora_rank: Option<u32>,
    pub resolution: Option<u32>,
}

async fn list_lora_datasets(State(state): State<AppState>) -> Json<Vec<DatasetInfo>> {
    let datasets_dir = state.workspace_dir.join("datasets");
    let mut datasets = Vec::new();

    if let Ok(mut entries) = tokio::fs::read_dir(&datasets_dir).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            if path.is_dir() {
                let dataset_name = entry.file_name().to_string_lossy().to_string();
                let mut images = Vec::new();

                if let Ok(mut img_entries) = tokio::fs::read_dir(&path).await {
                    while let Ok(Some(img_entry)) = img_entries.next_entry().await {
                        let img_path = img_entry.path();
                        if img_path.is_file() {
                            let ext = img_path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
                            if matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "webp") {
                                let img_name = img_entry.file_name().to_string_lossy().to_string();
                                let txt_path = img_path.with_extension("txt");
                                let caption = match tokio::fs::read_to_string(&txt_path).await {
                                    Ok(s) => s.trim().to_string(),
                                    Err(_) => String::new(),
                                };
                                let size_bytes = img_entry.metadata().await.map(|m| m.len()).unwrap_or(0);
                                images.push(DatasetImageItem {
                                    name: img_name.clone(),
                                    path: format!("/api/lora/datasets/{}/image/{}", dataset_name, img_name),
                                    caption,
                                    size_bytes,
                                });
                            }
                        }
                    }
                }
                images.sort_by(|a, b| a.name.cmp(&b.name));
                let count = images.len();
                datasets.push(DatasetInfo {
                    name: dataset_name,
                    image_count: count,
                    images,
                });
            }
        }
    }
    datasets.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    Json(datasets)
}

async fn serve_dataset_image(
    State(state): State<AppState>,
    AxumPath((dataset_name, filename)): AxumPath<(String, String)>,
) -> Result<Response, StatusCode> {
    let sanitized_dataset = Path::new(&dataset_name)
        .file_name()
        .ok_or(StatusCode::BAD_REQUEST)?
        .to_str()
        .ok_or(StatusCode::BAD_REQUEST)?;
    let sanitized_file = Path::new(&filename)
        .file_name()
        .ok_or(StatusCode::BAD_REQUEST)?
        .to_str()
        .ok_or(StatusCode::BAD_REQUEST)?;

    let file_path = state.workspace_dir.join("datasets").join(sanitized_dataset).join(sanitized_file);
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
    if let Ok(hdr) = mime_type.parse() {
        response.headers_mut().insert(axum::http::header::CONTENT_TYPE, hdr);
    }
    Ok(response)
}

async fn upload_dataset_images(
    State(state): State<AppState>,
    AxumPath(dataset_name): AxumPath<String>,
    mut multipart: Multipart,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let sanitized_dataset = Path::new(&dataset_name)
        .file_name()
        .ok_or(StatusCode::BAD_REQUEST)?
        .to_string_lossy()
        .to_string();

    let target_dir = state.workspace_dir.join("datasets").join(&sanitized_dataset);
    tokio::fs::create_dir_all(&target_dir).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let mut uploaded_files = Vec::new();

    while let Ok(Some(field)) = multipart.next_field().await {
        if let Some(filename) = field.file_name() {
            let sanitized_name = Path::new(filename)
                .file_name()
                .ok_or(StatusCode::BAD_REQUEST)?
                .to_string_lossy()
                .to_string();

            let target_path = target_dir.join(&sanitized_name);
            let data = field.bytes().await.map_err(|_| StatusCode::BAD_REQUEST)?;
            tokio::fs::write(&target_path, data).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            uploaded_files.push(sanitized_name);
        }
    }

    Ok(Json(serde_json::json!({
        "status": "success",
        "dataset": sanitized_dataset,
        "uploaded_count": uploaded_files.len(),
        "files": uploaded_files
    })))
}

async fn autocaption_dataset(
    State(state): State<AppState>,
    Json(payload): Json<AutoCaptionPayload>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let sanitized_dataset = Path::new(&payload.dataset_name)
        .file_name()
        .ok_or(StatusCode::BAD_REQUEST)?
        .to_string_lossy()
        .to_string();

    let dataset_dir = state.workspace_dir.join("datasets").join(&sanitized_dataset);
    if !dataset_dir.exists() {
        return Err(StatusCode::NOT_FOUND);
    }

    let category = payload.category.unwrap_or_else(|| "general".to_string());
    let overwrite = payload.overwrite.unwrap_or(false);

    let mut cmd = tokio::process::Command::new(crate::utils::get_python_binary());
    crate::utils::configure_python_command(&mut cmd);
    cmd.args([
        "src/caption_engine.py",
        "--dataset-dir", dataset_dir.to_str().unwrap(),
        "--trigger", &payload.trigger,
        "--category", &category,
        "--json",
    ]);

    if overwrite {
        cmd.arg("--overwrite");
    }

    let output = cmd.output().await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    if !output.status.success() {
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    let json_str = String::from_utf8_lossy(&output.stdout);
    let val: serde_json::Value = serde_json::from_str(&json_str).unwrap_or_else(|_| {
        serde_json::json!({ "success": true, "raw": json_str.to_string() })
    });

    Ok(Json(val))
}

async fn save_dataset_caption(
    State(state): State<AppState>,
    Json(payload): Json<SaveCaptionPayload>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let sanitized_dataset = Path::new(&payload.dataset_name)
        .file_name()
        .ok_or(StatusCode::BAD_REQUEST)?
        .to_string_lossy()
        .to_string();

    let sanitized_image = Path::new(&payload.image_name)
        .file_name()
        .ok_or(StatusCode::BAD_REQUEST)?
        .to_string_lossy()
        .to_string();

    let dataset_dir = state.workspace_dir.join("datasets").join(&sanitized_dataset);
    let img_path = dataset_dir.join(&sanitized_image);
    let txt_path = img_path.with_extension("txt");

    tokio::fs::write(&txt_path, payload.caption.trim().as_bytes())
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(serde_json::json!({ "status": "success", "saved": true })))
}

async fn start_lora_training(
    State(state): State<AppState>,
    Json(payload): Json<TrainLoraPayload>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let job_id = format!("job_{}", rand::random::<u32>());
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let sanitized_dataset = Path::new(&payload.dataset_name)
        .file_name()
        .ok_or(StatusCode::BAD_REQUEST)?
        .to_string_lossy()
        .to_string();

    let dataset_path = state.workspace_dir.join("datasets").join(&sanitized_dataset);
    if !dataset_path.exists() {
        return Err(StatusCode::NOT_FOUND);
    }

    let output_name = if payload.output_name.ends_with(".safetensors") {
        payload.output_name.clone()
    } else {
        format!("{}.safetensors", payload.output_name)
    };

    let base_model = payload.base_model.unwrap_or_else(|| "runwayml/stable-diffusion-v1-5".to_string());
    let steps = payload.train_steps.unwrap_or(500);
    let lr = payload.learning_rate.unwrap_or(1e-4);
    let rank = payload.lora_rank.unwrap_or(8);
    let resolution = payload.resolution.unwrap_or(512);
    let prompt_copy = payload.instance_prompt.clone();

    let job = ServerJob {
        id: job_id.clone(),
        pipeline_type: "lora_train".to_string(),
        status: "RUNNING".to_string(),
        prompt: Some(prompt_copy.clone()),
        source: Some(sanitized_dataset.clone()),
        result_file: Some(format!("loras/{}", output_name)),
        elapsed_seconds: 0,
        created_at: now,
        logs: vec![JobLog {
            level: "info".to_string(),
            message: format!("Starting LoRA training '{}' on dataset '{}' ({} steps)", output_name, sanitized_dataset, steps),
        }],
        runpod_job_id: None,
        runpod_status: None,
        runpod_endpoint: None,
        stage_info: Some(format!("Init LoRA: 0/{} steps", steps)),
    };

    {
        let mut jobs = state.jobs.lock().await;
        jobs.insert(0, job);
    }

    let state_clone = state.clone();
    let job_id_clone = job_id.clone();
    let out_name_clone = output_name.clone();

    tokio::spawn(async move {
        let start = Instant::now();

        let mut cmd = tokio::process::Command::new(crate::utils::get_python_binary());
        crate::utils::configure_python_command(&mut cmd);
        cmd.args([
            "src/lora_train_engine.py",
            "--dataset-dir", dataset_path.to_str().unwrap(),
            "--instance-prompt", &prompt_copy,
            "--output-name", &out_name_clone,
            "--output-dir", "loras",
            "--base-model", &base_model,
            "--resolution", &resolution.to_string(),
            "--train-steps", &steps.to_string(),
            "--learning-rate", &lr.to_string(),
            "--lora-rank", &rank.to_string(),
            "--gradient-accumulation-steps", "4",
            "--mixed-precision", "fp16",
            "--device", "cuda",
        ]);

        cmd.stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                let mut jobs = state_clone.jobs.lock().await;
                if let Some(j) = jobs.iter_mut().find(|job| job.id == job_id_clone) {
                    j.status = "FAILED".to_string();
                    j.logs.push(JobLog {
                        level: "error".to_string(),
                        message: format!("Failed to start lora_train_engine.py: {}", e),
                    });
                }
                return;
            }
        };

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
                            let line_str: String = l;
                            let mut jobs = state_clone.jobs.lock().await;
                            if let Some(j) = jobs.iter_mut().find(|job| job.id == job_id_clone) {
                                j.elapsed_seconds = start.elapsed().as_secs();
                                if line_str.starts_with("[PROGRESS]") {
                                    j.stage_info = Some(line_str.clone());
                                    j.logs.push(JobLog {
                                        level: "progress".to_string(),
                                        message: line_str,
                                    });
                                } else {
                                    j.logs.push(JobLog {
                                        level: "info".to_string(),
                                        message: line_str,
                                    });
                                }
                            }
                        }
                        _ => stdout_done = true,
                    }
                }
                line = stderr_reader.next_line(), if !stderr_done => {
                    match line {
                        Ok(Some(l)) => {
                            let line_str: String = l;
                            let mut jobs = state_clone.jobs.lock().await;
                            if let Some(j) = jobs.iter_mut().find(|job| job.id == job_id_clone) {
                                j.logs.push(JobLog {
                                    level: "warn".to_string(),
                                    message: line_str,
                                });
                            }
                        }
                        _ => stderr_done = true,
                    }
                }
            }
        }

        let status = child.wait().await;
        let elapsed = start.elapsed().as_secs();

        let mut jobs = state_clone.jobs.lock().await;
        if let Some(j) = jobs.iter_mut().find(|job| job.id == job_id_clone) {
            j.elapsed_seconds = elapsed;
            match status {
                Ok(s) if s.success() => {
                    j.status = "COMPLETED".to_string();
                    j.logs.push(JobLog {
                        level: "success".to_string(),
                        message: format!("🎉 LoRA training completed successfully in {}s! File saved to loras/{}", elapsed, out_name_clone),
                    });
                }
                Ok(s) => {
                    j.status = "FAILED".to_string();
                    j.logs.push(JobLog {
                        level: "error".to_string(),
                        message: format!("Training process exited with code {:?}", s.code()),
                    });
                }
                Err(e) => {
                    j.status = "FAILED".to_string();
                    j.logs.push(JobLog {
                        level: "error".to_string(),
                        message: format!("System error waiting for process: {}", e),
                    });
                }
            }
        }
    });

    Ok(Json(serde_json::json!({ "id": job_id })))
}

