use anyhow::{Context, Result};
use base64::Engine;
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;

/// Retourne le nom ou chemin de l'exécutable Python adapté à la plateforme (Windows ou Unix).
pub fn get_python_binary() -> String {
    if let Ok(py) = std::env::var("PYTHON") {
        if !py.trim().is_empty() {
            return py;
        }
    }
    // Détection automatique de l'environnement virtuel .venv
    let venv_py = if cfg!(windows) {
        Path::new(".venv").join("Scripts").join("python.exe")
    } else {
        Path::new(".venv").join("bin").join("python")
    };
    if venv_py.exists() {
        return venv_py.to_string_lossy().to_string();
    }
    if cfg!(windows) {
        "python".to_string()
    } else {
        "python3".to_string()
    }
}

/// Configure les variables d'environnement pour l'exécution Python (caches sur disque du projet pour éviter de saturer C:).
pub fn configure_python_command(cmd: &mut tokio::process::Command) {
    let workspace = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let hf_cache = workspace.join(".hf_cache");
    let hf_hub = hf_cache.join("hub");
    let torch_cache = workspace.join(".torch_cache");

    let _ = std::fs::create_dir_all(&hf_hub);
    let _ = std::fs::create_dir_all(&torch_cache);

    cmd.env("HF_HOME", &hf_cache);
    cmd.env("HUGGINGFACE_HUB_CACHE", &hf_hub);
    cmd.env("TORCH_HOME", &torch_cache);
    cmd.env("PYTHONIOENCODING", "utf-8");
    cmd.env("PYTHONUNBUFFERED", "1");
    cmd.env("CUBLAS_WORKSPACE_CONFIG", ":4096:8");
    cmd.env("COQUI_TOS_AGREED", "1");
    cmd.env("SAFETENSORS_BACKEND", "pread");
    cmd.env("HF_HUB_DISABLE_SYMLINKS_WARNING", "1");
    if let Ok(tok) = std::env::var("HF_TOKEN") {
        if !tok.trim().is_empty() {
            cmd.env("HF_TOKEN", tok.trim());
            cmd.env("HUGGINGFACE_HUB_TOKEN", tok.trim());
        }
    }
}

/// Convertit un fichier image local en Data URI standard (`data:image/png;base64,...`).
pub fn image_to_data_uri(path: &Path) -> Result<String> {
    if !path.exists() {
        anyhow::bail!("Le fichier image '{}' n'existe pas.", path.display());
    }

    let mime = mime_guess::from_path(path)
        .first_or_octet_stream()
        .to_string();

    let bytes = std::fs::read(path)
        .with_context(|| format!("Impossible de lire le fichier image '{}'", path.display()))?;

    let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
    Ok(format!("data:{};base64,{}", mime, encoded))
}

/// Convertit un fichier vidéo local en Data URI standard (`data:video/mp4;base64,...`).
pub fn video_to_data_uri(path: &Path) -> Result<String> {
    if !path.exists() {
        anyhow::bail!("Le fichier vidéo '{}' n'existe pas.", path.display());
    }

    let mime = mime_guess::from_path(path)
        .first_or_octet_stream()
        .to_string();

    let bytes = std::fs::read(path)
        .with_context(|| format!("Impossible de lire le fichier vidéo '{}'", path.display()))?;

    let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
    Ok(format!("data:{};base64,{}", mime, encoded))
}

/// Vérifie si une chaîne est une URL HTTP(S) ou un Data URI.
pub fn is_url_or_data_uri(s: &str) -> bool {
    s.starts_with("http://")
        || s.starts_with("https://")
        || s.starts_with("data:image/")
        || s.starts_with("data:video/")
}

/// Téléverse un fichier volumineux local vers un hébergeur temporaire sécurisé pour respecter la limite de 10 Mo de RunPod.
pub async fn upload_temp_file(path: &Path) -> Result<String> {
    if !path.exists() {
        anyhow::bail!("Le fichier '{}' n'existe pas.", path.display());
    }

    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("file.mp4")
        .to_string();

    let bytes = tokio::fs::read(path)
        .await
        .with_context(|| format!("Impossible de lire le fichier '{}'", path.display()))?;

    let part = reqwest::multipart::Part::bytes(bytes).file_name(file_name);

    let form = reqwest::multipart::Form::new()
        .text("reqtype", "fileupload")
        .text("time", "1h")
        .part("fileToUpload", part);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()?;

    let resp = client
        .post("https://litterbox.catbox.moe/resources/internals/api.php")
        .multipart(form)
        .send()
        .await
        .with_context(|| "Erreur lors de l'upload temporaire du fichier vers l'hébergeur")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("Échec de l'hébergement temporaire (HTTP {}): {}", status, text);
    }

    let url = resp.text().await?.trim().to_string();
    if url.starts_with("http://") || url.starts_with("https://") {
        Ok(url)
    } else {
        anyhow::bail!("Réponse inattendue lors de l'upload temporaire : {}", url);
    }
}

/// Résout une entrée d'image : si c'est un chemin local, il est converti en Data URI (ou hébergé si > 5 Mo) ; si c'est déjà une URL, il est renvoyé tel quel.
pub async fn resolve_image_input(input: &str) -> Result<String> {
    if is_url_or_data_uri(input) {
        Ok(input.to_string())
    } else {
        let path = Path::new(input);
        let file_size = if let Ok(meta) = tokio::fs::metadata(path).await {
            meta.len()
        } else {
            0
        };

        if file_size > 5 * 1024 * 1024 {
            println!("  Image volumineuse ({:.1} Mo > 5 Mo) : hébergement temporaire pour RunPod...", file_size as f64 / 1_048_576.0);
            upload_temp_file(path).await
        } else {
            image_to_data_uri(path)
        }
    }
}

/// Résout une entrée de vidéo : si c'est un chemin local, il est converti en Data URI (ou hébergé si > 5 Mo) ; si c'est déjà une URL, il est renvoyé tel quel.
pub async fn resolve_video_input(input: &str) -> Result<String> {
    if is_url_or_data_uri(input) {
        Ok(input.to_string())
    } else {
        let path = Path::new(input);
        let file_size = if let Ok(meta) = tokio::fs::metadata(path).await {
            meta.len()
        } else {
            0
        };

        if file_size > 5 * 1024 * 1024 {
            println!("  Vidéo volumineuse ({:.1} Mo > 5 Mo) : hébergement temporaire pour RunPod...", file_size as f64 / 1_048_576.0);
            upload_temp_file(path).await
        } else {
            video_to_data_uri(path)
        }
    }
}

/// Résout une entrée média (image ou vidéo) : renvoie le Data URI ou l'URL ainsi qu'un booléen `is_video`.
pub async fn resolve_media_input(input: &str) -> Result<(String, bool)> {
    if input.starts_with("data:video/") {
        return Ok((input.to_string(), true));
    }
    if input.starts_with("data:image/") {
        return Ok((input.to_string(), false));
    }

    if input.starts_with("http://") || input.starts_with("https://") {
        let is_video = input.ends_with(".mp4")
            || input.ends_with(".webm")
            || input.ends_with(".mov")
            || input.ends_with(".mkv");
        return Ok((input.to_string(), is_video));
    }

    let path = Path::new(input);
    let mime = mime_guess::from_path(path).first_or_octet_stream().to_string();
    let is_video = mime.starts_with("video/");

    let file_size = if let Ok(meta) = tokio::fs::metadata(path).await {
        meta.len()
    } else {
        0
    };

    if file_size > 5 * 1024 * 1024 {
        println!("  Fichier volumineux ({:.1} Mo > 5 Mo) : hébergement temporaire pour RunPod...", file_size as f64 / 1_048_576.0);
        let url = upload_temp_file(path).await?;
        Ok((url, is_video))
    } else if is_video {
        Ok((video_to_data_uri(path)?, true))
    } else {
        Ok((image_to_data_uri(path)?, false))
    }
}

/// Télécharge un fichier depuis une URL de manière asynchrone avec suivi de progression visuel.
pub async fn download_file(url: &str, output_path: &Path, show_progress: bool) -> Result<()> {
    if let Some(parent) = output_path.parent() {
        if !parent.exists() {
            tokio::fs::create_dir_all(parent).await.with_context(|| {
                format!(
                    "Impossible de créer le dossier parent '{}'",
                    parent.display()
                )
            })?;
        }
    }

    if !url.starts_with("http://") && !url.starts_with("https://") && !url.starts_with("data:") {
        let src = Path::new(url);
        if src.exists() {
            if src != output_path {
                tokio::fs::copy(src, output_path)
                    .await
                    .with_context(|| format!("Impossible de copier le fichier de '{}' vers '{}'", src.display(), output_path.display()))?;
            }
            return Ok(());
        }
    }

    let client = reqwest::Client::new();
    let response = client
        .get(url)
        .send()
        .await
        .with_context(|| format!("Échec de la requête HTTP vers {}", url))?;

    if !response.status().is_success() {
        anyhow::bail!(
            "Erreur de téléchargement HTTP {} pour l'URL {}",
            response.status(),
            url
        );
    }

    let total_size = response.content_length();

    let pb = if show_progress {
        let pb = match total_size {
            Some(size) => {
                let pb = ProgressBar::new(size);
                pb.set_style(
                    ProgressStyle::default_bar()
                        .template("  [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                        .unwrap()
                        .progress_chars("#>-"),
                );
                pb
            }
            None => {
                let pb = ProgressBar::new_spinner();
                pb.set_style(
                    ProgressStyle::default_spinner()
                        .template("  [{elapsed_precise}] {bytes} téléchargés...")
                        .unwrap(),
                );
                pb
            }
        };
        Some(pb)
    } else {
        None
    };

    let mut file = tokio::fs::File::create(output_path)
        .await
        .with_context(|| format!("Impossible de créer le fichier '{}'", output_path.display()))?;

    let mut stream = response.bytes_stream();
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.with_context(|| "Erreur lors de la lecture du flux HTTP")?;
        file.write_all(&chunk)
            .await
            .with_context(|| "Erreur lors de l'écriture sur le disque")?;
        if let Some(ref pb) = pb {
            pb.inc(chunk.len() as u64);
        }
    }

    file.flush()
        .await
        .with_context(|| "Erreur lors de la finalisation du fichier")?;

    if let Some(pb) = pb {
        pb.finish_and_clear();
    }

    Ok(())
}

/// Si un fichier avec le même chemin existe déjà, incrémente le nom de fichier (ex: `video_1.mp4`, `video_2.mp4`) pour éviter tout écrasement.
pub fn get_unique_incremental_path(path: &Path) -> PathBuf {
    if !path.exists() {
        return path.to_path_buf();
    }

    let parent = path.parent().unwrap_or_else(|| Path::new(""));
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|e| format!(".{}", e))
        .unwrap_or_default();

    let mut counter = 1;
    loop {
        let candidate_name = format!("{}_{}{}", stem, counter, ext);
        let candidate_path = parent.join(candidate_name);
        if !candidate_path.exists() {
            return candidate_path;
        }
        counter += 1;
    }
}

/// Ouvre un fichier dans l'application par défaut du système (visionneuse d'image, lecteur vidéo, etc.).
pub fn open_file_preview(path: &Path) -> Result<()> {
    if path.exists() {
        open::that_detached(path).with_context(|| {
            format!("Impossible d'ouvrir le fichier '{}'", path.display())
        })?;
    }
    Ok(())
}

/// Récupère l'adresse IP locale (LAN / Wi-Fi) de la machine pour l'accès mobile.
pub fn get_local_lan_ip() -> Option<String> {
    let socket = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("8.8.8.8:80").ok()?;
    socket.local_addr().ok().map(|addr| addr.ip().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_is_url_or_data_uri() {
        assert!(is_url_or_data_uri("http://example.com/test.png"));
        assert!(is_url_or_data_uri("https://runpod.io/video.mp4"));
        assert!(is_url_or_data_uri("data:image/png;base64,iVBORw0KGgo="));
        assert!(is_url_or_data_uri("data:video/mp4;base64,AAAA"));
        assert!(!is_url_or_data_uri("./local_image.png"));
        assert!(!is_url_or_data_uri("/path/to/image.jpg"));
    }

    #[test]
    fn test_image_to_data_uri() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.png");
        {
            let mut f = std::fs::File::create(&file_path).unwrap();
            f.write_all(b"fake png content").unwrap();
        }

        let res = image_to_data_uri(&file_path).unwrap();
        assert!(res.starts_with("data:image/png;base64,"));
    }

    #[test]
    fn test_video_to_data_uri() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.mp4");
        {
            let mut f = std::fs::File::create(&file_path).unwrap();
            f.write_all(b"fake mp4 content").unwrap();
        }

        let res = video_to_data_uri(&file_path).unwrap();
        assert!(res.starts_with("data:video/mp4;base64,"));
    }

    #[test]
    fn test_get_unique_incremental_path() {
        let dir = tempfile::tempdir().unwrap();
        let file1 = dir.path().join("video.mp4");
        std::fs::File::create(&file1).unwrap();

        let unique1 = get_unique_incremental_path(&file1);
        assert_eq!(unique1, dir.path().join("video_1.mp4"));

        std::fs::File::create(&unique1).unwrap();
        let unique2 = get_unique_incremental_path(&file1);
        assert_eq!(unique2, dir.path().join("video_2.mp4"));
    }
}

