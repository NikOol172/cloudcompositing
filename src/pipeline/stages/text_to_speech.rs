use crate::client::RunpodClient;
use crate::pipeline::{PipelineContext, Stage};
use anyhow::{Context, Result};
use async_trait::async_trait;
use console::style;
use std::path::PathBuf;
use tokio::io::AsyncBufReadExt;

/// Étape de Synthèse Vocale (Text-to-Speech) avec support de Kokoro-82M et XTTS-v2.
pub struct TextToSpeechStage {
    pub text: String,
    pub engine: String,
    pub voice: String,
    pub language: String,
    pub speed: f32,
    pub speaker_wav: Option<String>,
    pub output_path: Option<PathBuf>,
}

impl Default for TextToSpeechStage {
    fn default() -> Self {
        Self {
            text: String::new(),
            engine: "kokoro".to_string(),
            voice: String::new(),
            language: "fr".to_string(),
            speed: 1.0,
            speaker_wav: None,
            output_path: None,
        }
    }
}

impl TextToSpeechStage {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            ..Default::default()
        }
    }

    pub fn with_engine(mut self, engine: impl Into<String>) -> Self {
        self.engine = engine.into();
        self
    }

    pub fn with_voice(mut self, voice: impl Into<String>) -> Self {
        self.voice = voice.into();
        self
    }

    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = language.into();
        self
    }

    pub fn with_speed(mut self, speed: f32) -> Self {
        self.speed = speed;
        self
    }

    pub fn with_speaker_wav(mut self, speaker_wav: Option<String>) -> Self {
        self.speaker_wav = speaker_wav;
        self
    }

    pub fn with_output_path(mut self, output: impl Into<PathBuf>) -> Self {
        self.output_path = Some(output.into());
        self
    }
}

#[async_trait]
impl Stage for TextToSpeechStage {
    fn name(&self) -> &str {
        "Synthèse Vocale (Text-to-Speech)"
    }

    async fn execute(&self, ctx: &mut PipelineContext, _client: &RunpodClient) -> Result<()> {
        let effective_text = if !self.text.trim().is_empty() {
            self.text.as_str()
        } else {
            ctx.effective_prompt()
        };

        if effective_text.trim().is_empty() {
            anyhow::bail!("Aucun texte fourni pour la synthèse vocale.");
        }

        let final_output = if let Some(ref out) = self.output_path {
            out.clone()
        } else {
            std::env::temp_dir().join(format!("tts_{}.wav", rand::random::<u32>()))
        };

        println!(
            "  Paramètres TTS : Moteur: {}, Langue: {}, Voix/Ref: {}, Vitesse: {}x",
            style(&self.engine.to_uppercase()).cyan().bold(),
            style(&self.language).green(),
            style(if let Some(ref spk) = self.speaker_wav { spk.as_str() } else if !self.voice.is_empty() { self.voice.as_str() } else { "Défaut" }).yellow(),
            self.speed
        );
        println!("  Lancement du moteur de synthèse vocale local...");

        let mut cmd = tokio::process::Command::new(crate::utils::get_python_binary());
        crate::utils::configure_python_command(&mut cmd);
        cmd.args([
            "src/tts_engine.py",
            "--text", effective_text,
            "--engine", &self.engine,
            "--language", &self.language,
            "--speed", &self.speed.to_string(),
            "--output", final_output.to_str().unwrap(),
        ]);

        if !self.voice.is_empty() {
            cmd.args(["--voice", &self.voice]);
        }

        if let Some(ref spk) = self.speaker_wav {
            cmd.args(["--speaker-wav", spk]);
        }

        cmd.stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        let mut child = cmd
            .spawn()
            .context("Impossible de démarrer le script tts_engine.py")?;

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
                        Ok(Some(l)) => println!("  {}", l),
                        _ => stdout_done = true,
                    }
                }
                line = stderr_reader.next_line(), if !stderr_done => {
                    match line {
                        Ok(Some(l)) => eprintln!("  {}", style(l).red()),
                        _ => stderr_done = true,
                    }
                }
            }
        }

        let status = child.wait().await?;
        if !status.success() {
            anyhow::bail!("Le script tts_engine.py s'est terminé avec une erreur (code {:?})", status.code());
        }

        if !final_output.exists() {
            anyhow::bail!("Le fichier audio généré n'a pas été trouvé à l'emplacement : {}", final_output.display());
        }

        ctx.downloaded_audio_path = Some(final_output.clone());
        ctx.set_meta("audio_path", final_output.to_string_lossy().to_string());

        println!("  Audio sauvegardé : {}", style(final_output.display()).green().bold());
        Ok(())
    }
}
