mod cli;
mod client;
mod models;
mod pipeline;
mod server;
mod utils;

use clap::Parser;
use cli::{Cli, Commands};
use client::RunpodClient;
use console::style;
use pipeline::stages::{
    DownloadStage, FaceSwapStage, ImageToImageStage, ImageToVideoStage, InteractiveReviewStage, PromptEnhanceStage,
    TextToImageStage, TextToSpeechStage, VideoToVideoStage,
};
use pipeline::{Pipeline, PipelineContext};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Chargement automatique d'un fichier .env s'il existe
    let _ = dotenvy::dotenv();

    let cli = Cli::parse();

    // Configuration des logs
    let filter = if cli.verbose {
        "runpod_pipeline=debug,info"
    } else {
        "runpod_pipeline=info,warn,error"
    };

    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(filter))
        .with(tracing_subscriber::fmt::layer().without_time().with_target(false))
        .init();

    // Si la commande est 'serve', lancer directement le serveur web
    if let Commands::Serve(args) = cli.command {
        return server::start_server(args.port, args.open).await;
    }

    // Synthèse vocale locale (ne requiert pas de clé RunPod)
    if let Commands::Tts(args) = cli.command {
        println!(
            "{}",
            style(format!("=== Synthèse Vocale (Text-to-Speech) [{}] ===", args.engine.to_uppercase()))
                .bold()
                .magenta()
        );

        let dummy_client = RunpodClient::with_base_url("local", cli.base_url);
        let mut ctx = PipelineContext::new(&args.text);

        let stage = TextToSpeechStage::new(args.text)
            .with_engine(args.engine)
            .with_voice(args.voice)
            .with_language(args.language)
            .with_speed(args.speed)
            .with_speaker_wav(args.speaker_wav)
            .with_output_path(args.output);

        let pipeline = Pipeline::new("Text-to-Speech").add_stage(stage);
        return pipeline.run(&mut ctx, &dummy_client).await;
    }

    // Entraînement LoRA local (ne requiert pas de clé RunPod)
    if let Commands::TrainLora(args) = cli.command {
        println!(
            "{}",
            style(format!("=== Entraînement LoRA Fine-Tuning [{}] ===", args.output))
                .bold()
                .magenta()
        );

        let dummy_client = RunpodClient::with_base_url("local", cli.base_url);
        let mut ctx = PipelineContext::new(&args.prompt);

        let stage = pipeline::stages::LoraTrainingStage::new(args.dataset, args.output)
            .with_instance_prompt(args.prompt)
            .with_base_model(args.base_model)
            .with_train_steps(args.steps)
            .with_learning_rate(args.lr)
            .with_lora_rank(args.rank)
            .with_resolution(args.resolution);

        let pipeline = Pipeline::new("LoRA Training").add_stage(stage);
        return pipeline.run(&mut ctx, &dummy_client).await;
    }

    // Récupération de la clé API
    let api_key = match cli.api_key.or_else(|| std::env::var("RUNPOD_API_KEY").ok()) {
        Some(k) if !k.trim().is_empty() => k.trim().to_string(),
        _ => {
            eprintln!(
                "{}",
                style("Erreur : La clé API RunPod n'a pas été trouvée.").red().bold()
            );
            eprintln!(
                "Veuillez définir la variable d'environnement {} ou spécifier {}",
                style("RUNPOD_API_KEY").cyan(),
                style("--api-key <VOTRE_CLE>").cyan()
            );
            std::process::exit(1);
        }
    };

    let client = RunpodClient::with_base_url(api_key, cli.base_url);

    match cli.command {
        Commands::Serve(_) => unreachable!(),
        Commands::Tts(_) => unreachable!(),
        Commands::TrainLora(_) => unreachable!(),
        Commands::Img2Vid(args) => {
            let video_model = std::str::FromStr::from_str(&args.model).unwrap_or(models::VideoModel::Wan2_5);
            println!(
                "{}",
                style(format!("=== Pipeline Image-to-Video [{}] ===", video_model.display_name())).bold().magenta()
            );

            let mut stage = ImageToVideoStage::new()
                .with_model(video_model)
                .with_image(args.image)
                .with_duration(args.duration)
                .with_resolution(args.resolution)
                .with_prompt(args.prompt);

            if let Some(ep) = args.endpoint {
                stage = stage.with_endpoint(ep);
            }

            let mut ctx = PipelineContext::default();
            let pipeline = Pipeline::new("Image-to-Video")
                .add_stage(stage)
                .add_stage(DownloadStage::new().download_video(args.output.clone()));

            pipeline.run(&mut ctx, &client).await?;

            if let Some(ref path) = ctx.downloaded_video_path {
                println!(
                    "{} Vidéo générée et sauvegardée à : {}",
                    style("🎉").bold(),
                    style(path.display()).cyan().bold()
                );
            }
        }

        Commands::Txt2Vid(args) => {
            let video_model = std::str::FromStr::from_str(&args.model).unwrap_or(models::VideoModel::Wan2_5);
            println!(
                "{}",
                style(format!("=== Pipeline Complet Text-to-Video [Flux + {}] ===", video_model.display_name())).bold().magenta()
            );

            let mut ctx = PipelineContext::new(args.prompt.clone());
            let mut pipeline = Pipeline::new("Text-to-Video");

            if args.enhance_prompt {
                pipeline = pipeline.add_stage(PromptEnhanceStage::new());
            }

            let mut t2i_stage = TextToImageStage::new()
                .with_endpoint(args.flux_endpoint)
                .with_dimensions(args.width, args.height)
                .with_steps(args.steps)
                .with_local(args.local);

            if let Some(ref m) = args.local_model {
                t2i_stage = t2i_stage.with_local_model(m.clone());
            }

            pipeline = pipeline
                .add_stage(t2i_stage)
                .add_stage(DownloadStage::new().download_image(args.image_output.clone()));

            if args.interactive {
                pipeline = pipeline.add_stage(InteractiveReviewStage::new().with_auto_open(true));
            }

            let mut i2v_stage = ImageToVideoStage::new()
                .with_model(video_model)
                .with_duration(args.duration)
                .with_resolution(args.resolution);

            if let Some(ep) = args.video_endpoint {
                i2v_stage = i2v_stage.with_endpoint(ep);
            } else if args.wan_endpoint != "wan-2-5" {
                i2v_stage = i2v_stage.with_endpoint(args.wan_endpoint);
            }

            pipeline = pipeline
                .add_stage(i2v_stage)
                .add_stage(DownloadStage::new().download_video(args.video_output.clone()));

            pipeline.run(&mut ctx, &client).await?;

            println!("{}", style("🎉 Pipeline terminé avec succès !").green().bold());
            if let Some(ref p) = ctx.downloaded_image_path {
                println!("  • Image intermédiaire : {}", style(p.display()).cyan());
            }
            if let Some(ref p) = ctx.downloaded_video_path {
                println!("  • Vidéo finale : {}", style(p.display()).cyan().bold());
            }
        }

        Commands::Txt2Img(args) => {
            println!(
                "{}",
                style("=== Pipeline Text-to-Image ===").bold().magenta()
            );

            let mut ctx = PipelineContext::new(args.prompt.clone());
            let mut pipeline = Pipeline::new("Text-to-Image");

            if args.enhance_prompt {
                pipeline = pipeline.add_stage(PromptEnhanceStage::new());
            }

            let mut stage = TextToImageStage::new()
                .with_endpoint(args.endpoint)
                .with_dimensions(args.width, args.height)
                .with_steps(args.steps)
                .with_local(args.local);

            if let Some(ref m) = args.local_model {
                stage = stage.with_local_model(m.clone());
            }

            if let Some(seed) = args.seed {
                stage = stage.with_seed(seed);
            }

            pipeline = pipeline
                .add_stage(stage)
                .add_stage(DownloadStage::new().download_image(args.output.clone()));

            pipeline.run(&mut ctx, &client).await?;

            if let Some(ref path) = ctx.downloaded_image_path {
                println!(
                    "{} Image générée et sauvegardée à : {}",
                    style("🎉").bold(),
                    style(path.display()).cyan().bold()
                );
            }
        }

        Commands::Img2Img(args) => {
            println!(
                "{}",
                style("=== Pipeline Image-to-Image ===").bold().magenta()
            );

            let mut ctx = PipelineContext::new(args.prompt.clone());
            let mut pipeline = Pipeline::new("Image-to-Image");

            let mut stage = ImageToImageStage::new()
                .with_endpoint(args.endpoint)
                .with_image(args.image)
                .with_prompt(args.prompt)
                .with_strength(args.strength)
                .with_steps(args.steps)
                .with_local(args.local);

            if args.width > 0 && args.height > 0 {
                stage = stage.with_dimensions(args.width, args.height);
            }

            if let Some(ref m) = args.local_model {
                stage = stage.with_local_model(m.clone());
            }

            if let Some(mask) = args.mask {
                stage = stage.with_mask(mask);
            }

            if let Some(seed) = args.seed {
                stage = stage.with_seed(seed);
            }

            pipeline = pipeline
                .add_stage(stage)
                .add_stage(DownloadStage::new().download_image(args.output.clone()));

            pipeline.run(&mut ctx, &client).await?;

            if let Some(ref path) = ctx.downloaded_image_path {
                println!(
                    "{} Image transformée et sauvegardée à : {}",
                    style("🎉").bold(),
                    style(path.display()).cyan().bold()
                );
            }
        }

        Commands::Vid2Vid(args) => {
            let video_model = std::str::FromStr::from_str(&args.model).unwrap_or(models::VideoModel::Wan2_5);
            println!(
                "{}",
                style(format!("=== Pipeline Video-to-Video [{}] ===", video_model.display_name())).bold().magenta()
            );

            let mut ctx = PipelineContext::new(args.prompt.clone());
            let mut stage = VideoToVideoStage::new()
                .with_model(video_model)
                .with_video(args.video)
                .with_duration(args.duration)
                .with_strength(args.strength)
                .with_resolution(args.resolution)
                .with_prompt(args.prompt);

            if let Some(ep) = args.endpoint {
                stage = stage.with_endpoint(ep);
            }
            if let Some(seed) = args.seed {
                stage = stage.with_seed(seed);
            }
            if let Some(neg) = args.negative_prompt {
                stage = stage.with_negative_prompt(neg);
            }

            let pipeline = Pipeline::new("Video-to-Video")
                .add_stage(stage)
                .add_stage(DownloadStage::new().download_video(args.output.clone()));

            pipeline.run(&mut ctx, &client).await?;

            if let Some(ref path) = ctx.downloaded_video_path {
                println!(
                    "{} Vidéo transformée et sauvegardée à : {}",
                    style("🎉").bold(),
                    style(path.display()).cyan().bold()
                );
            }
        }

        Commands::FaceSwap(args) => {
            println!(
                "{}",
                style("=== Pipeline Face Swap (Remplacement de Visage) ===").bold().magenta()
            );

            let is_video_target = args.target.ends_with(".mp4")
                || args.target.ends_with(".webm")
                || args.target.ends_with(".mov")
                || args.target.ends_with(".mkv");

            let output_path = args.output.unwrap_or_else(|| {
                if is_video_target {
                    std::path::PathBuf::from("output_faceswap.mp4")
                } else {
                    std::path::PathBuf::from("output_faceswap.png")
                }
            });

            let mut ctx = PipelineContext::default();
            let stage = FaceSwapStage::new(args.source, args.target)
                .with_endpoint(args.endpoint)
                .with_restore_face(args.restore_face)
                .with_face_index(args.face_index);

            let download_stage = if is_video_target {
                DownloadStage::new().download_video(output_path.clone())
            } else {
                DownloadStage::new().download_image(output_path.clone())
            };

            let pipeline = Pipeline::new("Face-Swap")
                .add_stage(stage)
                .add_stage(download_stage);

            pipeline.run(&mut ctx, &client).await?;

            println!("{}", style("🎉 Face Swap terminé avec succès !").green().bold());
            if let Some(ref path) = ctx.downloaded_video_path.or(ctx.downloaded_image_path) {
                println!("  • Résultat sauvegardé : {}", style(path.display()).cyan().bold());
            }
        }

        Commands::Enhance(args) => {
            println!(
                "{}",
                style("=== Enrichissement de Prompt (LLM) ===").bold().magenta()
            );

            let mut ctx = PipelineContext::new(args.prompt.clone());
            let stage = PromptEnhanceStage::new().with_endpoint(args.endpoint, args.model);
            let pipeline = Pipeline::new("Prompt-Enhance").add_stage(stage);

            pipeline.run(&mut ctx, &client).await?;

            if let Some(ref enhanced) = ctx.enhanced_prompt {
                println!("\n{}", style("Résultat final :").bold());
                println!("{}", style(enhanced).green().bold());
            }
        }
    }

    Ok(())
}
