#!/usr/bin/env python3
"""
High-Performance Local Text-to-Speech (TTS) Engine supporting Kokoro-82M & Coqui XTTS-v2.
Optimized for CUDA / Pod GPUs and CPU inference.
"""

import os
import sys
import site
import argparse
import time
import functools

# Always unbuffer print outputs for real-time streaming in server/CLI
print = functools.partial(print, flush=True)

# Force UTF-8 encoding on Windows to prevent UnicodeEncodeError with emojis/accents
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Configure caches on Drive D: to avoid filling system drive C:
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HOME"] = os.path.join(base_dir, ".hf_cache")
os.environ["HF_HUB_CACHE"] = os.path.join(base_dir, ".hf_cache", "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(base_dir, ".hf_cache", "hub")
os.environ["TORCH_HOME"] = os.path.join(base_dir, ".torch_cache")
os.environ["COQUI_MODEL_HOME"] = os.path.join(base_dir, ".hf_cache", "coqui")
# Automatically accept non-commercial CPML terms to avoid interactive prompt [y/n]
os.environ["COQUI_TOS_AGREED"] = "1"

# Ensure user site-packages and standard paths are in sys.path
for p in [
    site.getusersitepackages(),
    os.path.expanduser("~/.local/lib/python3.10/site-packages"),
    os.path.expanduser("~/.local/lib/python3.11/site-packages"),
    os.path.expanduser("~/.local/lib/python3.12/site-packages"),
    "/usr/local/lib/python3.10/dist-packages",
    "/usr/lib/python3/dist-packages",
    "/usr/lib/python3.10",
]:
    if p and os.path.exists(p) and p not in sys.path:
        sys.path.append(p)

def parse_args():
    parser = argparse.ArgumentParser(description="Local Text-to-Speech Engine (Kokoro-82M & XTTS-v2)")
    parser.add_argument("--text", required=True, help="Text to synthesize into speech")
    parser.add_argument("--output", required=True, help="Output destination audio file (.wav)")
    parser.add_argument(
        "--engine",
        choices=["kokoro", "xtts"],
        default="kokoro",
        help="TTS engine: 'kokoro' (Kokoro-82M, ultra fast) or 'xtts' (Coqui XTTS-v2, voice cloning)",
    )
    parser.add_argument(
        "--voice",
        default="",
        help="Voice identifier (e.g. ff_siwis, af_bella, af_sarah, am_adam, bf_emma for Kokoro, or speaker name for XTTS)",
    )
    parser.add_argument(
        "--speaker-wav",
        default="",
        help="Reference speaker audio file (.wav/.mp3) for zero-shot voice cloning with XTTS-v2",
    )
    parser.add_argument(
        "--language",
        default="fr",
        help="Language code: 'fr' (French), 'en' (English), 'es' (Spanish), 'de' (German), 'it' (Italian), etc.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech rate speed factor (0.5 to 2.0, default: 1.0)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Execution device ('auto', 'cuda', 'cpu')",
    )
    return parser.parse_args()

def resolve_device(device_arg: str):
    if device_arg in ["cuda", "cpu"]:
        return device_arg
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

def run_kokoro(text: str, output: str, voice: str, language: str, speed: float, device: str):
    print(f"[*] Initialisation du moteur Kokoro-82M (Device: {device})...")
    start_time = time.time()

    try:
        import soundfile as sf
        import numpy as np
    except ImportError as e:
        print(f"[ERREUR] Dépendance audio manquante: {e}")
        print("Veuillez installer: pip install soundfile numpy scipy")
        sys.exit(1)

    try:
        from kokoro import KPipeline
    except ImportError:
        print("[ERREUR] Le package 'kokoro' n'est pas encore installé.")
        print("Veuillez exécuter: pip install kokoro soundfile scipy")
        sys.exit(1)

    # Lang code mapping for Kokoro:
    # 'a' => American English, 'b' => British English, 'f' => French, 'e' => Spanish, 'i' => Italian, 'p' => Portuguese
    lang_lower = language.lower().strip()
    if lang_lower in ["fr", "fra", "french"]:
        lang_code = "f"
        default_voice = "ff_siwis"
    elif lang_lower in ["en-gb", "gb", "british"]:
        lang_code = "b"
        default_voice = "bf_emma"
    elif lang_lower in ["es", "spa", "spanish"]:
        lang_code = "e"
        default_voice = "ef_dora"
    elif lang_lower in ["it", "ita", "italian"]:
        lang_code = "i"
        default_voice = "if_sara"
    elif lang_lower in ["pt", "por", "portuguese"]:
        lang_code = "p"
        default_voice = "pf_dora"
    elif lang_lower in ["ja", "jp", "japanese"]:
        lang_code = "j"
        default_voice = "jf_alpha"
    elif lang_lower in ["zh", "chi", "chinese"]:
        lang_code = "z"
        default_voice = "zf_xiaobei"
    else: # default American English
        lang_code = "a"
        default_voice = "af_bella"

    chosen_voice = voice if voice and voice.strip() else default_voice

    print(f"[*] Langue: '{lang_lower}' (Code: '{lang_code}'), Voix: '{chosen_voice}', Vitesse: {speed}x")

    pipeline = KPipeline(lang_code=lang_code)

    generator = pipeline(
        text,
        voice=chosen_voice,
        speed=speed,
        split_pattern=r"\n+",
    )

    all_audio = []
    sample_rate = 24000

    for i, (gs, ps, audio) in enumerate(generator):
        if audio is not None and len(audio) > 0:
            all_audio.append(audio)

    if not all_audio:
        raise RuntimeError("Kokoro n'a produit aucun segment audio pour le texte fourni.")

    concatenated = np.concatenate(all_audio)
    
    # Ensure directory exists
    out_dir = os.path.dirname(os.path.abspath(output))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    sf.write(output, concatenated, sample_rate)
    duration = len(concatenated) / sample_rate
    elapsed = time.time() - start_time
    print(f"[+] Audio généré avec succès en {elapsed:.2f}s ! Durée: {duration:.2f}s -> {output}")

def run_xtts(text: str, output: str, voice: str, speaker_wav: str, language: str, speed: float, device: str):
    print(f"[*] Initialisation du moteur Coqui XTTS-v2 (Device: {device})...")
    start_time = time.time()

    try:
        from TTS.api import TTS
    except ImportError:
        print("[ERREUR] Le package 'coqui-tts' n'est pas installé.")
        print("Veuillez installer Coqui TTS via: pip install coqui-tts torchaudio")
        sys.exit(1)

    import torch

    lang_lower = language.lower().strip()
    # Normalize common language codes for XTTS
    lang_map = {
        "fr": "fr", "french": "fr",
        "en": "en", "english": "en", "en-us": "en", "en-gb": "en",
        "es": "es", "spanish": "es",
        "de": "de", "german": "de",
        "it": "it", "italian": "it",
        "pt": "pt", "portuguese": "pt",
        "pl": "pl", "polish": "pl",
        "tr": "tr", "turkish": "tr",
        "ru": "ru", "russian": "ru",
        "nl": "nl", "dutch": "nl",
        "cs": "cs", "czech": "cs",
        "ar": "ar", "arabic": "ar",
        "zh": "zh-cn", "chinese": "zh-cn",
        "ja": "ja", "japanese": "ja",
        "ko": "ko", "korean": "ko",
        "hu": "hu", "hungarian": "hu",
        "hi": "hi", "hindi": "hi",
    }
    xtts_lang = lang_map.get(lang_lower, "fr")

    # Load XTTS-v2 model
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    if device == "cuda" and torch.cuda.is_available():
        tts.to("cuda")

    # Ensure target output directory exists
    out_dir = os.path.dirname(os.path.abspath(output))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if speaker_wav and os.path.exists(speaker_wav):
        print(f"[*] Clonage vocal par référence audio : '{speaker_wav}'")
        print(f"[*] Langue: '{xtts_lang}', Vitesse: {speed}x")
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language=xtts_lang,
            file_path=output,
            speed=speed,
        )
    else:
        # Fallback to predefined speaker if available or first available speaker
        speaker_name = voice if voice and voice.strip() else None
        if not speaker_name and hasattr(tts, "speakers") and tts.speakers:
            speaker_name = tts.speakers[0]
            print(f"[*] Aucune référence audio fournie, utilisation de la voix intégrée: '{speaker_name}'")
        elif speaker_name:
            print(f"[*] Utilisation de la voix intégrée : '{speaker_name}'")
        else:
            raise ValueError(
                "Pour XTTS-v2, veuillez spécifier soit un fichier audio de référence (--speaker-wav), "
                "soit une voix intégrée (--voice)."
            )

        tts.tts_to_file(
            text=text,
            speaker=speaker_name,
            language=xtts_lang,
            file_path=output,
            speed=speed,
        )

    elapsed = time.time() - start_time
    print(f"[+] Audio généré avec succès en {elapsed:.2f}s ! -> {output}")

def main():
    args = parse_args()
    device = resolve_device(args.device)

    print(f"==================================================")
    print(f"🎙️ Moteur TTS RunPod Studio")
    print(f" • Moteur    : {args.engine.upper()}")
    print(f" • Langue    : {args.language}")
    print(f" • Voix/Ref  : {args.voice or args.speaker_wav or 'Défaut'}")
    print(f" • Sortie    : {args.output}")
    print(f" • Device    : {device.upper()}")
    print(f"==================================================")

    if args.engine == "kokoro":
        run_kokoro(
            text=args.text,
            output=args.output,
            voice=args.voice,
            language=args.language,
            speed=args.speed,
            device=device,
        )
    elif args.engine == "xtts":
        run_xtts(
            text=args.text,
            output=args.output,
            voice=args.voice,
            speaker_wav=args.speaker_wav,
            language=args.language,
            speed=args.speed,
            device=device,
        )
    else:
        print(f"[ERREUR] Moteur non reconnu: {args.engine}")
        sys.exit(1)

if __name__ == "__main__":
    main()
