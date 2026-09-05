#!/usr/bin/env python3
"""
Captioning Engine for LoRA Fine-Tuning Datasets.
Generates structured, high-quality descriptive prompts for training images
using trigger tokens, category-specific tags, and context.
"""

import os
import sys
import argparse
import json
import random
from pathlib import Path
from PIL import Image

PERSON_TEMPLATES = [
    "a high quality photo of {trigger}, looking directly at camera, portrait lighting, detailed face, 8k uhd",
    "a close-up portrait of {trigger}, clear facial features, studio lighting, natural skin texture",
    "a photo of {trigger}, three-quarter view, depth of field, sharp focus, professional photography",
    "a realistic candid photo of {trigger}, natural daylight, detailed, highly aesthetic",
    "a cinematic shot of {trigger}, soft rim lighting, photorealistic, master quality",
    "a portrait of {trigger}, neutral background, balanced lighting, high resolution",
    "a dynamic shot of {trigger}, detailed expression, fine details, masterpiece",
    "a studio portrait of {trigger}, elegant composition, sharp detail, 4k",
]

PRODUCT_TEMPLATES = [
    "a commercial product photograph of {trigger}, studio backdrop, clean reflections, 8k uhd",
    "a professional product shot of {trigger}, softbox lighting, pristine condition, sharp focus",
    "an advertisement photo of {trigger}, isolated on minimalist background, high end commercial photography",
    "a detailed close-up shot of {trigger}, showing texture and material quality, crisp focus",
    "a studio promotional photograph of {trigger}, sleek presentation, balanced illumination",
    "an elegant commercial showcase of {trigger}, modern styling, premium look, 4k resolution",
]

STYLE_TEMPLATES = [
    "an artwork in the distinctive aesthetic style of {trigger}, vivid colors, detailed composition",
    "a digital masterpiece created in the visual style of {trigger}, intricate textures, striking atmosphere",
    "a captivating illustration in {trigger} art style, unique aesthetic, high dynamic range",
    "a detailed artistic rendering in {trigger} signature style, expressive and polished",
]

APPAREL_TEMPLATES = [
    "a commercial advertisement photo of a woman wearing {trigger}, detailed knit fabric texture, compression socks, studio lighting, clean background, 8k uhd",
    "a high quality lifestyle photo of an athletic woman wearing {trigger}, sharp focus on legs and compression socks, natural daylight, photorealistic",
    "a detailed close-up shot of {trigger} worn on calf and ankle, showing elastic compression texture and stitching, professional commercial photography",
    "a fitness commercial photograph of a woman wearing {trigger} after workout, athletic wear, clean aesthetic, sharp details, master quality",
    "a professional product catalog photo of {trigger} worn by a female model, perfect lighting, crisp focus on compression garment",
    "a realistic candid photo of a woman relaxing at home wearing {trigger}, cozy atmosphere, detailed fabric, high resolution",
    "a commercial photo of a woman wearing {trigger}, dynamic pose, healthy active lifestyle, clean studio backdrop, 4k uhd",
]

GENERAL_TEMPLATES = [
    "a high resolution detailed photograph of {trigger}, excellent lighting, sharp focus, 8k",
    "a clear shot of {trigger}, centered composition, natural lighting, high quality",
    "a crisp and vibrant photo of {trigger}, professional quality, sharp details",
]

def get_templates_for_category(category: str):
    cat = category.lower()
    if "apparel" in cat or "clothing" in cat or "sock" in cat or "compression" in cat or "wear" in cat or "bas" in cat:
        return APPAREL_TEMPLATES
    elif "person" in cat or "face" in cat or "portrait" in cat:
        return PERSON_TEMPLATES
    elif "product" in cat or "object" in cat or "item" in cat:
        return PRODUCT_TEMPLATES
    elif "style" in cat or "art" in cat:
        return STYLE_TEMPLATES
    return GENERAL_TEMPLATES

def caption_dataset(dataset_dir: Path, trigger: str, category: str = "general", overwrite: bool = False):
    images = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.PNG", "*.JPG", "*.JPEG", "*.WEBP"):
        images.extend(dataset_dir.glob(ext))
    images = sorted(list(set(images)))

    if not images:
        return {"success": False, "error": f"No images found in {dataset_dir}", "count": 0}

    templates = get_templates_for_category(category)
    results = []

    for i, img_path in enumerate(images):
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists() and not overwrite:
            caption = txt_path.read_text(encoding="utf-8").strip()
        else:
            template = templates[i % len(templates)]
            caption = template.format(trigger=trigger)
            txt_path.write_text(caption, encoding="utf-8")

        # Get image dimensions
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            w, h = 0, 0

        results.append({
            "image": img_path.name,
            "caption": caption,
            "width": w,
            "height": h,
            "size_bytes": img_path.stat().st_size if img_path.exists() else 0
        })

    return {
        "success": True,
        "count": len(results),
        "dataset_dir": str(dataset_dir),
        "trigger": trigger,
        "category": category,
        "items": results
    }

def main():
    parser = argparse.ArgumentParser(description="Auto-captioning utility for LoRA training datasets")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing dataset images")
    parser.add_argument("--trigger", default="sks person", help="Trigger word / token for the subject")
    parser.add_argument("--category", default="general", choices=["person", "product", "style", "general"], help="Subject category")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .txt captions")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"[ERROR] Directory '{dataset_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    result = caption_dataset(dataset_dir, args.trigger, args.category, args.overwrite)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if result["success"]:
            print(f"[INFO] Successfully captioned {result['count']} images for trigger '{args.trigger}'.")
            for item in result["items"]:
                print(f"  • {item['image']}: {item['caption']}")
        else:
            print(f"[ERROR] {result.get('error')}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
