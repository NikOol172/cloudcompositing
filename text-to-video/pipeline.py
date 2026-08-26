import requests
import time
import os
import re
import subprocess
import random

# Configuration
API_KEY = os.environ.get("RUNPOD_API_KEY")
BASE_URL = "https://api.runpod.ai/v2"

# Endpoint IDs
QWEN_ENDPOINT = "qwen3-32b-awq"
FLUX_ENDPOINT = "black-forest-labs-flux-1-schnell"
WAN_ENDPOINT = "wan-2-5"

def get_headers():
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

"""
def enhance_prompt(simple_prompt):
	#"Use Qwen3 32B to enhance a simple prompt into a detailed image description."
	print(f"Enhancing prompt: {simple_prompt}")

	try:	
		response = requests.post(
			f"{BASE_URL}/{QWEN_ENDPOINT}/openai/v1/chat/completions",
			headers=get_headers(),
			json={
				"model": "Qwen/Qwen3-32B-AWQ",
				"messages": [
				{
					"role": "system",
					"content": "You are an expert at writing prompts for AI image generation. "
					"Transform the user's simple idea into a detailed, vivid image description. "
					"Include details about lighting, style, composition, and atmosphere. "
					"Keep the description under 100 words. Output only the enhanced prompt, "
					"nothing else. Do not include any thinking or explanation.",
				},
				{"role": "user", "content": simple_prompt},
				],
				"max_tokens": 200,
				"temperature": 0.7,
			},
			timeout=30,
		)
	except Exception as e:
		print(f"  Warning: LLM enhancement failed ({e}), fallback to simple prompt.")
		return simple_prompt

	result = response.json()
	enhanced = result["choices"][0]["message"]["content"].strip()

	# Remove any <think>...</think> blocks (some models include reasoning)
	enhanced = re.sub(r"<think>.*?</think>", "", enhanced, flags=re.DOTALL).strip()
	# Also handle unclosed <think> tags
	enhanced = re.sub(r"<think>.*", "", enhanced, flags=re.DOTALL).strip()

	print(f"Enhanced prompt: {enhanced}")
	return enhanced
"""
	
def enhance_prompt(simple_prompt):
    """Use the prompt directly without LLM modification."""
    print(f"Using prompt directly: {simple_prompt}")
    return simple_prompt

def poll_for_completion(endpoint, job_id, timeout=300):
    """Poll an async job until completion."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        status_response = requests.get(
            f"{BASE_URL}/{endpoint}/status/{job_id}",
            headers=get_headers(),
        )
        status = status_response.json()

        if status["status"] == "COMPLETED":
            return status
        elif status["status"] == "FAILED":
            raise Exception(f"Job failed: {status}")
        else:
            print(f"  Status: {status['status']}, waiting...")
            time.sleep(5)

    raise Exception(f"Job timed out after {timeout} seconds")

def generate_image(prompt):
    """Use Flux Schnell to generate an image from the prompt."""
    print("Generating image with Flux Schnell...")

    # Submit async job
    response = requests.post(
        f"{BASE_URL}/{FLUX_ENDPOINT}/run",
        headers=get_headers(),
        json={
            "input": {
                "prompt": prompt,
                "width": 768,
                "height": 1344,
                "num_inference_steps": 4,
                "seed": random.randint(1, 99999999),
            }
        },
    )

    result = response.json()
    job_id = result["id"]
    print(f"  Job submitted: {job_id}")

        # Poll for completion
    status = poll_for_completion(FLUX_ENDPOINT, job_id)
    print("DEBUG status output:", status.get("output"))

    # Extraction de l'URL de l'image selon la réponse de l'endpoint
    output = status.get("output", {})
    if isinstance(output, dict):
        image_url = output.get("result") or output.get("message") or output.get("image")
        if not image_url and "images" in output and len(output["images"]) > 0:
            image_url = output["images"][0]
    elif isinstance(output, list) and len(output) > 0:
        image_url = output[0]
    else:
        image_url = output

    print(f"  Image URL: {image_url}")
    return image_url

def generate_video(image_url, prompt):
    """Use WAN 2.5 to animate the image into a video."""
    print("Generating video with WAN 2.5...")

    # Submit the job
    response = requests.post(
        f"{BASE_URL}/{WAN_ENDPOINT}/run",
        headers=get_headers(),
        json={
            "input": {
                "image": image_url,
                "prompt": prompt,
                "duration": 5,
                "resolution": "480p",
            }
        },
    )

    result = response.json()
    job_id = result["id"]
    print(f"  Job submitted: {job_id}")

    # Poll for completion
    status = poll_for_completion(WAN_ENDPOINT, job_id, timeout=600)
    print("DEBUG video status output:", status.get("output"))

    output = status.get("output", {})
    if isinstance(output, dict):
        video_url = output.get("result") or output.get("video_url") or output.get("video")
        if not video_url and "videos" in output and len(output["videos"]) > 0:
            video_url = output["videos"][0]
    elif isinstance(output, list) and len(output) > 0:
        video_url = output[0]
    else:
        video_url = output

    print(f"  Video URL: {video_url}")
    return video_url

def download_file(url, filename):
    """Download a file from a URL."""
    print(f"Downloading to {filename}...")
    response = requests.get(url)
    with open(filename, "wb") as f:
        f.write(response.content)
    print(f"Saved: {filename}")

def main():
    # Votre prompt
    simple_prompt = "Legs of a woman sitting comfortably in a plane wearing sock up to the knees"

    # Step 1: Enhance prompt
    enhanced_prompt = enhance_prompt(simple_prompt)

    # Step 2: Generate image
    image_url = generate_image(enhanced_prompt)

    # Step 3: Download & Preview Image
    download_file(image_url, "output_image.png")
    
    # Tente d'ouvrir l'image automatiquement dans Chrome
    try:
        subprocess.Popen(["google-chrome", "output_image.png"])
    except Exception:
        pass

    print("\n" + "="*50)
    print(" L'image a été téléchargée : output_image.png")
    print("="*50)

    # PAUSE : Attend votre validation
    input("\n Appuyez sur [ENTRÉE] pour lancer la génération de la vidéo (ou Ctrl+C pour annuler)... ")

    # Step 4: Generate video
    video_url = generate_video(image_url, enhanced_prompt)

    # Step 5: Download Video
    download_file(video_url, "output_video.mp4")

    print("\n Pipeline terminé avec succès !")
    print(" Image : output_image.png")
    print(" Vidéo : output_video.mp4")

if __name__ == "__main__":
    if not API_KEY:
        print("Error: Set RUNPOD_API_KEY environment variable")
        exit(1)
    main()

