import argparse
import json
import urllib.request
import urllib.parse
import time
import sys
import uuid
import os

try:
    import websocket
except ImportError:
    print("[WARN] websocket-client not installed. Using fallback polling.", flush=True)
    websocket = None

def queue_prompt(prompt, client_id, server_address):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type, server_address):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(prompt_id, server_address):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())
        
def main():
    parser = argparse.ArgumentParser(description="ComfyUI Headless Engine")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, default="sd_xl_base_1.0.safetensors")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workflow", type=str, required=True, help="Path to JSON workflow API file")
    parser.add_argument("--server", type=str, default="127.0.0.1:8188", help="ComfyUI Server Address")
    
    args = parser.parse_args()
    
    with open(args.workflow, 'r', encoding='utf-8') as f:
        workflow = json.load(f)
        
    # We assume specific node IDs based on our sdxl_t2i.json template
    if "3" in workflow and workflow["3"]["class_type"] == "KSampler":
        workflow["3"]["inputs"]["seed"] = args.seed
        workflow["3"]["inputs"]["steps"] = args.steps
    if "4" in workflow and workflow["4"]["class_type"] == "CheckpointLoaderSimple":
        workflow["4"]["inputs"]["ckpt_name"] = args.model
    if "5" in workflow and workflow["5"]["class_type"] == "EmptyLatentImage":
        workflow["5"]["inputs"]["width"] = args.width
        workflow["5"]["inputs"]["height"] = args.height
    if "6" in workflow and workflow["6"]["class_type"] == "CLIPTextEncode":
        workflow["6"]["inputs"]["text"] = args.prompt
        
    client_id = str(uuid.uuid4())
    print(f"Queueing prompt to ComfyUI at {args.server}...", flush=True)
    
    try:
        response = queue_prompt(workflow, client_id, args.server)
        prompt_id = response['prompt_id']
    except Exception as e:
        print(f"[ERROR] Could not connect to ComfyUI at {args.server}: {e}", flush=True)
        sys.exit(1)
        
    print(f"Prompt queued, ID: {prompt_id}", flush=True)
    
    try:
        if websocket is not None:
            ws = websocket.WebSocket()
            ws.connect("ws://{}/ws?clientId={}".format(args.server, client_id))
        else:
            ws = None
    except Exception as e:
        print(f"[WARN] WebSocket connection failed. Using polling. {e}")
        ws = None
        
    image_data = None
    
    if ws:
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        break # Execution is done
                elif message['type'] == 'progress':
                    data = message['data']
                    print(f"Progress: {data['value']}/{data['max']}", flush=True)
        ws.close()
    else:
        # Polling fallback
        print("Polling for history...", flush=True)
        while True:
            time.sleep(2)
            history = get_history(prompt_id, args.server)
            if prompt_id in history:
                break
    
    history = get_history(prompt_id, args.server)
    try:
        # Find the saved image
        outputs = history[prompt_id]['outputs']
        for node_id in outputs:
            node_output = outputs[node_id]
            if 'images' in node_output:
                for image in node_output['images']:
                    image_data = get_image(image['filename'], image['subfolder'], image['type'], args.server)
                    # For now we just grab the first image
                    break
    except Exception as e:
        print(f"[ERROR] Failed to extract image from history: {e}", flush=True)
        sys.exit(1)
        
    if image_data:
        with open(args.output, "wb") as f:
            f.write(image_data)
        print(f"Saved generated image to {args.output}", flush=True)
    else:
        print("[ERROR] No image data found in ComfyUI output.", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
