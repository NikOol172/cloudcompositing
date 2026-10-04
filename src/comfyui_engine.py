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

def upload_image(filepath, server_address):
    import urllib.request
    import mimetypes
    from email.mime.multipart import MIMEMultipart
    from email.mime.application import MIMEApplication
    
    filename = os.path.basename(filepath)
    msg = MIMEMultipart()
    with open(filepath, "rb") as f:
        part = MIMEApplication(f.read(), Name=filename)
    part['Content-Disposition'] = f'form-data; name="image"; filename="{filename}"'
    msg.attach(part)
    
    req = urllib.request.Request("http://{}/upload/image".format(server_address), data=msg.as_bytes())
    req.add_header('Content-Type', msg.get_content_type())
    return json.loads(urllib.request.urlopen(req).read())

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
    parser.add_argument("--input_image", type=str, default=None, help="Input image for LoadImage nodes (or Target for FaceSwap)")
    parser.add_argument("--input_image2", type=str, default=None, help="Second input image (Source Face for FaceSwap)")
    parser.add_argument("--strength", type=float, default=None, help="Denoise strength for Img2Img")
    
    args = parser.parse_args()
    
    with open(args.workflow, 'r', encoding='utf-8') as f:
        workflow = json.load(f)
        
    # We assume specific node IDs based on our sdxl_t2i.json template
    if "3" in workflow and workflow["3"]["class_type"] == "KSampler":
        workflow["3"]["inputs"]["seed"] = args.seed
        workflow["3"]["inputs"]["steps"] = args.steps
        if args.strength is not None and "denoise" in workflow["3"]["inputs"]:
            workflow["3"]["inputs"]["denoise"] = args.strength
    if "4" in workflow and workflow["4"]["class_type"] == "CheckpointLoaderSimple":
        workflow["4"]["inputs"]["ckpt_name"] = args.model
    if "5" in workflow and workflow["5"]["class_type"] == "EmptyLatentImage":
        workflow["5"]["inputs"]["width"] = args.width
        workflow["5"]["inputs"]["height"] = args.height
    if "6" in workflow and workflow["6"]["class_type"] == "CLIPTextEncode":
        workflow["6"]["inputs"]["text"] = args.prompt
        
    uploaded_image_name = None
    if args.input_image:
        image_path_to_upload = args.input_image
        if args.input_image.startswith('http://') or args.input_image.startswith('https://'):
            import tempfile
            import uuid
            tmp_path = os.path.join(tempfile.gettempdir(), f"downloaded_{uuid.uuid4().hex}.png")
            print(f"Downloading image from {args.input_image}...", flush=True)
            urllib.request.urlretrieve(args.input_image, tmp_path)
            image_path_to_upload = tmp_path
            
        if os.path.exists(image_path_to_upload):
            print(f"Uploading image {image_path_to_upload} to ComfyUI...", flush=True)
            try:
                up_res = upload_image(image_path_to_upload, args.server)
                uploaded_image_name = up_res.get("name")
            except Exception as e:
                print(f"[WARN] Failed to upload image: {e}")
            
            if image_path_to_upload != args.input_image:
                os.remove(image_path_to_upload)

    uploaded_image2_name = None
    if args.input_image2:
        image_path_to_upload = args.input_image2
        if args.input_image2.startswith('http://') or args.input_image2.startswith('https://'):
            import tempfile
            import uuid
            tmp_path = os.path.join(tempfile.gettempdir(), f"downloaded_{uuid.uuid4().hex}.png")
            print(f"Downloading image 2 from {args.input_image2}...", flush=True)
            urllib.request.urlretrieve(args.input_image2, tmp_path)
            image_path_to_upload = tmp_path
            
        if os.path.exists(image_path_to_upload):
            print(f"Uploading image 2 {image_path_to_upload} to ComfyUI...", flush=True)
            try:
                up_res = upload_image(image_path_to_upload, args.server)
                uploaded_image2_name = up_res.get("name")
            except Exception as e:
                print(f"[WARN] Failed to upload image 2: {e}")
            
            if image_path_to_upload != args.input_image2:
                os.remove(image_path_to_upload)

    # Assign images to LoadImage nodes. We try to be smart about titles if there are multiple.
    load_image_nodes = []
    load_video_nodes = []
    for node_id, node in workflow.items():
        if node.get("class_type") == "LoadImage":
            load_image_nodes.append(node)
        elif node.get("class_type") in ["VHS_LoadVideo", "LoadVideo"]:
            load_video_nodes.append(node)
            
    if uploaded_image_name and not uploaded_image2_name:
        for node in load_image_nodes:
            node["inputs"]["image"] = uploaded_image_name
        for node in load_video_nodes:
            if "video" in node["inputs"]:
                node["inputs"]["video"] = uploaded_image_name
    elif uploaded_image_name and uploaded_image2_name:
        # Assume first is target, second is source. Check titles if possible
        for node in load_image_nodes:
            title = node.get("_meta", {}).get("title", "").lower()
            if "source" in title or "face" in title:
                node["inputs"]["image"] = uploaded_image2_name
            elif "target" in title or "base" in title:
                node["inputs"]["image"] = uploaded_image_name
            else:
                # Fallback to order if titles don't match
                if node == load_image_nodes[0]:
                    node["inputs"]["image"] = uploaded_image_name
                else:
                    node["inputs"]["image"] = uploaded_image2_name
        
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
        # Find the saved image or video
        outputs = history[prompt_id]['outputs']
        for node_id in outputs:
            node_output = outputs[node_id]
            for media_key in ['images', 'gifs']:
                if media_key in node_output:
                    for media in node_output[media_key]:
                        image_data = get_image(media['filename'], media['subfolder'], media['type'], args.server)
                        break
                if image_data:
                    break
            if image_data:
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
