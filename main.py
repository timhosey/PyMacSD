from fastapi import FastAPI, HTTPException
# --- Global exception handler and logging middleware ---
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from pydantic import BaseModel
import requests
import base64
import uuid
from pathlib import Path
import json, time, netifaces, subprocess, socket, time, traceback
from threading import Thread
import datetime

comfyui_process = None
last_access_time = datetime.datetime.now(datetime.UTC)
IDLE_TIMEOUT = datetime.timedelta(minutes=20)

def monitor_idle_shutdown():
    global comfyui_process
    while True:
        time.sleep(60)  # check every 60s
        if comfyui_process and comfyui_process.poll() is None:
            idle_duration = datetime.datetime.now(datetime.UTC) - last_access_time
            if idle_duration > IDLE_TIMEOUT:
                print("💤 Idle timeout reached, shutting down ComfyUI.")
                comfyui_process.terminate()
                comfyui_process = None

def is_backend_running(host="localhost", port=8188):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex((host, port)) == 0

def start_comfyui():
    global comfyui_process
    print("🐾 Nyaa~ Launching ComfyUI backend with extra sparkles!")
    with open("comfyui.log", "w") as log_file:
        comfyui_process = subprocess.Popen(
            ["python", "main.py"],
            cwd="./ComfyUI",
            stdout=log_file,
            stderr=log_file,
        )

def wait_for_backend(timeout=60):
    print("✨ Waiting for our comfy lil' backend to wake up~")
    for _ in range(timeout):
        try:
            res = requests.get("http://localhost:8188", timeout=5)
            if res.status_code == 200:
                print("🎉 Backend is up! Let's make some magic, nya~")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    print("Backend failed to start in time.")
    return False


if not is_backend_running():
    start_comfyui()
    wait_for_backend(timeout=180)

# Default negative prompt
DEFAULT_NEGATIVE = "lowres, blurry, bad anatomy, extra limbs, jpeg artifacts, grainy, normal quality, bad hands, error, text, logo, watermark, banner, extra digits, signature, cropped, out of frame, out of focus, bad proportions, deformed, disconnected limbs, disfigured, extra arms, extra hands, fused fingers, gross proportions, long neck, malformed limbs, missing arms, missing legs, poorly drawn face, mutation, poor quality, blurred, blurry, low contrast, poorly drawn hands, poorly drawn face, missing fingers, elongated fingers"

app = FastAPI()

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print("💥 Uh-oh! Yuki-chan caught a big bad error! >_<")
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": f"Unhandled exception: {str(exc)}"}
    )

@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"🐱✨ Incoming meow-quest: {request.method} {request.url}")
    response = await call_next(request)
    print(f"📦 Sending response with love! Status: {response.status_code}")
    return response

COMFYUI_URL = "http://localhost:8188"

class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    steps: int = 20
    cfg_scale: float = 7.0
    width: int = 512
    height: int = 512
    sampler_name: str = "euler"
    seed: int = -1  # -1 means random

@app.post("/generate")
def generate_image(data: GenerationRequest):
    global last_access_time
    last_access_time = datetime.datetime.now(datetime.UTC)
    workflow_path = Path("Anime.json")
    if not workflow_path.exists():
        raise HTTPException(status_code=500, detail="Workflow template not found.")

    with open(workflow_path, "r") as f:
        workflow = json.load(f)

    ENHANCEMENT_TAGS = "masterpiece, best quality, high-resolution, ultra-detailed, cinematic lighting, sharp focus, soft shadows, beautiful lighting"
    enhanced_prompt = f"{ENHANCEMENT_TAGS}, {data.prompt}"
    workflow["6"]["inputs"]["text"] = enhanced_prompt
    DEFAULT_NEGATIVE = "lowres, blurry, bad anatomy, watermark, text, extra limbs"
    final_negative_prompt = f"{DEFAULT_NEGATIVE}, {data.negative_prompt}".strip(", ")
    workflow["7"]["inputs"]["text"] = final_negative_prompt
    workflow["3"]["inputs"]["steps"] = data.steps
    workflow["3"]["inputs"]["cfg"] = data.cfg_scale
    workflow["3"]["inputs"]["sampler_name"] = data.sampler_name
    workflow["3"]["inputs"]["seed"] = 0 if data.seed < 0 else data.seed
    workflow["5"]["inputs"]["width"] = data.width
    workflow["5"]["inputs"]["height"] = data.height

    timestamp = int(time.time())
    filename_prefix = f"comfyui_{timestamp}"
    workflow["9"]["inputs"]["filename_prefix"] = filename_prefix

    try:
        res = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow})
        print(f"📨 ComfyUI POST /prompt response: {res.status_code} — {res.text}")
        res.raise_for_status()
        prompt_response = res.json()
        prompt_id = prompt_response.get("prompt_id")
        if not prompt_id:
            raise HTTPException(status_code=500, detail="No prompt_id returned by ComfyUI.")
    except requests.exceptions.RequestException as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")

    # Poll /history/{prompt_id} until it's ready
    print(f"📡 Polling history for prompt_id: {prompt_id}")
    output_path = Path("ComfyUI/output")
    output_path.mkdir(parents=True, exist_ok=True)
    output_filename = None
    for _ in range(60):  # Poll up to 60 seconds
        try:
            hist = requests.get(f"{COMFYUI_URL}/history/{prompt_id}").json()
            if prompt_id in hist and "outputs" in hist[prompt_id]:
                for node_output in hist[prompt_id]["outputs"].values():
                    images = node_output.get("images", [])
                    if images:
                        output_filename = images[0]["filename"]
                        print(f"🎉 Got filename from history: {output_filename}")
                        break
            if output_filename:
                break
        except Exception:
            traceback.print_exc()
        time.sleep(5)

    if not output_filename:
        print("😿 No output in history. Falling back to newest file in output folder...")
        pngs = list(output_path.glob("*.png"))
        if not pngs:
            raise HTTPException(status_code=500, detail="No output image received and no fallback file found.")
        newest_file = max(pngs, key=lambda f: f.stat().st_mtime)
        output_filename = newest_file.name
        print(f"✨ Using fallback image: {output_filename}")

    new_file = output_path / output_filename

    # Wait briefly to ensure file is fully flushed
    time.sleep(0.5)

    # Extra wait to ensure file is flushed and accessible
    for _ in range(10):
        if new_file.stat().st_size > 0:
            break
        time.sleep(0.2)
    else:
        raise HTTPException(status_code=500, detail="Image file not ready yet.")

    print(f"🌈 Image ready and sparkly at: {new_file}")

    def get_local_ip():
        for iface in netifaces.interfaces():
            if netifaces.AF_INET in netifaces.ifaddresses(iface):
                for link in netifaces.ifaddresses(iface)[netifaces.AF_INET]:
                    addr = link.get('addr')
                    if addr and not addr.startswith("127."):
                        return addr
        return "localhost"

    image_url = f"http://{get_local_ip()}:8000/image/{new_file.name}"
    return {
        "created": timestamp,
        "filename": new_file.name,
        "url": image_url
    }


from fastapi.responses import FileResponse

@app.get("/image/{filename}")
def get_image(filename: str):
    file_path = Path("ComfyUI/output") / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)

@app.post("/v1/images/generations")
def generate_openai_style(request: dict):
    global last_access_time
    last_access_time = datetime.datetime.now(datetime.UTC)
    prompt = request.get("prompt")
    size = request.get("size", "512x512")
    width, height = map(int, size.lower().split("x"))
    generation_data = GenerationRequest(
        prompt=prompt,
        negative_prompt=DEFAULT_NEGATIVE,
        width=width,
        height=height
    )

    result = generate_image(generation_data)
    return {
        "created": result["created"],
        "data": [
            {"url": result["url"]}
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, timeout_keep_alive=300)
    # Start idle monitor thread
    Thread(target=monitor_idle_shutdown, daemon=True).start()