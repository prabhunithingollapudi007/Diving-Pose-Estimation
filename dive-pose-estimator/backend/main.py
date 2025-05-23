# backend/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
import os
from pipeline import run_pipeline

app = FastAPI()

# Allow frontend on localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    # Save uploaded video
    file_id = str(uuid.uuid4())
    input_path = f"{UPLOAD_DIR}/{file_id}.mp4"
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Define output paths
    output_video_path = f"{OUTPUT_DIR}/{file_id}_out.mp4"
    output_json_path = f"{OUTPUT_DIR}/{file_id}_metrics.json"

    # Run your pipeline
    run_pipeline(input_path, output_video_path, output_json_path)

    return {
        "video_url": f"/result/video/{file_id}",
        "metrics_url": f"/result/metrics/{file_id}"
    }

@app.get("/result/video/{file_id}")
async def get_video(file_id: str):
    video_path = f"{OUTPUT_DIR}/{file_id}_out.mp4"
    return FileResponse(video_path, media_type="video/mp4")

@app.get("/result/metrics/{file_id}")
async def get_metrics(file_id: str):
    json_path = f"{OUTPUT_DIR}/{file_id}_metrics.json"
    with open(json_path, "r") as f:
        data = f.read()
    return JSONResponse(content=data)
