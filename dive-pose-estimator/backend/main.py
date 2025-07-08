# backend/main.py
from fastapi import FastAPI, UploadFile, File, Form
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
async def upload_video(file: UploadFile = File(...),
    rotate: bool = Form(...),
    stage_detection: bool = Form(...),
    start_time: float = Form(...),
    end_time: float = Form(...),
    board_height: float = Form(...),
    diver_height: float = Form(...)
                       ):
    # Save uploaded video
    file_id = str(uuid.uuid4())
    input_path = f"{UPLOAD_DIR}/{file_id}.mp4"
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Define output paths
    output_video_path = f"{OUTPUT_DIR}/{file_id}_out.webm"
    output_json_path = f"{OUTPUT_DIR}/{file_id}_metrics.json"
    
    """ # Parameters for the pipeline
    rotate = True  # Set to True if you want to rotate the video
    stage_detection = False  # Set to True if you want to detect stages
    start_time = 12  # Start
    end_time = 18
    board_height = 5
    diver_height = 1.75 """

    # Run your pipeline
    ex = run_pipeline(input_path, output_video_path, output_json_path, rotate, stage_detection, start_time, end_time, board_height, diver_height)

    if isinstance(ex, Exception):
        return JSONResponse(status_code=500, content={"error": str(ex)})

    return {
        "video_url": f"/result/video/{file_id}",
        "metrics_url": f"/result/metrics/{file_id}"
    }

@app.get("/result/video/{file_id}")
async def get_video(file_id: str):
    video_path = f"{OUTPUT_DIR}/{file_id}_out.webm"
    return FileResponse(video_path, media_type="video/webm")

@app.get("/result/metrics/{file_id}")
async def get_metrics(file_id: str):
    json_path = f"{OUTPUT_DIR}/{file_id}_metrics.json"
    with open(json_path, "r") as f:
        data = f.read()
    return JSONResponse(content=data)
