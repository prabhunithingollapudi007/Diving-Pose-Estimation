import json, shutil
import subprocess
# backend/pipeline.py

OUTPUT_DIR = "outputs"

def run_pipeline(input_path: str, output_video_path: str, output_json_path: str):
    # Place your existing logic here
    # Save pose-estimated video to output_video_path
    # Save metrics to output_json_path as JSON
    print(f"Running pipeline on {input_path}")
    # dummy outputs
    shutil.copy(input_path, output_video_path)
    # read the json file
    joint_angles_file = f"{OUTPUT_DIR}/joint_angles.json"
    filtered_metrics_file = f"{OUTPUT_DIR}/filtered_metrics.json"
    video_file = f"{OUTPUT_DIR}/pose_estimation_output.webm"
    with open(joint_angles_file, "r") as f:
        joint_angles = json.load(f)
    with open(filtered_metrics_file, "r") as f:
        filtered_metrics = json.load(f)
    json.dump({
        "filtered_metrics": filtered_metrics,
        "joint_angles": joint_angles
    }, open(output_json_path, "w"))
    # 
    # Save the video file to output_video_path
    # subprocess.run(["ffmpeg", "-i", video_file, output_video_path])
    # For now, just copy the video file
    shutil.copy(video_file, output_video_path)
    print(f"Pipeline completed. Outputs saved to {output_video_path} and {output_json_path}")

