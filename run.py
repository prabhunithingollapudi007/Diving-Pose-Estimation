""" This is the main entry point for the application."""

import subprocess
from argparse import ArgumentParser

# Input options
parser = ArgumentParser()
parser.add_argument("--input_video", type=str, required=True, help="Path to the input video")
parser.add_argument("--output_base_path", type=str, required=True, help="Base path for results")
parser.add_argument("--rotate", action="store_true", help="Rotate the video")
parser.add_argument("--stage_detection", action="store_true", help="Detect stages")
parser.add_argument("--start_time", type=float, default=None, help="Start time for processing in seconds")
parser.add_argument("--end_time", type=float, default=None, help="End time for processing in seconds")


""" Example usage:
C:/Users/prabh/.conda/envs/openmmlab/python.exe "c:/Users/prabh/FAU/Study/MaD Project Pose Estimation/run.py" --input_video .\data\raw\Jana_107B_3.5Salti_vorwaerts.avi --output_base_path .\dive-pose-estimator\results\ --rotate --start_time 18 --end_time 25
"""

# Parse the arguments
args = parser.parse_args()

input_video = args.input_video
output_base_path = args.output_base_path
rotate = args.rotate
start_time = args.start_time
end_time = args.end_time
stage_detection = args.stage_detection

# Step 1: Rotate the video if needed

# pass the arguments to the rotate_video.py script

command = [
    "C:/Users/prabh/.conda/envs/openmmlab/python.exe",
    "dive-pose-estimator/rotate_video.py",
    "--input_video",
    input_video,
    "--output_base_path",
    output_base_path,
]

if rotate:
    command.append("--rotate")
if start_time is not None:
    command.extend(["--start_time", str(start_time)])
if end_time is not None:
    command.extend(["--end_time", str(end_time)])

print("Calling rotate video.py")
# subprocess.run(command)
print("==================================================================")

# Step 2: Proceed to segment the video using RVM
command = [
    "C:/Users/prabh/.conda/envs/openmmlab/python.exe",
    "models/RobustVideoMatting/bg_remove_rvm.py",
    "--input_video",
    f"{output_base_path}/rotated_video.mp4",
    "--output_base_path",
    output_base_path,
]

print("Calling remove background using RVM")
# subprocess.run(command)
print("==================================================================")

# Step 3: Auto Trim the video if start and end time are not provided
if start_time is None and end_time is None:
    command = [
        "C:/Users/prabh/.conda/envs/openmmlab/python.exe",
        "dive-pose-estimator/trim_video.py",
        "--input_video",
        f"{output_base_path}/segmented_video.mp4",
        "--output_base_path",
        output_base_path,
    ]

    print("Calling auto trim")
    # subprocess.run(command)
    print("==================================================================")

# Step 4: Proceed to extract the pose using RTMPOSE
command = [
    "C:/Users/prabh/.conda/envs/openmmlab/python.exe",
    "models/mmpose/demo/pose_estimation.py",
    "--input_video",
    f"{output_base_path}/segmented_video.mp4",
    "--output_base_path",
    output_base_path,
]

print("Calling pose estimation using RTMPOSE")
# subprocess.run(command)
print("==================================================================")

# Step 5: Proceed to visualize the pose and analyze the results
command = [
    "C:/Users/prabh/.conda/envs/openmmlab/python.exe",
    "dive-pose-estimator/visualize_keypoints.py",
    "--input_video",
    f"{output_base_path}/segmented_video.mp4",
    "--output_base_path",
    output_base_path,
]

if stage_detection:
    command.extend(["--stage_detection", str(stage_detection)])

print("Calling visualize keypoints")
subprocess.run(command)
print("==================================================================")