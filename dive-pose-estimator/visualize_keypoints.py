import json
import numpy as np
import cv2
import time
from joint_angles import process_pose_angles, compute_total_rotation, detect_stages, get_all_filtered_metrics
from utils import draw_keypoints, is_bbox_valid, is_next_bbox_valid, is_bbox_in_center, pixel_to_meter
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from config import MAX_CONSECUTIVE_INVALID_FRAMES, STAGES, DIVER_ON_BOARD_HEIGHT_PIXEL, WATER_HEIGHT_PIXEL, INITIAL_DIVER_HEIGHT_METERS, INITIAL_DIVER_HEIGHT_METERS, BOARD_HEIGHT_METERS

# Input paths
parser = ArgumentParser()
parser.add_argument("--base_name", type=str, required=True, help="Base name of the video file")
base_name = parser.parse_args().base_name

file_path = f"data/pose-estimated/{base_name}/results_{base_name}_trimmed.json"
trimmed_video_path = f'data/trimmed/{base_name}_trimmed.mp4'
output_video_path = f"data/pose-estimated/{base_name}/{base_name}_side_by_side.mp4"

# Load JSON file
with open(file_path, 'r') as f:
    data = json.load(f)

# Extract metadata
instance_info = data["instance_info"]
meta_info = data["meta_info"]
skeleton_links = meta_info["skeleton_links"]

# Video settings
cap = cv2.VideoCapture(trimmed_video_path)
fps = 30
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create VideoWriter for output
output_width = frame_width * 2  # Side-by-side layout
output_height = frame_height
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

# Tracking Variables
frame_idx = 0
frame_count = 0
total_time = 0
previous_bbox = None  # Stores last known bounding box for the main diver

angles_per_frame = {
    "Torso": [],
    "Hip": [],
    "Knee": [],
    "Arm": [],
    "Total Rotation": []
}

torso_angles = []  # Stores per-frame torso angles
total_rotation_over_time = []  # Tracks cumulative rotation per frame
diver_com_over_time = []  # Tracks diver center of mass over time
diver_max_y_over_time = []  # Tracks diver max Y-coordinate over time
consecutive_invalid_frames = 0

# Loop through frames
while cap.isOpened():
    ret, trimmed_frame = cap.read()
    if not ret:
        break

    start_time = time.time()
    pose_frame = np.zeros((frame_height, frame_width, 3), np.uint8)

    best_match_instance = None

    if frame_idx < len(instance_info):
        frame = instance_info[frame_idx]
        instances = frame["instances"]

        for instance_id, instance in enumerate(instances):
            bbox = instance.get("bbox", None)

            # Fix possible nested lists
            if isinstance(bbox, list) and len(bbox) == 1:
                bbox = bbox[0]  

            if not bbox or not is_bbox_valid(bbox):
                continue # Skip invalid bbox

            # Check if the bbox is in the center of the frame for the first frame
            if previous_bbox is None and is_bbox_in_center(bbox, frame_height):
                best_match_instance = instance
                previous_bbox = bbox
                break
            
            if not previous_bbox:
                continue

            # Skip if the next bbox is too far from the previous bbox
            if not is_next_bbox_valid(bbox, previous_bbox):
                continue
            
            best_match_instance = instance
            break

        if best_match_instance:
            keypoints = best_match_instance["keypoints"]
            previous_bbox = best_match_instance["bbox"][0]  # Update latest bbox

            # Draw keypoints
            pose_frame = draw_keypoints(pose_frame, keypoints, skeleton_links, previous_bbox)
            pose_frame, angles, torso_angles, com, max_y = process_pose_angles(pose_frame, keypoints, torso_angles)

            # Compute total rotation angle
            total_rotation = compute_total_rotation(torso_angles)
            total_rotation_over_time.append(total_rotation)

            # Diver center of mass
            diver_com_over_time.append(com)

            # Diver max Y-coordinate
            diver_max_y_over_time.append(max_y)

            # reset consecutive invalid frames
            consecutive_invalid_frames = 0
        else:
            consecutive_invalid_frames += 1
            if consecutive_invalid_frames > MAX_CONSECUTIVE_INVALID_FRAMES:
                previous_bbox = None  # Reset previous bbox if no match for a while

            
    combined_frame = np.hstack((trimmed_frame, pose_frame))
    out.write(combined_frame)

    frame_idx += 1
    frame_time = time.time() - start_time
    total_time += frame_time
    frame_count += 1

    if frame_count % 100 == 0:
        print(f"Processed {frame_count} frames")

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Print overall stats
average_time_per_frame = total_time / frame_count if frame_count > 0 else 0
print(f"Processed {frame_count} frames in {total_time:.2f} seconds.")
print(f"Average time per frame: {average_time_per_frame:.3f} seconds ({1/average_time_per_frame:.2f} FPS)")
print(f"Output video saved at {output_video_path}")


com_x, com_y, rotation_angles, velocity_y, acceleration_y, rotation_rate, rotation_acceleration, max_y = get_all_filtered_metrics(diver_com_over_time, total_rotation_over_time, diver_max_y_over_time)

# Detect dive stages
# stage_indices = detect_stages(com_x, com_y, rotation_angles, velocity_y, acceleration_y, rotation_rate, rotation_acceleration, STAGES)

plt.figure(figsize=(12, 6))

# Rotation angle plot
plt.subplot(2, 2, 1)
plt.plot(total_rotation_over_time, label="Total Rotation", color='blue')
plt.title("Total Rotation Angle Over Time with Stages")
plt.xlabel("Frame")
plt.ylabel("Total Rotation Angle (degrees)")
plt.legend()
plt.grid(True)

# Center of mass plot
plt.subplot(2, 2, 2)
plt.plot(com_x, com_y, label="Center of Mass Path", marker='o', color='blue')
plt.title("Diver Center of Mass Path with Stages")
plt.xlabel("X Position (pixels)")
plt.ylabel("Y Position (pixels)")
plt.legend()
plt.grid(True)

# Velocity plots
plt.subplot(2, 2, 3)
plt.plot(velocity_y, label="Velocity Y", color='blue')
plt.plot(acceleration_y, label="Acceleration Y", color='red')
plt.title("Diver Velocity and Acceleration with Stages")
plt.xlabel("Frame")
plt.ylabel("Velocity / Acceleration (pixels/frame)")
plt.legend()
plt.grid(True)

# Rotation rate plots
plt.subplot(2, 2, 4)
plt.plot(rotation_rate, label="Rotation Rate", color='blue')
plt.plot(rotation_acceleration, label="Rotation Acceleration", color='red')
plt.title("Diver Rotation Rate and Acceleration with Stages")
plt.xlabel("Frame")
plt.ylabel("Rotation Rate / Acceleration (degrees/frame)")
plt.legend()
plt.grid(True)

# Save the figure
plt.savefig(f"data/pose-estimated/{base_name}/metrics.png")

# Find the height of the diver

max_y = [frame_height - y for y in max_y]  # Convert Y-coordinate to top-left origin

# Assume y_board and y_water are known (fixed reference points)
y_max = max(max_y)  # Highest detected CoM
y_min = min(max_y)  # Lowest detected CoM (end of fall)

# Convert each CoM Y-coordinate to real-world height

scaling_factor = (BOARD_HEIGHT_METERS + INITIAL_DIVER_HEIGHT_METERS) / (DIVER_ON_BOARD_HEIGHT_PIXEL - WATER_HEIGHT_PIXEL)

print(f"Scaling factor: {scaling_factor} to convert pixel-based CoM to real-world height")

diver_heights = [pixel_to_meter(y, scaling_factor) for y in max_y]

frame_times = np.arange(len(max_y)) / fps  # Assuming constant fps

# Plot the diver height over time
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(frame_times, diver_heights, label="Pose Estimation Height", color="blue")
plt.xlabel("Time (s)")
plt.ylabel("Height (m)")
plt.title("Diver Height: Pose Estimation")
plt.legend()
plt.grid(True)

# Plot the diver height over frames
plt.subplot(1, 2, 2)
plt.plot(com_y, diver_heights, label="Pose Estimation COM Pixel comparision", color="blue")
plt.xlabel("frame")
plt.ylabel("Height (pixels)")
plt.title("Diver Height: Raw Data")
plt.legend()
plt.grid(True)
# Save the figure

plt.savefig(f"data/pose-estimated/{base_name}/heights.png")
plt.show()