import json
import numpy as np
import cv2
import time
from joint_angles import process_pose_angles, compute_total_rotation, detect_stages, get_all_filtered_metrics
from utils import draw_keypoints, is_bbox_valid, is_next_bbox_valid, is_bbox_in_center, pixel_to_meter
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from config import MAX_CONSECUTIVE_INVALID_FRAMES, STAGES, DIVER_ON_BOARD_HEIGHT_PIXEL, WATER_HEIGHT_PIXEL, INITIAL_DIVER_HEIGHT_METERS, INITIAL_DIVER_HEIGHT_METERS, BOARD_HEIGHT_METERS
from filtering import kalman_filter, gaussian_filter, moving_average_filter
from utils import compute_angular_velocity

# Input paths
parser = ArgumentParser()
parser.add_argument("--base_name", type=str, required=True, help="Base name of the video file")
base_name = parser.parse_args().base_name

file_path = f"data/pose-estimated/{base_name}/results_{base_name}_trimmed.json"
trimmed_video_path = f'data/trimmed/{base_name}_trimmed.mp4'
# output_video_path = f"data/pose-estimated/{base_name}/{base_name}_side_by_side.mp4"
# output_base_path = f"data/pose-estimated/{base_name}"
output_video_path = f"dive-pose-estimator/results/{base_name}_side_by_side.mp4"
output_base_path = f"dive-pose-estimator/results"

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

joint_angles = {
    "Torso": [],
    "Hip": [],
    "Knee": [],
    "Arm": [],
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

            # Append joint angles
            for joint, angle in angles.items():
                joint_angles[joint].append(angle)

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


com_x, com_y, total_rotation_over_time, max_y = get_all_filtered_metrics(diver_com_over_time, total_rotation_over_time, diver_max_y_over_time)

rotation_rate = compute_angular_velocity(total_rotation_over_time, fps)

# Find the height of the diver

max_y = [frame_height - y for y in max_y]  # Convert Y-coordinate to top-left origin

# Assume y_board and y_water are known (fixed reference points)
y_max = max(max_y)  # Highest detected CoM
y_min = min(max_y)  # Lowest detected CoM (end of fall)

# Convert each CoM Y-coordinate to real-world height

scaling_factor = (BOARD_HEIGHT_METERS + INITIAL_DIVER_HEIGHT_METERS) / (DIVER_ON_BOARD_HEIGHT_PIXEL - WATER_HEIGHT_PIXEL)

diver_heights = [pixel_to_meter(y, scaling_factor) for y in max_y]

frame_times = np.arange(len(max_y)) / fps  # Assuming constant fps

# Filter joint angles
for joint, angles in joint_angles.items():
    joint_angles[joint] = kalman_filter(angles)

# Detect dive stages
stage_indices = detect_stages(joint_angles, torso_angles, diver_heights, total_rotation_over_time, output_video_path, output_base_path, STAGES)

# Rotation angle plot
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.plot(total_rotation_over_time, label="Total Rotation", color='blue')
for stage, idx in stage_indices.items():
    plt.axvline(x=idx, color='red', linestyle='--', label=stage)
plt.title("Total Rotation Angle Over Time with Stages")
plt.xlabel("Frame")
plt.ylabel("Total Rotation Angle (degrees)")
plt.legend()
plt.grid(True)

# Center of mass plot
plt.subplot(1, 3, 2)
plt.plot(com_x, com_y, label="Center of Mass Path", marker='o', color='blue')
plt.title("Diver Center of Mass Path")
plt.xlabel("X Position (pixels)")
plt.ylabel("Y Position (pixels)")
plt.legend()
plt.grid(True)

# Rotation rate plots
plt.subplot(1, 3, 3)
plt.plot(rotation_rate, label="Rotation Rate", color='blue')
for stage, idx in stage_indices.items():
    plt.axvline(x=idx, color='red', linestyle='--', label=stage)
plt.title("Diver Rotation Rate with Stages")
plt.xlabel("Frame")
plt.ylabel("Rotation Rate (degrees/second)")
plt.legend()
plt.grid(True)

# Save the figure
plt.savefig(f"{output_base_path}/metrics.png")


# Plot the diver height over time
plt.figure(figsize=(12, 6))
plt.plot(frame_times, diver_heights, label="Pose Estimation Height", color="blue")

# Find max height and time
max_height = max(diver_heights)
max_height_idx = diver_heights.index(max_height)
max_height_time = frame_times[max_height_idx]

plt.plot(max_height_time, max_height, 'ro', label=f"Max Height: {max_height:.2f}m at {max_height_time:.2f}s")

# Add stage lines
for stage, idx in stage_indices.items():
    plt.axvline(x=idx / fps, color='red', linestyle='--', label=f"{stage} Start")
    plt.text(idx / fps, max_height, stage, color='red', fontsize=8, ha='center')

plt.xlabel("Time (s)")
plt.ylabel("Height (m)")
plt.title("Diver Height: Pose Estimation")
plt.legend()
plt.grid(True)

plt.savefig(f"{output_base_path}/diver_height.png")

# Joint angles plot for each joint on 2 * 2 grid
plt.figure(figsize=(12, 6))
for i, (joint, angles) in enumerate(joint_angles.items()):
    plt.subplot(2, 2, i+1)
    plt.plot(angles, label=joint)
    for stage, idx in stage_indices.items():
        plt.axvline(x=idx, color='red', linestyle='--', label=stage)
    plt.title(f"{joint} Angle Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Angle (degrees)")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig(f"{output_base_path}/joint_angles.png")

