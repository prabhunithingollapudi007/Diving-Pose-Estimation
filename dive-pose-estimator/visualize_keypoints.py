import json
import numpy as np
import cv2
import time
from filter_keypoints import KeypointFilter
from joint_angles import process_pose_angles, compute_total_rotation
from utils import draw_keypoints, bbox_distance, is_bbox_valid, is_next_bbox_valid, is_bbox_in_center
from argparse import ArgumentParser
import matplotlib.pyplot as plt

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

# Initialize keypoint filter
keypoint_filter = KeypointFilter(window_size=5)

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

            if previous_bbox is None and is_bbox_in_center(bbox, frame_height):
                best_match_instance = instance
                previous_bbox = bbox
                break
            
            if not previous_bbox:
                continue

            if not is_next_bbox_valid(bbox, previous_bbox):
                continue
            
            best_match_instance = instance
            break

        if best_match_instance:
            keypoints = best_match_instance["keypoints"]
            previous_bbox = best_match_instance["bbox"][0]  # Update latest bbox

            # Draw keypoints
            pose_frame = draw_keypoints(pose_frame, keypoints, skeleton_links, previous_bbox)
            pose_frame, angles, torso_angles, com = process_pose_angles(pose_frame, keypoints, torso_angles)

            # Compute total rotation angle
            total_rotation = compute_total_rotation(torso_angles)
            total_rotation_over_time.append(total_rotation)

            # Diver center of mass
            diver_com_over_time.append(com)
            
        else:
            # If no valid bbox is found, still add rotation (but duplicate last value)
            if total_rotation_over_time:
                total_rotation_over_time.append(total_rotation_over_time[-1])
            else:
                total_rotation_over_time.append(0)  # If first frame is missing

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

# Plot total rotation angle and diver com over time
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.plot(total_rotation_over_time)
plt.title("Total Rotation Angle Over Time")
plt.xlabel("Frame")
plt.ylabel("Total Rotation Angle (degrees)")
plt.grid(True)

plt.subplot(1, 2, 2)
com_x, com_y = zip(*diver_com_over_time)
plt.plot(com_x, com_y, label="Center of Mass Path", marker='o', color='blue')
# Add title and labels
plt.title("Diver Center of Mass Path")
plt.xlabel("X Position (pixels)")
plt.ylabel("Y Position (pixels)")

# Add gridlines for better readability
plt.grid(True)

plt.savefig(f"data/pose-estimated/{base_name}/total_rotation.png")
plt.show()
