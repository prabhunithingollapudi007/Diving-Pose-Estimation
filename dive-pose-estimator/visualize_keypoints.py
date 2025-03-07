""" Visualize the pose estimation results. """

import json
import numpy as np
import cv2
import time
from filter_keypoints import KeypointFilter
from joint_angles import process_pose_angles, compute_total_rotation
from utils import draw_keypoints, colors
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

# Extract metadata and instance info and skeleton links
instance_info = data["instance_info"]
meta_info = data["meta_info"]
skeleton_links = meta_info["skeleton_links"]

# Initialize the filter
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

# Initialize variables
frame_idx = 0

# Initialize timing variables
frame_count = 0
total_time = 0

angles_per_frame = {
    "Torso": [],
    "Hip": [],
    "Knee": [],
    "Arm": [],
    "Total Rotation": []
}

torso_angles = []

# Loop through frames
while cap.isOpened():
    ret, trimmed_frame = cap.read()
    if not ret:
        break

    # Start time for this frame
    start_time = time.time()

    # Create a blank canvas for the pose visualization
    pose_frame = np.zeros((frame_height, frame_width, 3), np.uint8)

    # Process keypoints for the current frame
    if frame_idx < len(instance_info):
        frame = instance_info[frame_idx]
        instances = frame["instances"]
        main_diver_instance = None
        # Draw keypoints and bounding boxes for each instance ( multi-person pose estimation )
        for instance_id, instance in enumerate(instances):
            if main_diver_instance is None:
                main_diver_instance = instance_id
            elif main_diver_instance != instance_id:
                continue
            keypoints = instance["keypoints"]
            bbox = instance["bbox"]

            pose_frame = draw_keypoints(pose_frame, keypoints, colors, skeleton_links, bbox)
            
            pose_frame, angles, torso_angles = process_pose_angles(pose_frame, keypoints, torso_angles)
            for joint, angle in angles.items():
                angles_per_frame[joint].append(angle)

            # Only process the first instance for now
            break

    # Concatenate original, segmented, and pose visualization
    combined_frame = np.hstack((trimmed_frame, pose_frame))
    # Write output frame
    out.write(combined_frame)

    frame_idx += 1

    # Calculate processing time for this frame
    frame_time = time.time() - start_time
    total_time += frame_time
    frame_count += 1

    # Print progress
    if frame_count % 100 == 0:
        print(f"Processed {frame_count} frames")

# Release the VideoWriter and destroy all windows
cap.release()
out.release()
cv2.destroyAllWindows()

# Print overall stats
average_time_per_frame = total_time / frame_count if frame_count > 0 else 0
print(f"Processed {frame_count} frames in {total_time:.2f} seconds.")
print(f"Average time per frame: {average_time_per_frame:.3f} seconds ({1/average_time_per_frame:.2f} FPS)")

print(f"Output video saved at {output_video_path}")

# Plot total rotation angle over time
print("Total rotation angles:", torso_angles)
print("Total rotation angle over time:", compute_total_rotation(torso_angles))
plt.figure(figsize=(10, 6))
plt.plot(torso_angles, label="Total Rotation Angle")
plt.title("Total Rotation Angle Over Time")
plt.xlabel("Frame Index")
plt.ylabel("Angle (degrees)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f"data/pose-estimated/{base_name}/rotation_angles.png")
plt.show()

