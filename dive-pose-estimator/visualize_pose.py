""" Visualize the pose estimation results. """

import json
import numpy as np


# Input json and output video paths
file_path = "../data/pose-estimated/results_two-person-sync_rotated_rvm.json"
output_video_path = "two-person-sync_rotated_rvm_pose.mp4"

# Load JSON file
with open(file_path, 'r') as f:
    data = json.load(f)

# Extract metadata and instance info
meta_info = data["meta_info"]
instance_info = data["instance_info"]

# Collect all keypoints across frames
all_keypoints = []

for frame in instance_info:
    frame_id = frame["frame_id"]
    instances = frame["instances"]
    if instances:  # Ensure there are keypoints
        keypoints = [np.array(inst["keypoints"]) for inst in instances]
        all_keypoints.append(keypoints)
    else:
        all_keypoints.append([])


# Visualize the pose estimation results on blank canvas
import cv2

# Output video settings
fps = 25
frame_width = 2048
frame_height = 1152

# Create a VideoWriter for the output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Create a blank canvas 
canvas = np.zeros((frame_height, frame_width, 3), np.uint8)

# Define colors for each keypoint

colors = [

    (255, 0, 0),  # Nose
    (255, 85, 0),  # Left eye
    (255, 170, 0),  # Right eye
    (255, 255, 0),  # Left ear
    (170, 255, 0),  # Right ear
    (85, 255, 0),  # Left shoulder
    (0, 255, 0),  # Right shoulder
    (0, 255, 85),  # Left elbow
    (0, 255, 170),  # Right elbow
    (0, 255, 255),  # Left wrist
    (0, 170, 255),  # Right wrist
    (0, 85, 255),  # Left hip
    (0, 0, 255),  # Right hip
    (85, 0, 255),  # Left knee
    (170, 0, 255),  # Right knee
    (255, 0, 255),  # Left ankle
    (255, 0, 170)  # Right ankle
]

# Define the number of frames
num_frames = len(all_keypoints)

skeleton_links = meta_info["skeleton_links"]

# Loop through frames
for frame_idx in range(num_frames):
    # Create blank canvas for the frame
    frame = canvas.copy()

    # Get the keypoints for the frame
    keypoints = all_keypoints[frame_idx]

    # Loop through each instance in the frame
    for instance_keypoints in keypoints:
        # Draw the keypoints on the frame and skeleton connections
        for i, keypoint in enumerate(instance_keypoints):
            x, y = keypoint
            cv2.circle(frame, (int(x), int(y)), 5, colors[i], -1)

        # Draw the skeleton connections
        for connection in skeleton_links:
            start_idx, end_idx = connection
            if all(instance_keypoints[start_idx]) and all(instance_keypoints[end_idx]):
                start_point = (int(instance_keypoints[start_idx][0]), int(instance_keypoints[start_idx][1]))
                end_point = (int(instance_keypoints[end_idx][0]), int(instance_keypoints[end_idx][1]))
                cv2.line(frame, start_point, end_point, colors[start_idx], 2)

    # Write the frame to the output video
    out.write(frame)


# Release the VideoWriter and destroy all windows
out.release()
cv2.destroyAllWindows()

print(f"Output video saved at {output_video_path}")
