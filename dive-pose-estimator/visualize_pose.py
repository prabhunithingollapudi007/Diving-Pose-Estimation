""" Visualize the pose estimation results. """

import json
import numpy as np
import cv2
import time
from filter_keypoints import KeypointFilter
from joint_angles import process_pose_angles
from utils import draw_keypoints
from argparse import ArgumentParser
import matplotlib.pyplot as plt


# Input paths
parser = ArgumentParser()
parser.add_argument("--base_name", type=str, required=True, help="Base name of the video file")
base_name = parser.parse_args().base_name

file_path = f"data/pose-estimated/{base_name}/results_{base_name}_segmented.json"
input_video_path = f'data/preprocessed/{base_name}_rotated.mp4'
segmented_video_path = f'data/segmented/{base_name}_segmented.mp4'
output_video_path = f"data/{base_name}_side_by_side.mp4"
filtering = False

# Load JSON file
with open(file_path, 'r') as f:
    data = json.load(f)

# Extract metadata and instance info
meta_info = data["meta_info"]
instance_info = data["instance_info"]

# Initialize the filter
keypoint_filter = KeypointFilter(window_size=5)

# Video settings
cap = cv2.VideoCapture(input_video_path)
fps = 30
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# Create VideoWriter for output
output_width = frame_width * 3  # Side-by-side layout
output_height = frame_height
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

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

# Skeleton links from metadata
skeleton_links = meta_info["skeleton_links"]

# Open input video
cap = cv2.VideoCapture(input_video_path)
segmented_cap = cv2.VideoCapture(segmented_video_path)
frame_idx = 0

# Initialize timing variables
frame_count = 0
total_time = 0

angles_per_frame = {"Torso": [], "Hip": [], "Knee": [], "Arm": []}

# Loop through frames
while cap.isOpened() and segmented_cap.isOpened():
    ret, original_frame = cap.read()
    ret_segmented, segmented_frame = segmented_cap.read()
    if not ret:
        break

    # Start time for this frame
    start_time = time.time()

    # Create a blank canvas for the pose visualization
    pose_frame = np.zeros((frame_height, frame_width, 3), np.uint8)
    filtered_frame = np.zeros((frame_height, frame_width, 3), np.uint8)

    # Process keypoints for the current frame
    if frame_idx < len(instance_info):
        frame = instance_info[frame_idx]
        instances = frame["instances"]

        # Draw keypoints and bounding boxes for each instance ( multi-person pose estimation )
        for instance_id, instance in enumerate(instances):
            keypoints = instance["keypoints"]
            bbox = instance["bbox"]

            pose_frame = draw_keypoints(pose_frame, keypoints, colors, skeleton_links, bbox)
            pose_frame, angles = process_pose_angles(pose_frame, keypoints)
            for joint, angle in angles.items():
                angles_per_frame[joint].append(angle)

            # Filter keypoints
            if filtering:
                segmented_frame = pose_frame
                keypoints = keypoint_filter.filter_keypoints(instance_id, keypoints)
                filtered_frame = draw_keypoints(filtered_frame, keypoints, colors, skeleton_links, bbox)
                filtered_frame = process_pose_angles(filtered_frame, keypoints)
                for joint, angle in angles.items():
                    angles_per_frame[joint].append(angle)
                pose_frame = filtered_frame

            # Only process the first instance for now
            break

    # Concatenate original, segmented, and pose visualization
    combined_frame = np.hstack((original_frame, segmented_frame, pose_frame))
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
out.release()
cv2.destroyAllWindows()

# Print overall stats
average_time_per_frame = total_time / frame_count if frame_count > 0 else 0
print(f"Processed {frame_count} frames in {total_time:.2f} seconds.")
print(f"Average time per frame: {average_time_per_frame:.3f} seconds ({1/average_time_per_frame:.2f} FPS)")

print(f"Output video saved at {output_video_path}")

# Plot angles over time

plt.figure(figsize=(12, 6))
for joint, angles in angles_per_frame.items():
    plt.plot(angles, label=joint)

plt.title("Joint Angles Over Time")
plt.xlabel("Frame Index")
plt.ylabel("Angle (degrees)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f"data/{base_name}_joint_angles.png")
plt.show()

