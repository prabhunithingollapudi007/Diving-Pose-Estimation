""" Visualize the pose estimation results. """

import json
import numpy as np
import cv2
import time

# Input paths
file_path = "data/pose-estimated/Jana/results_Jana_segmented.json"
input_video_path = 'data/preprocessed/Jana_rotated.mp4'
segmented_video_path = 'data/segmented/Jana_segmented.mp4'
output_video_path = "data/Jana_side_by_side.mp4"

# Load JSON file
with open(file_path, 'r') as f:
    data = json.load(f)

# Extract metadata and instance info
meta_info = data["meta_info"]
instance_info = data["instance_info"]

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

    # Process keypoints for the current frame
    if frame_idx < len(instance_info):
        frame = instance_info[frame_idx]
        instances = frame["instances"]

        # Draw keypoints and bounding boxes for each instance ( multi-person pose estimation )
        for instance_id, instance in enumerate(instances):
            keypoints = instance["keypoints"]
            bbox = instance["bbox"]

            # Draw keypoints on pose frame
            for i, keypoint in enumerate(keypoints):
                x, y = keypoint
                cv2.circle(pose_frame, (int(x), int(y)), 5, colors[i], -1)

            # Draw skeleton
            for start_idx, end_idx in skeleton_links:
                if all(keypoints[start_idx]) and all(keypoints[end_idx]):
                    start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                    end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                    cv2.line(pose_frame, start_point, end_point, colors[start_idx], 2)

            if len(bbox) != 0:
                # Draw bounding boxes on the original frame
                for box in bbox:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(pose_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

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