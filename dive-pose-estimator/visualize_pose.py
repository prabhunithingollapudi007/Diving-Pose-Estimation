""" Visualize the pose estimation results. """

import json
import numpy as np
import cv2
import time

from filter_keypoints import moving_average_filter

# Input json and video paths
file_path = "../data/pose-estimated/Jana/results_Jana_segmented.json"
input_video_path = '../data/preprocessed/Jana_rotated.mp4'
segmented_video_path = '../data/segmented/Jana_segmented.mp4'
output_video_path = "../data/Jana_side_by_side.mp4"

# Load JSON file
with open(file_path, 'r') as f:
    data = json.load(f)

# Extract metadata and instance info
meta_info = data["meta_info"]
instance_info = data["instance_info"]

# Collect all keypoints across frames
all_frames = []
raw_keypoints = []

for frame in instance_info:
    instances = frame["instances"]
    if instances:  # Ensure there are keypoints
        keypoints = [np.array(inst["keypoints"]) for inst in instances]
        bbox = [np.array(inst["bbox"][0]) for inst in instances]
        raw_keypoints.append(keypoints if len(keypoints) != 0 else [])
        all_frames.append({'keypoints': keypoints, 'bbox': bbox})
    else:
        all_frames.append({'keypoints': [], 'bbox': []})

""" 
# Apply smoothing filters to keypoints
smoothed_keypoints = moving_average_filter(raw_keypoints, window_size=5)


# Replace raw keypoints with smoothed keypoints
for i, frame in enumerate(all_frames):
    frame['smoothed_keypoints'] = smoothed_keypoints[i]
 """

# Print the number of frames and instances
num_frames = len(all_frames)
num_instances = len(all_frames[0])

print(f"Number of frames: {num_frames}")

# Video settings
cap = cv2.VideoCapture(input_video_path)
fps = 30
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# Create a VideoWriter for the output
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
    smoothed_frame = np.zeros((frame_height, frame_width, 3), np.uint8)

    # Get keypoints for the current frame
    if frame_idx < len(all_frames):
        frame = all_frames[frame_idx]
        keypoints = frame['keypoints']
        bbox = frame['bbox']
        # smoothed_keypoints = frame['smoothed_keypoints']

        if len(keypoints) != 0:
            # Draw keypoints and skeleton on pose_frame
            for instance_keypoints in keypoints:
                for i, keypoint in enumerate(instance_keypoints):
                    x, y = keypoint  # Assuming keypoints are [x, y, visibility]
                    cv2.circle(pose_frame, (int(x), int(y)), 5, colors[i], -1)

            # Draw the skeleton connections
            for connection in skeleton_links:
                start_idx, end_idx = connection
                if all(instance_keypoints[start_idx]) and all(instance_keypoints[end_idx]):
                    start_point = (int(instance_keypoints[start_idx][0]), int(instance_keypoints[start_idx][1]))
                    end_point = (int(instance_keypoints[end_idx][0]), int(instance_keypoints[end_idx][1]))
                    cv2.line(pose_frame, start_point, end_point, colors[start_idx], 2)

        """ if len(smoothed_keypoints) != 0:
            # Draw keypoints and skeleton on smoothed_frame
            for instance_keypoints in smoothed_keypoints:
                for i, keypoint in enumerate(instance_keypoints):
                    x, y = keypoint  # Assuming keypoints are [x, y, visibility]
                    cv2.circle(smoothed_frame, (int(x), int(y)), 5, colors[i], -1)

            # Draw the skeleton connections
            for connection in skeleton_links:
                start_idx, end_idx = connection
                if all(instance_keypoints[start_idx]) and all(instance_keypoints[end_idx]):
                    start_point = (int(instance_keypoints[start_idx][0]), int(instance_keypoints[start_idx][1]))
                    end_point = (int(instance_keypoints[end_idx][0]), int(instance_keypoints[end_idx][1]))
                    cv2.line(smoothed_frame, start_point, end_point, colors[start_idx], 2) """

        if len(bbox) != 0:
            # Draw bounding boxes on the original frame
            for box in bbox:
                x1, y1, x2, y2 = box
                cv2.rectangle(pose_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Concatenate the original frame and pose visualization side by side
    combined_frame = np.hstack((original_frame, pose_frame, smoothed_frame))

    # Write the combined frame to the output video
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