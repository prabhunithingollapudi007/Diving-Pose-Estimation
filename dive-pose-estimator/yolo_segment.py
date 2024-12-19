from ultralytics import YOLO
import cv2
import numpy as np
import time
# from bg_remove import remove_background
from bg_remove_yolo import process_frame

# Input and output video paths
input_video_path = "../data/preprocessed/two-person-sync_rotated.mp4"
output_video_path = "two-person-sync_rotated_yolo.mp4"

# Open the input video
cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video writer for saving output
out = cv2.VideoWriter(
    output_video_path,
    cv2.VideoWriter_fourcc(*'XVID'),
    fps,
    (frame_width, frame_height)
)

# Background color (black)
background_color = (0, 0, 0)


# Initialize timing variables
frame_count = 0
total_time = 0

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Start time for this frame
    start_time = time.time()
    
    frame = process_frame(frame)

    # Write the frame to output
    out.write(frame)

    # Calculate processing time for this frame
    frame_time = time.time() - start_time
    total_time += frame_time
    frame_count += 1

    # Print progress
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
