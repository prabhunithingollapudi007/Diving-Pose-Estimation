import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Input and output video paths
input_video_path = "../data/preprocessed/two-person-sync_rotated.mp4"
output_video_path = "two-person-sync_rotated_mediapipe.mp4"

# Open the input video
cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a VideoWriter for the output
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

# Process each frame
print("Processing video...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Start time for this frame
    start_time = time.time()

    # Convert the frame to RGB (required by MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform segmentation
    results = selfie_segmentation.process(rgb_frame)

    # Generate the mask
    mask = results.segmentation_mask
    mask = (mask > 0.2).astype(np.uint8) * 255  # Binary mask

    # Dilate the mask to remove noise
    kernel = np.ones((200, 200), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Apply the mask to the frame
    foreground = cv2.bitwise_and(frame, frame, mask=mask)

    # Create a black background
    background = np.full(frame.shape, background_color, dtype=np.uint8)

    # Combine foreground and background
    mask_inv = cv2.bitwise_not(mask)
    combined = cv2.add(foreground, cv2.bitwise_and(background, background, mask=mask_inv))

    # Write the frame to the output video
    out.write(combined)

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
