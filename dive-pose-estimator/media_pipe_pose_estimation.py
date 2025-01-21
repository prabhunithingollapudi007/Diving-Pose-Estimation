import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.2, min_tracking_confidence=0.2)

# Input and output video paths
input_video = '../data/segmented/Jana_segmented.mp4'
output_video = '../data/pose-estimated/Jana_mediapipe.mp4'

# Open video
cap = cv2.VideoCapture(input_video)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Video writer to save the output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))


# Initialize timing variables
frame_count = 0
total_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Start time for this frame
    start_time = time.time()

    # Convert the frame to RGB (MediaPipe works with RGB images)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for pose estimation
    results = pose.process(frame_rgb)

    # Draw pose landmarks on the frame
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
        )

    # Write the frame with landmarks to the output video
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
pose.close()
cv2.destroyAllWindows()

# Print overall stats
average_time_per_frame = total_time / frame_count if frame_count > 0 else 0
print(f"Processed {frame_count} frames in {total_time:.2f} seconds.")
print(f"Average time per frame: {average_time_per_frame:.3f} seconds ({1/average_time_per_frame:.2f} FPS)")

print(f"Output video saved at {output_video}")