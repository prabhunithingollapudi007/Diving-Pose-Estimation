import tensorflow as tf
import tensorflow_hub as hub
import cv2
import time

# Load MoveNet model
movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
input_size = 256  # Input size for MoveNet Thunder

# Function to draw keypoints and connections
def draw_keypoints(frame, keypoints, confidence_threshold=0.2):
    height, width, _ = frame.shape
    for kp in keypoints:
        y, x, conf = kp
        if conf > confidence_threshold:
            cv2.circle(frame, (int(x * width), int(y * height)), 5, (0, 255, 0), -1)

def draw_connections(frame, keypoints, edges, confidence_threshold=0.2):
    height, width, _ = frame.shape
    for (p1, p2) in edges:
        y1, x1, c1 = keypoints[p1]
        y2, x2, c2 = keypoints[p2]
        if c1 > confidence_threshold and c2 > confidence_threshold:
            p1 = (int(x1 * width), int(y1 * height))
            p2 = (int(x2 * width), int(y2 * height))
            cv2.line(frame, p1, p2, (255, 0, 0), 2)

KEYPOINT_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11),
    (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

# Input and output video paths
input_video = '../data/segmented/Jana_segmented.mp4'
output_video = '../data/pose-estimated/Jana_movenet.mp4'

cap = cv2.VideoCapture(input_video)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

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

    # Resize frame to model input size and normalize
    resized_frame = cv2.resize(frame, (input_size, input_size))
    input_tensor = tf.convert_to_tensor(resized_frame, dtype=tf.int32)
    input_tensor = tf.expand_dims(input_tensor, axis=0)  # Add batch dimension


    # Run inference
    outputs = movenet.signatures['serving_default'](input_tensor)
    keypoints = outputs['output_0'].numpy()[0, 0, :, :]

    # Draw keypoints and connections
    draw_keypoints(frame, keypoints)
    draw_connections(frame, keypoints, KEYPOINT_EDGES)

    out.write(frame)

        # Calculate processing time for this frame
    frame_time = time.time() - start_time
    total_time += frame_time
    frame_count += 1

    # Print progress
    if frame_count % 100 == 0:
        print(f"Processed {frame_count} frames")

cap.release()
out.release()
cv2.destroyAllWindows()

# Print overall stats
average_time_per_frame = total_time / frame_count if frame_count > 0 else 0
print(f"Processed {frame_count} frames in {total_time:.2f} seconds.")
print(f"Average time per frame: {average_time_per_frame:.3f} seconds ({1/average_time_per_frame:.2f} FPS)")

print(f"Output video saved at {output_video}")