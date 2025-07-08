import cv2
from argparse import ArgumentParser
from config import FPS

# Input and output video paths
parser = ArgumentParser()

parser.add_argument("--input_video", type=str, required=True, help="Path to the input video")
parser.add_argument("--output_base_path", type=str, required=True, help="Base path for results")
parser.add_argument("--rotate", action="store_true", help="Rotate the video")
parser.add_argument("--start_time", type=float, default=None, help="Start time for processing in seconds")
parser.add_argument("--end_time", type=float, default=None, help="End time for processing in seconds")

input_video = parser.parse_args().input_video
output_base_path = parser.parse_args().output_base_path
output_video = f"{output_base_path}/rotated_video.mp4"

start_time = parser.parse_args().start_time
end_time = parser.parse_args().end_time

print(f"Input Video path for rotation: {input_video}")
print(f"Output Video path for rotation: {output_video}")
rotate = parser.parse_args().rotate
cap = cv2.VideoCapture(input_video)
fps = FPS

if not cap.isOpened():
    print("Error: Could not open input video.")
    exit()

# Get input video frame dimensions
ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame from input video.")
    cap.release()
    exit()

frame_height, frame_width = frame.shape[:2]
print(f"Input video dimensions: {frame_width}x{frame_height} and fps: {fps}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
if rotate:
    # Rotate the frame dimensions if the video is rotated
    frame_height, frame_width = frame_width, frame_height
print(f"Output video dimensions: {frame_width}x{frame_height}")
out = cv2.VideoWriter(output_video, fourcc, 30.0, (frame_width, frame_height))

if not out.isOpened():
    print("Error: Could not open output video.")
    cap.release()
    exit()
start_frame = None
end_frame = None
if start_time is not None and end_time is not None:
    # Calculate the frame numbers to start and end processing
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    print(f"Processing frames from {start_frame} to {end_frame} ({start_time} to {end_time} seconds)")
else:
    print("Processing all frames")


start_ticks = cv2.getTickCount()

frame_count = 0

while cap.isOpened():
    ret, original_frame = cap.read()
    if not ret:
        break

    if start_frame is not None and end_frame is not None:
        if frame_count <= start_frame or frame_count >= end_frame:
            frame_count += 1
            continue

    # Rotate the frame if specified
    if rotate:
        rotated_frame = cv2.rotate(original_frame, cv2.ROTATE_90_CLOCKWISE)
    else:
        rotated_frame = original_frame
    # Write the frame to the output video
    out.write(rotated_frame)

    frame_count += 1
    

cap.release()
out.release()
cv2.destroyAllWindows()

end_ticks = cv2.getTickCount()

total_time = (end_ticks - start_ticks) / cv2.getTickFrequency()
print(f"Total time taken for rotation: {total_time:.2f} seconds for {frame_count} frames")

print(f"Output video saved at {output_video}")