import cv2
import numpy as np
from argparse import ArgumentParser

# Input and output video paths
parser = ArgumentParser()
parser.add_argument("--base_name", type=str, required=True, help="Base name of the video file")
base_name = parser.parse_args().base_name
input_video_path = f"../data/segmented/{base_name}_segmented.mp4"
output_video_path = f"../data/trimmed/{base_name}_trimmed.mp4"

# Open input video
cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

frame_count = 0
start_frame = None
end_frame = None
previous_y = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1

    # Convert to grayscale for contour detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Find contours (diver detection)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)  # Assume the diver is the largest object
        x, y, w, h = cv2.boundingRect(largest_contour)  # Get bounding box

        if previous_y is not None:
            if start_frame is None and y < previous_y - 10:  # Diver is jumping off
                start_frame = frame_count

            if end_frame is None:
                # Check if diver is fully submerged
                if h < 20 or w < 20:  # Diver disappears (small bounding box)
                    end_frame = frame_count
                    print("Diver is fully submerged")
                    break
                elif y > frame_height * 0.9:  # Alternative: diver reaches bottom of frame
                    end_frame = frame_count
                    print("Diver reaches bottom of frame")
                    break

        previous_y = y  # Update previous y-coordinate

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame or 0)  # Start from detected start frame

frame_count = 0
# Write trimmed frames to output
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or (end_frame and frame_count > end_frame):
        break

    out.write(frame)
    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Trimmed video saved to: {output_video_path}")
