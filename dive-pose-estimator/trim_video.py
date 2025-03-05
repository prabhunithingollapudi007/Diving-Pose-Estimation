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
previous_bbox = None  # Bounding box of the diver
previous_y = None  # Y-coordinate of the diver in the previous frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1

    # Convert to grayscale for contour detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Find contours (potential people)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 20 and h > 20:  # Ignore small noise
            detected_bboxes.append((x, y, w, h))

    # Identify the main diver by proximity to the previous bbox
    closest_bbox = None
    min_dist = float("inf")

    for bbox in detected_bboxes:
        x, y, w, h = bbox

        if previous_bbox is None:
            # First frame: Select the highest person (smallest y)
            detected_bboxes.sort(key=lambda b: b[1])  # Sort by y-coordinate
            closest_bbox = detected_bboxes[0] if detected_bboxes else None
        else:
            # Compute distance from previous bbox
            px, py, pw, ph = previous_bbox
            dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)  # Euclidean distance
            if dist < min_dist:
                min_dist = dist
                closest_bbox = bbox  # Select closest person to previous bbox

    # Draw bounding boxes for visualization
    for bbox in detected_bboxes:
        x, y, w, h = bbox
        color = (0, 0, 255)  # Red for other objects
        if bbox == closest_bbox:
            color = (0, 255, 0)  # Green for detected diver
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    if closest_bbox:
        previous_bbox = closest_bbox  # Update previous bbox
        x, y, w, h = closest_bbox

        # Detect jump (y decreases)
        if previous_y is not None:
            velocity_y = previous_y - y  # Difference in y-coordinate (speed of movement)
            if start_frame is None and velocity_y > 10:  # Threshold for detecting jump (adjust as needed)
                start_frame = frame_count
                print(f"Jump detected at frame {start_frame} with velocity {velocity_y} at y={y}")

        # Detect reaching the bottom (more gradual threshold, percentage of screen)
        if y > frame_height * 0.75:  # The diver is at a lower position (85% of frame height)
            end_frame = frame_count
            print(f"Diver reached bottom at frame {end_frame} with y={y}")
            break

        previous_y = y # Update previous y-coordinate
    else:
        print("No diver detected.")


# Validate start_frame and end_frame
if start_frame is None or end_frame is None or start_frame >= end_frame:
    if start_frame is None:
        print("No jump detected.")
        start_frame = 0
    if end_frame is None:
        print("No bottom reached.")
        end_frame = frame_count - 1
    else:
        print("Invalid start_frame and end_frame. Resetting to full video.")
        start_frame = 0
        end_frame = frame_count - 1

# Trim the video from start_frame to end_frame
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

frame_count = start_frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_count > end_frame:
        break

    out.write(frame)
    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Trimmed video saved to: {output_video_path}")
