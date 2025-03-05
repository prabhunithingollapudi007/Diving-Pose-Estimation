import cv2
import torch
import numpy as np
from model.model import MattingNetwork
import time
from argparse import ArgumentParser

# Load RVM model
model_path = "rvm_mobilenetv3.pth"
model = MattingNetwork('mobilenetv3')
model.load_state_dict(torch.load(model_path))
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = model.to(device).eval()

# Input and output video paths
parser = ArgumentParser()
parser.add_argument("--base_name", type=str, required=True, help="Base name of the video file")
base_name = parser.parse_args().base_name
input_video_path = f"../../data/preprocessed/{base_name}_rotated.mp4"
output_video_path = f"../../data/segmented/{base_name}_segmented.mp4"

# Open input video
cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video writer
out = cv2.VideoWriter(
    output_video_path,
    cv2.VideoWriter_fourcc(*'XVID'),
    fps,
    (frame_width, frame_height)
)

# Initialize RVM memory
r1, r2, r3, r4 = [None] * 4

print(f"Output video will be saved to: {output_video_path}")

# Initialize timing variables
frame_count = 0
total_time = 0

# Process frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    # Preprocess the frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
    input_tensor = torch.from_numpy(np.transpose(rgb_frame, (2, 0, 1))).unsqueeze(0).float().to(device)

    # Perform matting
    with torch.no_grad():
        fgr, pha, r1, r2, r3, r4 = model(input_tensor, r1, r2, r3, r4)

    # Post-process the mask
    pha = pha[0][0].cpu().numpy()
    mask = (pha * 255).astype(np.uint8)

    # Find the largest connected component (assumed to be the diver)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels > 1:
        # Get the largest non-background component
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.uint8(labels == largest_label) * 255

    # Dilate the mask to remove small holes
    kernel = np.ones((100, 100), np.uint8)  # Adjusted kernel size for better precision
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Apply the mask to the original frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Write the frame to the output video
    out.write(masked_frame)

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
