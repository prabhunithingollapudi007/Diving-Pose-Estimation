import cv2

input_video = '../data/raw/Jana_107B_3.5Salti_vorwaerts.avi'
output_video = '../data/preprocessed/Jana_rotated.mp4'

print(f"Input Video path: {input_video}")
cap = cv2.VideoCapture(input_video)

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
print(f"Input video dimensions: {frame_width}x{frame_height}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, 30.0, (frame_height, frame_width))

if not out.isOpened():
    print("Error: Could not open output video.")
    cap.release()
    exit()

print(f"Output video dimensions: {frame_height}x{frame_width}")

start_time = cv2.getTickCount()

while cap.isOpened():
    ret, original_frame = cap.read()
    if not ret:
        break

    rotated_frame = cv2.rotate(original_frame, cv2.ROTATE_90_CLOCKWISE)
    # Write the frame to the output video
    out.write(rotated_frame)

cap.release()
out.release()
cv2.destroyAllWindows()

end_time = cv2.getTickCount()

total_time = (end_time - start_time) / cv2.getTickFrequency()
print(f"Total time taken: {total_time:.2f} seconds")

print(f"Output video saved at {output_video}")