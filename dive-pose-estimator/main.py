import argparse, os
import cv2
from preprocess import process_frame

def read_args():
    parser = argparse.ArgumentParser(description='Dive Pose Estimator')
    parser.add_argument('--video', type=str, default='../data/raw/video.mp4', help='Path to the video file', required=True)
    parser.add_argument('--output', type=str, default='../data/interim/video_output.mp4', help='Path to the output video file', required=True)
    parser.add_argument('--model', type=str, default='models/movenet_thunder.tflite', help='Path to the model', required=False)

    parser.add_argument('--width', type=int, default=256, help='Resize input to specific width.', required=False)
    parser.add_argument('--height', type=int, default=256, help='Resize input to specific height.', required=False)

    args = parser.parse_args()
    return args

def process_video(args):

    print(f"Video path: {args.video}")
    print(f"Output path: {args.output}")

    # Check if video file exists
    if not os.path.isfile(args.video):
        print(f"Error: Video file not found at {args.video}")
        return

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print("Error: Could not open video file" + args.video)
            return
    except:
        print("File not found: " + args.video)
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap = cv2.VideoCapture(args.video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # output video is rotated 90 degrees clockwise

    out = cv2.VideoWriter(args.output, fourcc, 30.0, (frame_height, frame_width))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame here
        processed_frame = process_frame(frame, args.width, args.height)
        
        out.write(processed_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    args = read_args()
    process_video(args)

if __name__ == '__main__':
    main()
