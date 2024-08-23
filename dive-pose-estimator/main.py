import argparse, os
import cv2
from preprocess import preprocess_frame
from infer import run_movenet
from postprocess import draw_keypoints

def read_args():
    parser = argparse.ArgumentParser(description='Dive Pose Estimator')
    parser.add_argument('--video', type=str, default='../data/raw/video.mp4', help='Path to the input video file', required=True)
    parser.add_argument('--output', type=str, default='../data/interim/video_output.mp4', help='Path to the output video file', required=True)
    parser.add_argument('--model', type=str, default='../models/movenet_thunder.tflite', help='Path to the model', required=False)
    parser.add_argument('--rotate', type=bool, default=True, help='Rotate the video 90 degrees clockwise.', required=False)
    parser.add_argument('--resize', type=bool, default=False, help='Resize the input video.', required=False)
    parser.add_argument('--width', type=int, default=256, help='Resize input to specific width.', required=False)
    parser.add_argument('--height', type=int, default=256, help='Resize input to specific height.', required=False)
    parser.add_argument('--live', type=bool, default=False, help='Display the output video in a window.', required=False)

    args = parser.parse_args()
    return args

def process_video(args):

    print(f"Input Video path: {args.video}")
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

    target_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    target_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.resize:
        target_frame_width = args.width
        target_frame_height = args.height

    if args.rotate:
        target_frame_width, target_frame_height = target_frame_height, target_frame_width

    cap = cv2.VideoCapture(args.video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    print(f"Output video dimensions: {target_frame_width}x{target_frame_height} as the flag resize is set to {args.resize} and rotate is set to {args.rotate}")

    out = cv2.VideoWriter(args.output, fourcc, 30.0, (target_frame_width, target_frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame here
        frame = preprocess_frame(frame, rotate=args.rotate, resize=args.resize, width=target_frame_width, height=target_frame_height)
        

        # Run MoveNet inference
        # keypoints = run_movenet(frame)

        # Post-process the keypoints (e.g., draw them on the frame)
        # frame = draw_keypoints(frame, keypoints)

        # Display the output frame
        if args.live:
            cv2.imshow('Live Pose Estimation', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Write the frame to the output video
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    args = read_args()
    process_video(args)

if __name__ == '__main__':
    main()
