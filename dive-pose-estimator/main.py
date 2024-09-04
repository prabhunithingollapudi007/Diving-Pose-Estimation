import argparse, os
import cv2
import numpy as np
from preprocess import preprocess_frame
from infer import run_movenet, upscale_keypoints
from postprocess import draw_keypoints

MODEL_INPUT_SIZE = (256, 256)

def read_args():
    parser = argparse.ArgumentParser(description='Dive Pose Estimator')
    parser.add_argument('--video', type=str, help='Path to the input video file', required=True)
    parser.add_argument('--output', type=str, help='Path to the output video file', required=True)
    parser.add_argument('--model', type=str, default='../models/movenet_thunder.tflite', help='Path to the model', required=False)
    parser.add_argument('--rotate', type=lambda x: (str(x).lower() == 'true'), choices=[True, False], default=True, help='Rotate the video 90 degrees clockwise.')
    parser.add_argument('--resize', type=lambda x: (str(x).lower() == 'true'), choices=[True, False], default=False, help='Resize the input video.')
    parser.add_argument('--width', type=int, default=192, help='Resize input to specific width.', required=False)
    parser.add_argument('--height', type=int, default=192, help='Resize input to specific height.', required=False)
    parser.add_argument('--live', type=lambda x: (str(x).lower() == 'true'), choices=[True, False], default=False, help='Display the output video in a window.')

    args = parser.parse_args()
    return args

def validate_args(args):
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

    # print the arguments
    print("===== Arguments =====")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("=====================")

def process_video(args):
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

    if args.resize == True:
        target_frame_width = args.width
        target_frame_height = args.height

    if args.rotate == True:
        target_frame_width, target_frame_height = target_frame_height, target_frame_width

    cap = cv2.VideoCapture(args.video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    print(f"Output video dimensions: {target_frame_width}x{target_frame_height} as the flag resize is set to {args.resize} and rotate is set to {args.rotate}")

    out = cv2.VideoWriter(args.output, fourcc, 30.0, (target_frame_width, target_frame_height))

    while cap.isOpened():
        ret, original_frame = cap.read()
        if not ret:
            break
        # Process the frame here
        processed_frame = preprocess_frame(original_frame, rotate=args.rotate, resize=args.resize, width=target_frame_width, height=target_frame_height)

        # Run MoveNet inference
        # By default, the frame is resized to 192x192 pixels (the input size for MoveNet)
        size_reduced_frame = preprocess_frame(processed_frame, rotate=args.rotate, resize=True, width=MODEL_INPUT_SIZE[0], height=MODEL_INPUT_SIZE[1])
        keypoints = run_movenet(size_reduced_frame)
        keypoints = upscale_keypoints(keypoints, target_frame_width, target_frame_height)

        # Post-process the keypoints (e.g., draw them on the frame)
        frame = draw_keypoints(size_reduced_frame, keypoints, threshold=0.2)
        # Display the output frame
        if args.live == True:
            # Concatenate the original frame and the frame with keypoints side by side
            side_by_side_frame = np.hstack((processed_frame, frame))
            cv2.imshow('Live Pose Estimation', side_by_side_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Write the frame to the output video
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    args = read_args()
    validate_args(args)
    process_video(args)

if __name__ == '__main__':
    main()
