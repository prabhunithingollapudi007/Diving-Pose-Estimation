import cv2
import torch
from pathlib import Path

# command to run the script:
# python yolo_detection.py --source ../data/preprocessed/Elena_rotated.mp4 --output-dir ../data/preprocessed/Elena_yolo --model yolov5s --confidence 0.5 --device cpu

# Load YOLOv5 model
# Make sure you have `torch.hub` properly set up or install YOLOv5 locally.
def load_yolo_model(model_name="yolov5s", device="cuda"):
    print("Loading YOLO model...")
    model = torch.hub.load("ultralytics/yolov5", model_name, device=device)
    return model

# Detect persons in an image or video
def detect_persons(model, source, output_dir="detections", confidence=0.5):
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving detections to {output_dir}")

    # Check if source is a video or an image
    is_video = True if source.endswith(".mp4") else False
    print(f"Processing {'video' if is_video else 'image'} at {source}")
    cap = cv2.VideoCapture(source)


    # Initialize VideoWriter if processing a video
    if is_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi
        output_video_path = output_dir / "output_video.mp4"
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))


    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video processing complete.")
            break

        # Run YOLO inference
        results = model(frame)
        detections = results.xyxy[0]  # Extract bounding boxes

        # Filter detections for persons (class_id = 0 in COCO for "person")
        person_detections = [
            {
                "x1": int(box[0]),
                "y1": int(box[1]),
                "x2": int(box[2]),
                "y2": int(box[3]),
                "confidence": float(box[4]),
            }
            for box in detections if int(box[5]) == 0 and float(box[4]) > confidence
        ]

        """ # Save detections
        save_path = output_dir / f"detections_frame_{frame_id}.json"
        with open(save_path, "w") as f:
            import json
            json.dump(person_detections, f) """

        # Annotate frame with detections
        for det in person_detections:
            cv2.rectangle(
                frame,
                (det["x1"], det["y1"]),
                (det["x2"], det["y2"]),
                (0, 255, 0),
                2,
            )
            """ cv2.putText(
                frame,
                f"Person {det['confidence']:.2f}",
                (det["x1"], det["y1"] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            ) """

        # Save or display annotated frame
        out.write(frame)
        """ output_image_path = output_dir / f"frame_{frame_id}.jpg"
        cv2.imwrite(str(output_image_path), frame) """
        frame_id += 1

        # If not a video, break after processing the single image
        if not is_video:
            break

    if is_video:
        print(f"Processed {frame_id} frames.")
        cap.release()
        out.release()
    print(f"Detections saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    # Command-line arguments
    parser = argparse.ArgumentParser(description="YOLOv5 Person Detection Script")
    parser.add_argument(
        "--source", type=str, required=True, help="Path to the input image or video"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="detections",
        help="Directory to save detections and annotated images",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov5s",
        help="YOLO model type (e.g., yolov5s, yolov5m, yolov5l)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for detection",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run inference (cuda or cpu)"
    )

    args = parser.parse_args()

    # Load model and run detection
    yolo_model = load_yolo_model(args.model, args.device)
    detect_persons(yolo_model, args.source, args.output_dir, args.confidence)
