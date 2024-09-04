# Use this to convert images to video
import os
import cv2

def image_to_video(image_folder, video_path, fps=30):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape

    # Use 'XVID' or 'MP4V' codec for MP4 format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
    print(f"Video saved to {video_path}")

# Example usage
image_folder = "../data/raw/ideal-test"
video_path = "../data/raw/ideal-test.mp4"

image_to_video(image_folder, video_path)