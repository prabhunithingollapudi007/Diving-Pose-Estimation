import cv2
import tensorflow as tf

def rotate_frame(frame):
    return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

def preprocess_frame(frame, rotate, resize, width, height):

    # Rotate the frame 90 degrees clockwise
    if rotate:
        frame = rotate_frame(frame)
        
    # Resize and pad the frame to 192x192, which is the input size for MoveNet
    if resize:
        frame = tf.image.resize_with_pad(frame, target_height=height, target_width=width)
        frame = frame.numpy()

    return frame