import numpy as np
import tensorflow as tf, tensorflow_hub as hub
import shutil, os

# Check TensorFlow and TensorFlow Hub versions
print(f"TensorFlow version: {tf.__version__}")
print(f"TensorFlow Hub version: {hub.__version__}")

# Load the MoveNet model
try:
    movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
    movenet_fn = movenet.signatures['serving_default']
except ValueError as e:
    print(f"Error loading model: {e}")
    # Handle the error, e.g., by exiting or using a fallback model
    exit(1)


def run_movenet(frame):
    # Perform inference using MoveNet
    # Convert frame to tensor if it isn't already
    if not isinstance(frame, tf.Tensor):
        frame = tf.convert_to_tensor(frame, dtype=tf.int32)
    
    # Expand dimensions to add batch dimension
    frame = tf.expand_dims(frame, axis=0)
    
    outputs = movenet_fn(frame)
    keypoints = outputs['output_0'].numpy()
    return keypoints

def upscale_keypoints(keypoints, width, height):
    keypoints = np.squeeze(keypoints)
    keypoints[:, 1] *= width
    keypoints[:, 0] *= height
    return keypoints
