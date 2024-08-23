import numpy as np
import tensorflow as tf, tensorflow_hub as hub
import cv2

# Load the MoveNet model
movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

def run_movenet(frame):
    # Perform inference using MoveNet
    outputs = movenet(frame)
    keypoints = outputs['output_0'].numpy()
    return keypoints
""" 
def run_movenet(frame, model_path):
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get the expected input shape
    input_shape = input_details[0]['shape']

    # Resize the frame to the expected input shape
    resized_frame = cv2.resize(frame, (input_shape[2], input_shape[1]))

    # Test the model on random input data.
    input_data = np.array(resized_frame, dtype=np.uint8)
    input_data = np.expand_dims(input_data, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores """