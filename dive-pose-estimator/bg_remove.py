import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)


def remove_background(frame):
    # Process the image and get the segmentation mask
    results = selfie_segmentation.process(frame)
    mask = results.segmentation_mask

    # Convert the mask to a binary mask
    mask = np.stack((mask,) * 3, axis=-1)
    mask = (mask > 0.1).astype(np.uint8)

    # dilate the mask to remove noise
    kernel = np.ones((300, 300), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Apply the mask to the original frame
    frame = frame * mask

    return frame