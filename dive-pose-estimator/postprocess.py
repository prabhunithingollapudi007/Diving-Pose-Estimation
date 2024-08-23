import cv2
import numpy as np

def draw_keypoints(frame, keypoints, threshold=0.2):
    h, w, _ = frame.shape
    keypoints = np.squeeze(keypoints)

    # Draw keypoints on the frame
    for i in range(17):
        confidence = keypoints[i][2]
        if confidence > threshold:
            x = int(keypoints[i][1] * w)
            y = int(keypoints[i][0] * h)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
    
    return frame
