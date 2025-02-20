import numpy as np
import cv2
import json

def calculate_angle(a, b, c):
    """Calculate the angle ABC (in degrees) using the cosine rule."""
    a, b, c = np.array(a), np.array(b), np.array(c)

    # Vectors
    AB = a - b
    BC = c - b

    # Dot product & magnitude
    dot_product = np.dot(AB, BC)
    mag_ab = np.linalg.norm(AB)
    mag_bc = np.linalg.norm(BC)

    # Avoid division by zero
    if mag_ab == 0 or mag_bc == 0:
        return 0.0

    # Compute cosine of the angle
    cos_theta = dot_product / (mag_ab * mag_bc)
    
    # Ensure the cosine value is within the valid range [-1, 1]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Compute angle (in degrees)
    theta = np.arccos(cos_theta)
    return np.degrees(theta)

def put_text(frame, angle, joint, position_x, position_y):
    cv2.putText(
            frame,
            f"{joint}: {int(angle)} degrees",
            (position_x, position_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

HEAD = 0
LEFT_SHOULDER = 5
LEFT_ELBOW = 7
LEFT_WRIST = 9
LEFT_HIP = 11
LEFT_KNEE = 13
LEFT_ANKLE = 15

def process_pose_angles(pose_frame, keypoints):
    """Compute and display angles on the given pose frame."""
    keypoints = np.array(keypoints)  # Convert to NumPy array

    # Extract joint positions
    joints = {
        "Torso": (keypoints[HEAD], keypoints[LEFT_SHOULDER], keypoints[LEFT_HIP]),
        "Hip": (keypoints[LEFT_SHOULDER], keypoints[LEFT_HIP], keypoints[LEFT_KNEE]),
        "Knee": (keypoints[LEFT_HIP], keypoints[LEFT_KNEE], keypoints[LEFT_ANKLE]),
        "Arm": (keypoints[LEFT_SHOULDER], keypoints[LEFT_ELBOW], keypoints[LEFT_WRIST]),
    }

    # Define position for text overlay
    text_x = pose_frame.shape[1] - 200  # Right side
    text_y = 30

    # Calculate and draw angles
    for joint, (A, B, C) in joints.items():
        angle = calculate_angle(A, B, C)
        put_text(pose_frame, angle, joint, text_x, text_y)
        text_y += 30

    return pose_frame
