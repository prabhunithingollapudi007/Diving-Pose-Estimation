import numpy as np
import cv2
from filtering import gaussian_filter
from config import STAGES, FILTER_SIGMA

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


def calculate_orientation(a, b):
    """Compute the absolute torso angle using atan2."""
    delta_x = b[0] - a[0]
    delta_y = b[1] - a[1]
    return np.degrees(np.arctan2(delta_y, delta_x))

def put_text(frame, text, label, position_x, position_y):
    """Overlay text on frame."""
    cv2.putText(
            frame,
            f"{label}: {int(text)} degrees",
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

def compute_total_rotation(torso_angles):
    """Compute cumulative rotation angle from torso orientation over time."""
    total_rotation = 0
    for i in range(1, len(torso_angles)):
        delta_angle = torso_angles[i] - torso_angles[i - 1]

        # Handle angle wrapping (e.g., crossing -180° to 180°)
        if delta_angle > 180:
            delta_angle -= 360
        elif delta_angle < -180:
            delta_angle += 360

        total_rotation += abs(delta_angle)
    
    return total_rotation

def process_pose_angles(pose_frame, keypoints, torso_angles):
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

    joint_angle = {}

    # Calculate and draw angles
    for joint, (A, B, C) in joints.items():
        angle = calculate_angle(A, B, C)
        joint_angle[joint] = angle
        put_text(pose_frame, angle, joint, text_x, text_y)
        text_y += 30

    # Compute current torso orientation
    current_torso_angle = calculate_orientation(keypoints[LEFT_SHOULDER], keypoints[LEFT_HIP])

    # Append torso angle before computing total rotation
    torso_angles.append(current_torso_angle)

    # Compute total rotation angle
    total_rotation = compute_total_rotation(torso_angles)

    # Display center of mass
    com_x = (keypoints[LEFT_HIP][0] + keypoints[LEFT_SHOULDER][0] ) / 2
    com_y = (keypoints[LEFT_HIP][1] + keypoints[LEFT_SHOULDER][1] ) / 2
    com = (com_x, com_y)
    cv2.circle(pose_frame, (int(com[0]), int(com[1])), 10, (255, 255, 255), -1)

    # max y value
    # return the com_y value
    max_y = com_y

    # Display rotation information
    put_text(pose_frame, current_torso_angle, "Current Torso", text_x, text_y)
    text_y += 30
    put_text(pose_frame, total_rotation, "Total Rotation", text_x, text_y)

    return pose_frame, joint_angle, torso_angles, com, max_y

def get_all_filtered_metrics(com, total_rotation, diver_max_y_over_time):
    """Apply Gaussian filtering to all metrics."""
    com_x = [point[0] for point in com]
    com_y = [point[1] for point in com]

    filtered_com_y = gaussian_filter(com_y, FILTER_SIGMA)
    filtered_com_x = gaussian_filter(com_x, FILTER_SIGMA)
    filtered_total_rotation = gaussian_filter(total_rotation, FILTER_SIGMA)
    
    velocity_y = np.diff(filtered_com_y)
    filtered_velocity_y = gaussian_filter(velocity_y, FILTER_SIGMA)
    acceleration_y = np.diff(filtered_velocity_y)
    filtered_acceleration_y = gaussian_filter(acceleration_y, FILTER_SIGMA)

    rotation_rate = np.diff(filtered_total_rotation)
    filtered_rotation_rate = gaussian_filter(rotation_rate, FILTER_SIGMA)
    rotation_acceleration = np.diff(filtered_rotation_rate)
    filtered_rotation_acceleration = gaussian_filter(rotation_acceleration, FILTER_SIGMA)

    diver_max_y_over_time = gaussian_filter(diver_max_y_over_time, FILTER_SIGMA)
    
    return filtered_com_x, filtered_com_y, filtered_total_rotation, filtered_velocity_y, filtered_acceleration_y, filtered_rotation_rate, filtered_rotation_acceleration, diver_max_y_over_time


def detect_stages(joint_angles, torso_angles, diver_heights, total_rotation_over_time, output_video_path, output_base_path, stages):
    """Detects dive stages sequentially and returns a list of frame indices where transitions occur."""
    
    stage_indices = {}  # Store the frame index for each stage
    current_stage = 0

    capture = cv2.VideoCapture(output_video_path)
    max_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_count = 0

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        if frame_count >= max_frames:
            break

        # 1. Absprung (Takeoff) - total rotation angle is over 90 degrees and knee angle is over 150 degrees
        if (current_stage == 0) and (total_rotation_over_time[frame_count] >= 90):
            if joint_angles["Knee"][frame_count] >= 150:
                stage_indices[stages[current_stage]] = frame_count
                # Save this frame
                cv2.imwrite(f"{output_base_path}/stage_{stages[current_stage]}.png", frame)
                current_stage += 1

        # 2. Ansatz (Approach/Entry) - Significant rotation change
        if (current_stage == 1) and (torso_angles[frame_count] < 0):
            stage_indices[stages[current_stage]] = frame_count
            cv2.imwrite(f"{output_base_path}/stage_{stages[current_stage]}.png", frame)
            current_stage += 1

        # 3. Beginn Streckung (Start of Extension) - Rotation rate sign change
        if (current_stage == 2) and (torso_angles[frame_count - 1] <= 0) and (torso_angles[frame_count] > 0):
            if((180 - abs(torso_angles[frame_count - 1])) <= 20) and ((180 - abs(torso_angles[frame_count])) <= 20):
                stage_indices[stages[current_stage]] = frame_count
                cv2.imwrite(f"{output_base_path}/stage_{stages[current_stage]}.png", frame)
                current_stage += 1


        # 4. Ende Streckung (End of Extension) - Rotation acceleration stabilizes
        if (current_stage == 3) and (abs(90 - torso_angles[frame_count]) <= 8):
            if (abs(180 - joint_angles["Knee"][frame_count]) <= 8) and (abs(90 - joint_angles["Hip"][frame_count]) <= 8):
                stage_indices[stages[current_stage]] = frame_count
                cv2.imwrite(f"{output_base_path}/stage_{stages[current_stage]}.png", frame)
                current_stage += 1

        # 5. Eintauchen (Entry) - Peak downward velocity
        if (current_stage == 4) and (abs(90 + torso_angles[frame_count]) <= 5):
            stage_indices[stages[current_stage]] = frame_count
            cv2.imwrite(f"{output_base_path}/stage_{stages[current_stage]}.png", frame)
            break

        frame_count += 1

    capture.release()
    return stage_indices