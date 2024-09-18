import cv2
import numpy as np


RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 0, 0)
BLACK_COLOR = (0, 0, 0)

def draw_keypoints(frame, keypoints, threshold=0.2):
    copied_frame = np.copy(frame)
    keypoints = np.squeeze(keypoints)

    # Draw only joints with confidence above the threshold
    connections = [
        ('nose', 'left shoulder'), ('left shoulder', 'left elbow'), ('left elbow', 'left wrist'),
        ('nose', 'right shoulder'), ('right shoulder', 'right elbow'), ('right elbow', 'right wrist'),
        ('left shoulder', 'left hip'), ('right shoulder', 'right hip'), ('left hip', 'right hip'),
        ('left hip', 'left knee'), ('right hip', 'right knee'), ('left knee', 'left ankle'), ('right knee', 'right ankle')
    ]

    label = ["nose", "left eye", "right eye", "left ear", "right ear", "left shoulder", "right shoulder", "left elbow", "right elbow", "left wrist", "right wrist", "left hip", "right hip", "left knee", "right knee", "left ankle", "right ankle"]
    
    facial_keypoints = ["nose", "left eye", "right eye", "left ear", "right ear"]

    right_hand = ["right wrist", "right elbow", "right shoulder"]
    left_hand = ["left wrist", "left elbow", "left shoulder"]
    right_leg = ["right ankle", "right knee", "right hip"]
    left_leg = ["left ankle", "left knee", "left hip"]

    # Draw keypoints on the copied_frame
    dot_color = GREEN_COLOR

    for i in range(len(label)):
        confidence = keypoints[i][2]
        if confidence > threshold:
            x = int(keypoints[i][1])
            y = int(keypoints[i][0])
            cv2.circle(copied_frame, (x, y), 4, dot_color, -1)


    lines = [(label.index(start), label.index(end)) for start, end in connections]
    for line in lines:
        start = line[0]
        end = line[1]
        if keypoints[start][2] > threshold and keypoints[end][2] > threshold:
            start = (int(keypoints[start][1]), int(keypoints[start][0]))
            end = (int(keypoints[end][1]), int(keypoints[end][0]))
            
            
            if label[line[0]] in facial_keypoints and label[line[1]] in facial_keypoints:
                cv2.line(copied_frame, start, end, RED_COLOR, 2)
            elif label[line[0]] in right_hand and label[line[1]] in right_hand:
                cv2.line(copied_frame, start, end, BLUE_COLOR, 2)
            elif label[line[0]] in left_hand and label[line[1]] in left_hand:
                cv2.line(copied_frame, start, end, BLUE_COLOR, 2)
            elif label[line[0]] in right_leg and label[line[1]] in right_leg:
                cv2.line(copied_frame, start, end, GREEN_COLOR, 2)
            elif label[line[0]] in left_leg and label[line[1]] in left_leg:
                cv2.line(copied_frame, start, end, GREEN_COLOR, 2)
            else:
                cv2.line(copied_frame, start, end, BLACK_COLOR, 2)

    return copied_frame
