import cv2
import numpy as np

def draw_keypoints(frame, keypoints, threshold=0.2):
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

    # Draw keypoints on the frame
    for i in range(len(label)):
        dot_color = (0, 255, 0)
        if label[i] in facial_keypoints:
            dot_color = (0, 0, 255)
        confidence = keypoints[i][2]
        if confidence > threshold:
            x = int(keypoints[i][1])
            y = int(keypoints[i][0])
            cv2.circle(frame, (x, y), 4, dot_color, -1)


    lines = [(label.index(start), label.index(end)) for start, end in connections]
    for line in lines:
        start = line[0]
        end = line[1]
        if keypoints[start][2] > threshold and keypoints[end][2] > threshold:
            start = (int(keypoints[start][1]), int(keypoints[start][0]))
            end = (int(keypoints[end][1]), int(keypoints[end][0]))
            cv2.line(frame, start, end, (0, 255, 0), 2)

    return frame
