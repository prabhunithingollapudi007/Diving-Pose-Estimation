import cv2

def rotate_frame(frame):
    return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

def resize_frame(frame, width, height):
    return cv2.resize(frame, (width, height))

def process_frame(frame, width, height):
    frame = rotate_frame(frame)
    # frame = resize_frame(frame, width, height)
    return frame