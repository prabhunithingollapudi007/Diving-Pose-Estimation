import cv2
import numpy as np
from config import MAX_BBOX_DISTANCE, MIN_BBOX_AREA, COLORS, MIN_BBOX_START_HEIGHT_LOWER_LIMIT, MIN_BBOX_START_HEIGHT_UPPER_LIMIT

def draw_keypoints(frame, keypoints, skeleton_links, bbox):
    # Draw skeleton
    for start_idx, end_idx in skeleton_links:
        if all(keypoints[start_idx]) and all(keypoints[end_idx]):
            start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
            end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
            cv2.line(frame, start_point, end_point, COLORS[start_idx], 2)

    # Draw bounding boxes for keypoints
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    return frame

def bbox_distance(bbox1, bbox2):
    """Compute Euclidean distance between bounding box centers (x1, y1, x2, y2 format)."""
    # Unpack bbox
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Compute center points
    center1 = np.array([(x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2])
    center2 = np.array([(x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2])

    return np.linalg.norm(center1 - center2)


def is_bbox_valid(bbox):
    """Check if the bbox is too small or too big."""

    # Compute bbox area
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)

    # Skip if bbox is too small
    if area < MIN_BBOX_AREA:
        return False
    return True  # Valid bbox

def is_next_bbox_valid(bbox, previous_bbox):
    
    if not is_bbox_valid(bbox):
        return False
    
    if not is_bbox_valid(previous_bbox):
        return False

    # Skip if bbox is too far from the previous bbox
    dist = bbox_distance(previous_bbox, bbox)
    if dist > MAX_BBOX_DISTANCE:
        return False

    return True

def is_bbox_in_center(bbox, frame_height):
    """Check if the bbox is in the center of the frame."""
    # Compute bbox center
    _, y1, _, y2 = bbox
    center = (y1 + y2) / 2

    # Return True if bbox is in the center
    return MIN_BBOX_START_HEIGHT_LOWER_LIMIT * frame_height < center < MIN_BBOX_START_HEIGHT_UPPER_LIMIT * frame_height