import cv2
def draw_keypoints(frame, keypoints, colors, skeleton_links, bbox):
    # Draw keypoints
    for i, keypoint in enumerate(keypoints):
        x, y = keypoint
        cv2.circle(frame, (int(x), int(y)), 5, colors[i], -1)

    # Draw skeleton
    for start_idx, end_idx in skeleton_links:
        if all(keypoints[start_idx]) and all(keypoints[end_idx]):
            start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
            end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
            cv2.line(frame, start_point, end_point, colors[start_idx], 2)

    # Draw bounding boxes for keypoints
    if len(bbox) != 0:
        for box in bbox:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    return frame