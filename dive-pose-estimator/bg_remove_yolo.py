import cv2
import torch
import numpy as np
from torchvision import models
import torchvision.transforms as T
from PIL import Image

# Load YOLO model (assuming YOLOv5 for this example)
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Transformation for DeepLabV3 (if needed for better segmentation)
deeplab_model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
preprocess = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_frame(frame):
    image = Image.fromarray(frame)
    np_image = np.array(image)

    # Detect objects with YOLO, but filter only for 'person' class
    results = yolo_model(np_image)
    boxes = results.xyxy[0].cpu().numpy()

    # Create an empty mask for the entire frame
    mask = np.zeros(np_image.shape[:2], dtype=np.uint8)

    for box in boxes:
        x1, y1, x2, y2, conf, cls = map(int, box[:6])

        # Check if the detected class is "person"
        if cls == 0:  # '0' is the label for 'person' in YOLOv5
            cropped_image = image.crop((x1, y1, x2, y2))

            # Optionally, use DeepLabV3 for more precise segmentation within the bounding box
            input_tensor = preprocess(cropped_image).unsqueeze(0)
            with torch.no_grad():
                output = deeplab_model(input_tensor)['out'][0]
            seg_mask = output.argmax(0).byte().cpu().numpy()

            # Resize the segmentation mask to match the bounding box size
            seg_mask_resized = cv2.resize(seg_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)

            # Place the segmentation mask on the main mask
            mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], seg_mask_resized)

    # Apply the mask to keep only the person(s) and remove the background
    foreground = cv2.bitwise_and(np_image, np_image, mask=mask)
    background = np.zeros_like(np_image)

    # Combine foreground with transparent/black background
    result = np.where(mask[..., None], foreground, background)

    return result