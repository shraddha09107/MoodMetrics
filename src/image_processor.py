

import cv2
from emotion_engine import analyze_frame
from logger import log_detection


def process_image(image_path: str, zone: str = "uploaded_image", annotated_out_path: str = None):
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not read image: {image_path}")

    annotated, detections = analyze_frame(frame, zone)
    for d in detections:
        log_detection("image", d["zone"], d["emotion"], d["confidence"])

    if annotated_out_path:
        cv2.imwrite(annotated_out_path, annotated)

    return detections
