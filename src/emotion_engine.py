

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

MODEL_NAME = "dima806/facial_emotions_image_detection"

print(f"[emotion_engine] Loading model '{MODEL_NAME}' "
      f"(first run downloads it once, then it's cached locally)...")
_classifier = pipeline("image-classification", model=MODEL_NAME)
print("[emotion_engine] Model loaded.")


def detect_faces(frame: np.ndarray):
    """Stage 2: Face Detection. Returns a list of (x, y, w, h) boxes."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    return faces


def classify_emotion(face_crop_bgr: np.ndarray):
    """Stage 3: Emotion Classification via the Hugging Face pipeline.
    face_crop_bgr must be a face already cropped out of an OpenCV (BGR) frame.
    Returns (label: str, confidence: float 0-100)."""
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        return None, 0.0
    rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    results = _classifier(pil_img)  # sorted list of {"label":..., "score":...}
    top = results[0]
    return top["label"], round(top["score"] * 100, 1)


def analyze_frame(frame: np.ndarray, zone: str = "default"):
    """Runs Stage 2 + Stage 3 on one frame. Draws boxes + labels onto the
    frame in place, and returns (annotated_frame, detections list) where
    each detection is {"zone", "emotion", "confidence"}."""
    faces = detect_faces(frame)
    detections = []
    for (x, y, w, h) in faces:
        crop = frame[y:y + h, x:x + w]
        label, confidence = classify_emotion(crop)
        if label is None:
            continue
        detections.append({"zone": zone, "emotion": label, "confidence": confidence})
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame, f"{label} ({confidence:.0f}%)", (x, max(y - 10, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
    return frame, detections
