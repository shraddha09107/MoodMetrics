"""
Input Mode 3: Live Webcam (real-time).

A generator that yields JPEG-encoded frames in the multipart format Flask
needs for an MJPEG stream. Classification only runs every `classify_every`
frames to keep the feed smooth (the HF model is the slow part; drawing
boxes is fast) — the bounding box + label just persists on-screen between
classification passes so it doesn't flicker.
"""

import cv2
from emotion_engine import analyze_frame
from logger import log_detection


def gen_camera_frames(zone: str = "live_camera", camera_index: int = 0, classify_every: int = 8):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try a different camera index.")

    frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1

            if frame_count % classify_every == 0:
                frame, detections = analyze_frame(frame, zone)
                for d in detections:
                    log_detection("webcam", d["zone"], d["emotion"], d["confidence"])

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
    finally:
        cap.release()
