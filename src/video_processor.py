"""
Input Mode 2: Uploaded Video File.

Reads the video frame-by-frame but only runs face detection + emotion
classification every `sample_every` frames (classifying every single frame
of a video is slow and unnecessary — emotions don't change frame-to-frame).
Every classified face gets logged, tagged with the current timestamp inside
the video (in seconds) as its "zone" context if none is given.
"""

import cv2
from emotion_engine import analyze_frame
from logger import log_detection


def process_video(video_path: str, zone: str = "uploaded_video", sample_every: int = 15):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_idx = 0
    frames_analyzed = 0
    all_detections = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % sample_every == 0:
            _, detections = analyze_frame(frame, zone)
            for d in detections:
                log_detection("video", d["zone"], d["emotion"], d["confidence"])
            all_detections.extend(detections)
            frames_analyzed += 1

        frame_idx += 1

    cap.release()

    return {
        "total_frames": frame_idx,
        "frames_analyzed": frames_analyzed,
        "duration_seconds": round(frame_idx / fps, 1) if fps else None,
        "detections": all_detections,
    }
