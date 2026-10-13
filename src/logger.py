"""
Shared logger used by every input mode. Appends one row per detected face
to logs/emotion_log.csv, tagged with which input source it came from
(image / video / webcam) so the dashboard can break stats down that way too.
"""

import csv
import os
from datetime import datetime

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "emotion_log.csv")


def ensure_log_file():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "source", "zone", "emotion", "confidence"])


def log_detection(source: str, zone: str, emotion: str, confidence: float):
    ensure_log_file()
    with open(LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow(
            [datetime.now().isoformat(timespec="seconds"), source, zone, emotion, round(confidence, 2)]
        )
