

import os
import pandas as pd

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "emotion_log.csv")


def load_log() -> pd.DataFrame:
    if not os.path.exists(LOG_PATH):
        raise FileNotFoundError(
            f"No log found at {LOG_PATH}. Analyze an image, video, or webcam session first."
        )
    df = pd.read_csv(LOG_PATH, parse_dates=["timestamp"])
    return df


def overall_distribution(df: pd.DataFrame) -> pd.Series:
    return (df["emotion"].value_counts(normalize=True) * 100).round(1)


def zone_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return (pd.crosstab(df["zone"], df["emotion"], normalize="index") * 100).round(1)


def source_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Breakdown by input source: image / video / webcam."""
    return (pd.crosstab(df["source"], df["emotion"], normalize="index") * 100).round(1)


def hourly_trend(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["timestamp"].dt.floor("h")
    return df.groupby(["hour", "emotion"]).size().unstack(fill_value=0)


def build_summary_dict(df: pd.DataFrame) -> dict:
    return {
        "total_detections": len(df),
        "overall_distribution": overall_distribution(df).to_dict(),
        "zone_distribution": zone_distribution(df).to_dict(orient="index"),
        "source_distribution": source_distribution(df).to_dict(orient="index"),
        "date_range": {
            "start": str(df["timestamp"].min()),
            "end": str(df["timestamp"].max()),
        },
    }


if __name__ == "__main__":
    df = load_log()
    print("Overall emotion distribution (%):\n", overall_distribution(df))
    print("\nZone-wise distribution (%):\n", zone_distribution(df))
    print("\nSource-wise distribution (%):\n", source_distribution(df))
