EMOTION_COLORS = {
    "angry": "#E5484D",
    "disgust": "#8BAA3C",
    "fear": "#8B5CF6",
    "happy": "#F5A623",
    "neutral": "#9096A2",
    "sad": "#4A90D9",
    "surprise": "#E85D9E",
}

DEFAULT_COLOR = "#9096A2"


def color_for(emotion: str) -> str:
    return EMOTION_COLORS.get(str(emotion).lower(), DEFAULT_COLOR)
