"""
Stage 5: Generative AI Insight Layer — using Google's Gemini API (free tier).

Gemini's free tier requires no credit card and no billing setup, which is
why it's used here instead of a paid API. Rate limits apply (a few dozen
requests per minute, a daily cap) but that's more than enough for this
project's usage pattern (one report generated on demand from the dashboard).

Setup:
  1. Go to https://aistudio.google.com/apikey
  2. Sign in with any Google account, click "Create API key" — no card needed
  3. Create a .env file in the project root with:
         GEMINI_API_KEY=AIza...
  4. pip install google-genai python-dotenv (already in requirements.txt)
"""

import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

MODEL = "gemini-2.5-flash"  # fast, capable, and on Gemini's free tier

PROMPT_TEMPLATE = """You are a retail analytics assistant. Below is aggregated,
anonymized customer emotion data collected from images, video, and/or live
camera analysis (no identity data, only emotion percentages by zone, source,
and time).

Data:
{data}

Write a detailed report for a store manager with these clearly labeled sections:

1. Executive Summary — at least 3 full sentences (never fewer) covering the overall
   mood, how confident this reading is given the sample size, and what it means for
   the business. Do not compress this into one or two sentences.
2. Zone & Time Patterns — call out any zone, time-of-day, or input-source trends worth
   flagging. If the sample size is small, say so plainly rather than overstating confidence
   in the pattern.
3. Recommendations — 3-4 concrete, actionable steps a store manager could take this week.

Keep it plain-language, no jargon. Use short paragraphs or bullet points under each
heading so it's skimmable. Aim for 220-320 words — thorough, but not padded.
"""


def generate_insight_report(summary: dict) -> str:
    prompt = PROMPT_TEMPLATE.format(data=json.dumps(summary, indent=2))
    response = client.models.generate_content(model=MODEL, contents=prompt)
    return response.text


if __name__ == "__main__":
    from aggregator import load_log, build_summary_dict

    df = load_log()
    summary = build_summary_dict(df)
    report = generate_insight_report(summary)
    print(report)