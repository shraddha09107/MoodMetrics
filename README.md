# MoodMetrics

**AI-Powered Retail Emotion Insights**

MoodMetrics detects customer facial emotions from images, recorded video, or a live camera feed, aggregates the results, and uses a Generative AI layer (Google Gemini) to turn raw emotion statistics into plain-language reports and recommendations for store managers.

> JOVAC Program, Generative AI Track | GLA University, B.Tech CSE (AI & ML)

---

## Features

- Emotion detection from **single/multiple images, uploaded video, or live camera**
- Seven emotion classes: happy, sad, angry, surprised, neutral, fearful, disgusted
- Aggregation of emotion data over time with Pandas
- **AI-generated insight report** (executive summary, findings, recommendations); live sessions generate a report automatically on stop
- Flask web dashboard with a dark console theme and an emotion-colour system across charts
- **Privacy by design:** no face recognition or identity matching; raw frames are discarded immediately and only aggregate statistics are kept

## How It Works

```
Image / Video / Camera
        |
Face detection (OpenCV Haar Cascade)
        |
Emotion classification (ViT, Hugging Face Transformers + PyTorch)
        |
Analytics aggregation (Pandas)
        |
AI insight report (Google Gemini API)
        |
Flask dashboard
```

## Project Structure

```
MoodMetrics/
├── src/
│   ├── app.py                 # Flask app and routes
│   ├── emotion_engine.py      # ViT emotion model loading and inference
│   ├── image_processor.py     # Image / batch upload processing
│   ├── video_processor.py     # Video file processing
│   ├── camera_stream.py       # Live camera streaming and sessions
│   ├── aggregator.py          # Emotion statistics aggregation
│   ├── insight_generator.py   # Gemini prompt and report generation
│   ├── logger.py              # Logging utilities
│   ├── colors.py              # Emotion colour mapping
│   ├── templates/
│   │   ├── index.html         # Home / capture page
│   │   └── dashboard.html     # Analytics dashboard
│   └── static/
│       └── uploads/           # Temporary uploads (git-ignored)
├── logs/                      # Runtime logs (git-ignored)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Face detection | OpenCV (Haar Cascade) |
| Emotion model | `dima806/facial_emotions_image_detection` (ViT, ~91% accuracy) |
| Deep learning | PyTorch, Hugging Face Transformers |
| Analytics | Pandas, NumPy |
| GenAI | Google Gemini API |
| Web | Flask, HTML/CSS/JS |

## Getting Started

### Prerequisites

- Python **3.12** (3.13 is not recommended)
- A free Google Gemini API key
- A webcam (only for live mode)

### Installation

```bash
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

> **Windows PowerShell:** if activation is blocked, run
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

### Configuration

Copy `.env.example` to `.env` in the **project root** (next to `requirements.txt`) and add your key:

```
GEMINI_API_KEY=your_api_key_here
```

> Gemini model names change over time. If reports silently fail, check the model name in `insight_generator.py` against the current Gemini API docs.

### Run

```bash
python src/app.py
```

Open `http://127.0.0.1:5000` in your browser. The ViT model downloads automatically on first run, so allow a minute.

## Usage

1. **Image mode:** upload one or more photos.
2. **Video mode:** upload a recorded clip.
3. **Live mode:** start the camera, then stop the session. A full AI report is generated automatically.
4. Open the **dashboard** for emotion trends and the generated insight report.

## Privacy and Ethics

- No identity recognition or face matching
- Raw frames are discarded right after emotion extraction
- Only aggregate, non-identifiable statistics are stored
- Any real deployment needs clear signage and customer consent

## Limitations

- Facial expression is not a perfect proxy for inner emotion; results are indicative, not definitive
- Accuracy depends on lighting, camera angle, and face visibility
- Prototype scale only; not integrated with store CCTV or multi-store setups

## Team

| Name | Institution |
|---|---|
| Shraddha Singh | GLA University |
| Shreya Sharma | GLA University |
| Shreyansh Gupta | GLA University |
| Shraddha Singh | GLA University |

## Branching and Contribution

- `main`: stable, demo-ready code only
- `develop`: integration branch
- Work on `feature/<name>`, `fix/<name>`, or `docs/<name>` and open a Pull Request into `develop`
- Commit style: `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`

## License

Academic project. Add a license of your choice (e.g. MIT) before public reuse.
