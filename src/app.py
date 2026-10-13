

import os
import json
import uuid
from datetime import datetime

import pandas as pd
from flask import Flask, render_template, request, jsonify, Response, url_for
import plotly
import plotly.express as px

from image_processor import process_image
from video_processor import process_video
from camera_stream import gen_camera_frames
from colors import EMOTION_COLORS, color_for
from aggregator import (
    load_log,
    overall_distribution,
    zone_distribution,
    source_distribution,
    hourly_trend,
    build_summary_dict,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB upload limit for videos


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze_image", methods=["POST"])
def analyze_image():
    """Accepts one or more images under the same 'image' field (the <input>
    has the `multiple` attribute). Each is processed independently, then an
    aggregate summary across the whole batch is returned alongside the
    per-photo results."""
    files = request.files.getlist("image")
    zone = request.form.get("zone", "uploaded_image")
    files = [f for f in files if f and f.filename]
    if not files:
        return jsonify({"error": "No image uploaded"}), 400

    results = []
    emotion_counts = {}

    for file in files:
        ext = os.path.splitext(file.filename)[1] or ".jpg"
        uid = uuid.uuid4().hex[:8]
        in_path = os.path.join(UPLOAD_DIR, f"{uid}_input{ext}")
        out_path = os.path.join(UPLOAD_DIR, f"{uid}_annotated.jpg")
        file.save(in_path)

        try:
            detections = process_image(in_path, zone=zone, annotated_out_path=out_path)
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e), "detections": []})
            continue

        for d in detections:
            emotion_counts[d["emotion"]] = emotion_counts.get(d["emotion"], 0) + 1

        results.append({
            "filename": file.filename,
            "detections": detections,
            "annotated_image_url": url_for("static", filename=f"uploads/{uid}_annotated.jpg"),
        })

    total_faces = sum(len(r.get("detections", [])) for r in results)

    return jsonify({
        "results": results,
        "batch_summary": {
            "photos": len(results),
            "faces_detected": total_faces,
            "emotion_counts": emotion_counts,
        },
    })


@app.route("/analyze_video", methods=["POST"])
def analyze_video():
    file = request.files.get("video")
    zone = request.form.get("zone", "uploaded_video")
    if not file:
        return jsonify({"error": "No video uploaded"}), 400

    ext = os.path.splitext(file.filename)[1] or ".mp4"
    uid = uuid.uuid4().hex[:8]
    in_path = os.path.join(UPLOAD_DIR, f"{uid}_video{ext}")
    file.save(in_path)

    try:
        summary = process_video(in_path, zone=zone)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "total_frames": summary["total_frames"],
        "frames_analyzed": summary["frames_analyzed"],
        "duration_seconds": summary["duration_seconds"],
        "faces_detected": len(summary["detections"]),
    })


@app.route("/video_feed")
def video_feed():
    zone = request.args.get("zone", "live_camera")
    camera_index = int(request.args.get("camera_index", 0))
    return Response(
        gen_camera_frames(zone=zone, camera_index=camera_index),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/dashboard")
def dashboard():
    try:
        df = load_log()
    except FileNotFoundError as e:
        return render_template("dashboard.html", no_data=True, message=str(e))

    # Overall distribution -> both the pie chart and the CSS "mood bar" hero
    dist = overall_distribution(df).reset_index()
    dist.columns = ["emotion", "percent"]
    dist = dist.sort_values("percent", ascending=False)

    mood_bar = [
        {"emotion": row.emotion, "percent": row.percent, "color": color_for(row.emotion)}
        for row in dist.itertuples()
    ]

    pie_fig = px.pie(
        dist, names="emotion", values="percent", hole=0.6,
        color="emotion", color_discrete_map=EMOTION_COLORS,
    )
    pie_fig.update_traces(textfont_color="#14161C", marker=dict(line=dict(color="#14161C", width=2)))
    pie_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#EDEAE3", family="Inter, sans-serif", size=13),
        showlegend=True, legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        margin=dict(t=10, b=10, l=10, r=10),
    )
    pie_json = json.dumps(pie_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Trend line — melt to long form so color_discrete_map matches by emotion name
    trend = hourly_trend(df)
    trend_long = trend.reset_index().melt(id_vars="hour", var_name="emotion", value_name="count")
    line_fig = px.line(
        trend_long, x="hour", y="count", color="emotion", markers=True,
        color_discrete_map=EMOTION_COLORS,
    )
    line_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#EDEAE3", family="Inter, sans-serif", size=13),
        xaxis=dict(gridcolor="#2B2F3A", zerolinecolor="#2B2F3A"),
        yaxis=dict(gridcolor="#2B2F3A", zerolinecolor="#2B2F3A"),
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
        margin=dict(t=10, b=10, l=10, r=10),
    )
    line_json = json.dumps(line_fig, cls=plotly.utils.PlotlyJSONEncoder)

    zdist = zone_distribution(df).reset_index()
    sdist = source_distribution(df).reset_index()

    raw = df.tail(50).sort_values("timestamp", ascending=False)
    raw_table = raw.to_dict(orient="records")
    for row in raw_table:
        row["color"] = color_for(row["emotion"])

    return render_template(
        "dashboard.html",
        no_data=False,
        mood_bar=mood_bar,
        pie_json=pie_json,
        line_json=line_json,
        zone_table=zdist.to_dict(orient="records"),
        zone_columns=list(zdist.columns),
        source_table=sdist.to_dict(orient="records"),
        source_columns=list(sdist.columns),
        raw_table=raw_table,
        total_detections=len(df),
    )


@app.route("/server_time")
def server_time():
    """Returns the server's current time so the browser can mark exactly
    when a live camera session started — using the same clock the detection
    logger uses avoids timezone mismatches when filtering the log later."""
    return jsonify({"now": datetime.now().isoformat(timespec="seconds")})


def _filtered_log(since: str = None, zone: str = None, source: str = None) -> pd.DataFrame:
    df = load_log()
    if since:
        since_ts = pd.to_datetime(since)
        df = df[df["timestamp"] >= since_ts]
    if zone:
        df = df[df["zone"] == zone]
    if source:
        df = df[df["source"] == source]
    return df


@app.route("/session_summary")
def session_summary():

    since = request.args.get("since")
    zone = request.args.get("zone")
    source = request.args.get("source")
    try:
        df = _filtered_log(since, zone, source)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400

    if df.empty:
        return jsonify({"total": 0, "distribution": {}, "top_emotion": None})

    dist = overall_distribution(df).to_dict()
    top_emotion = max(dist, key=dist.get) if dist else None
    return jsonify({"total": len(df), "distribution": dist, "top_emotion": top_emotion})


@app.route("/generate_session_report")
def generate_session_report():

    since = request.args.get("since")
    zone = request.args.get("zone")
    source = request.args.get("source")
    try:
        df = _filtered_log(since, zone, source)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400

    if df.empty:
        return jsonify({"error": "No detections logged in this session yet."}), 400

    try:
        from insight_generator import generate_insight_report
        summary = build_summary_dict(df)
        report = generate_insight_report(summary)
        return jsonify({"report": report})
    except Exception as e:
        return jsonify({"error": f"Could not generate report: {e}"}), 500


@app.route("/generate_report")
def generate_report():
    try:
        df = load_log()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400

    try:
        from insight_generator import generate_insight_report
        summary = build_summary_dict(df)
        report = generate_insight_report(summary)
        return jsonify({"report": report})
    except Exception as e:
        return jsonify({"error": f"Could not generate report: {e}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)