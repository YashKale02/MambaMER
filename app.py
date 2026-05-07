"""
app.py — MambaMER Web Interface
================================
Flask-based web frontend for the MambaMER inference pipeline.

Usage:
    python3 app.py                     (or with your mamba_env)
    /home/yashkale/mamba_env/bin/python3 app.py

Then open http://localhost:5000 in your browser.
"""

import os
import sys
import json
import numpy as np
import torch
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify, send_from_directory
from models import MultimodalEmotionModel
from lyrics import LyricsWindowEncoder

# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "inference_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

PATHS = {
    "model":       os.path.join(BASE_DIR, "best_model.pt"),
    "stats":       os.path.join(BASE_DIR, "global_stats.npz"),
    "lexicon":     os.path.join(BASE_DIR, "NRC-VAD-Lexicon-v2.1.txt"),
    "features":    "/home/yashkale/MER_VER3/PMEmo2019/features/dynamic_features.csv",
    "annotations": "/home/yashkale/MER_VER3/PMEmo2019/annotations/dynamic_annotations.csv",
    "lyrics_dir":  "/home/yashkale/MER_VER3/PMEmo2019/lyrics",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB max upload

# ============================================================
# Lazy-loaded singletons
# ============================================================

_model = None
_encoder = None
_model_lock = threading.Lock()
_features_df = None


def get_model():
    global _model
    if _model is None:
        print("Loading MambaMER model...")
        _model = MultimodalEmotionModel().to(DEVICE)
        
        # Load the checkpoint
        state_dict = torch.load(PATHS["model"], map_location=DEVICE, weights_only=True)
        
        # Load checkpoint — shapes now match (max_len=500 in both model and checkpoint)
        _model.load_state_dict(state_dict, strict=True)
        
        _model.eval()
        print("Model loaded successfully.")
    return _model


def get_encoder():
    global _encoder
    if _encoder is None:
        print("Loading DeBERTa lyrics encoder...")
        _encoder = LyricsWindowEncoder(
            lexicon_path=PATHS["lexicon"],
            device=DEVICE,
        )
        _encoder.eval()
        print("Lyrics encoder loaded.")
    return _encoder


def get_features_df():
    global _features_df
    if _features_df is None:
        import pandas as pd
        print("Loading feature CSV...")
        _features_df = pd.read_csv(PATHS["features"])
        _features_df = _features_df.sort_values(["musicId", "frameTime"])
        _features_df["musicId"] = _features_df["musicId"].astype(int)
        print(f"Features loaded: {len(_features_df)} rows")
    return _features_df


# ============================================================
# Core Inference
# ============================================================

def run_inference(audio_features, timestamps, lrc_path, emotional_lag=2.0):
    """Run the full pipeline and return results dict."""

    T_orig = len(timestamps)

    # Apply emotional lag
    if emotional_lag > 0 and T_orig > 1:
        dt = float(np.median(np.diff(timestamps)))
        lag_frames = max(1, round(emotional_lag / dt))
        lag_frames = min(lag_frames, T_orig - 1)
        audio_features = audio_features[:-lag_frames]
        timestamps = timestamps[:-lag_frames]

    T = len(timestamps)

    # Encode lyrics
    encoder = get_encoder()
    with torch.no_grad():
        lyrics_emb = encoder(
            audio_timestamps=timestamps,
            lrc_path=lrc_path,
        )

    # Match lengths
    min_len = min(T, lyrics_emb.shape[0])
    audio_features = audio_features[:min_len]
    lyrics_emb = lyrics_emb[:min_len]
    timestamps = timestamps[:min_len]
    T = min_len

    # Prepare tensors
    model = get_model()
    audio_t = torch.from_numpy(audio_features.astype(np.float32)).unsqueeze(0).to(DEVICE)
    lyrics_t = lyrics_emb.unsqueeze(0).to(DEVICE)
    mask = torch.ones(1, T, device=DEVICE)

    # Forward pass
    with torch.no_grad():
        output = model(audio_t, lyrics_t, mask=mask)
        va_z = output["va_pred"].squeeze(0)

    # De-normalize
    stats = np.load(PATHS["stats"])
    g_mean = torch.tensor(stats["mean"], dtype=torch.float32, device=DEVICE)
    g_std = torch.tensor(stats["std"], dtype=torch.float32, device=DEVICE)
    va_pred = (va_z * g_std + g_mean).clamp(0.0, 1.0).cpu().numpy()

    return {
        "timestamps": timestamps.tolist(),
        "valence": va_pred[:, 0].tolist(),
        "arousal": va_pred[:, 1].tolist(),
        "valence_z": va_z[:, 0].cpu().numpy().tolist(),
        "arousal_z": va_z[:, 1].cpu().numpy().tolist(),
        "num_frames": T,
        "duration_sec": float(timestamps[-1]) if T > 0 else 0,
    }


def load_ground_truth(music_id, timestamps):
    """Load ground-truth VA for a PMEmo song."""
    import pandas as pd

    va_df = pd.read_csv(PATHS["annotations"])
    va_df = va_df.sort_values(["musicId", "frameTime"])
    va_df["musicId"] = va_df["musicId"].astype(int)

    song_va = va_df[va_df["musicId"] == music_id]
    if len(song_va) == 0:
        return None

    va_times = np.round(song_va["frameTime"].values, 2)
    query_times = np.round(np.array(timestamps), 2)

    common = np.intersect1d(query_times, va_times)
    if len(common) == 0:
        return None

    mask = np.isin(va_times, common)
    return {
        "valence": song_va["Valence(mean)"].values[mask].tolist(),
        "arousal": song_va["Arousal(mean)"].values[mask].tolist(),
    }


# ============================================================
# Routes
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/song-ids")
def get_song_ids():
    """Return list of available PMEmo song IDs."""
    df = get_features_df()
    ids = sorted(df["musicId"].unique().tolist())
    return jsonify({"ids": ids, "count": len(ids)})


@app.route("/api/predict/csv", methods=["POST"])
def predict_csv():
    """Run inference on a PMEmo song by musicId."""
    data = request.get_json()
    music_id = int(data.get("music_id", 1))
    compare_gt = data.get("compare_gt", False)
    emotional_lag = float(data.get("emotional_lag", 2.0))

    df = get_features_df()
    song_data = df[df["musicId"] == music_id]

    if len(song_data) == 0:
        return jsonify({"error": f"musicId {music_id} not found"}), 404

    timestamps = song_data["frameTime"].values.astype(np.float64)
    audio_features = song_data.iloc[:, 2:].values.astype(np.float32)

    lrc_path = os.path.join(PATHS["lyrics_dir"], f"{music_id}.lrc")

    result = run_inference(audio_features, timestamps, lrc_path, emotional_lag)
    result["music_id"] = music_id
    result["has_lyrics"] = os.path.exists(lrc_path)

    if compare_gt:
        gt = load_ground_truth(music_id, result["timestamps"])
        if gt:
            result["ground_truth"] = gt

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, f"prediction_{music_id}.csv")
    np.savetxt(
        csv_path,
        np.column_stack([result["timestamps"], result["valence"], result["arousal"]]),
        delimiter=",", header="timestamp,valence,arousal", comments="", fmt="%.6f",
    )
    result["csv_file"] = f"prediction_{music_id}.csv"

    return jsonify(result)


@app.route("/api/predict/audio", methods=["POST"])
def predict_audio():
    """Run inference on an uploaded audio/video file (requires opensmile)."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    lrc_file = request.files.get("lyrics")
    emotional_lag = float(request.form.get("emotional_lag", 2.0))

    # Save uploaded files temporarily
    audio_path = os.path.join(RESULTS_DIR, "upload_" + audio_file.filename)
    audio_file.save(audio_path)

    # If the file is a video format, extract audio with ffmpeg
    VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".webm", ".mov", ".flv"}
    ext = os.path.splitext(audio_path)[1].lower()
    extracted_wav = None

    if ext in VIDEO_EXTENSIONS:
        import subprocess
        extracted_wav = audio_path + ".extracted.wav"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path, "-vn", "-acodec", "pcm_s16le",
                 "-ar", "16000", "-ac", "1", extracted_wav],
                capture_output=True, text=True, check=True, timeout=120,
            )
            audio_path = extracted_wav
        except FileNotFoundError:
            if os.path.exists(audio_path):
                os.remove(audio_path)
            return jsonify({
                "error": "ffmpeg is required for video files (MP4, MKV, etc). "
                         "Install it with: sudo apt install ffmpeg"
            }), 500
        except subprocess.CalledProcessError as e:
            if os.path.exists(audio_path):
                os.remove(audio_path)
            return jsonify({"error": f"ffmpeg failed to extract audio: {e.stderr}"}), 500

    lrc_path = "NO_LYRICS"
    if lrc_file and lrc_file.filename:
        lrc_path = os.path.join(RESULTS_DIR, "upload_" + lrc_file.filename)
        lrc_file.save(lrc_path)

    try:
        import opensmile

        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )
        features_df = smile.process_file(audio_path)
        features_np = features_df.values.astype(np.float32)

        starts = features_df.index.get_level_values("start").total_seconds().values
        ends = features_df.index.get_level_values("end").total_seconds().values
        frame_times = (starts + ends) / 2.0

        max_time = frame_times[-1]
        bin_edges = np.arange(0.5, max_time + 0.5, 0.5)
        bin_indices = np.digitize(frame_times, bin_edges)

        agg_features, agg_times = [], []
        for b in range(len(bin_edges)):
            mask = bin_indices == b
            if mask.sum() > 0:
                agg_features.append(features_np[mask].mean(axis=0))
                agg_times.append(bin_edges[b] if b < len(bin_edges) else max_time)

        audio_features = np.stack(agg_features, axis=0)
        timestamps = np.array(agg_times)

        if audio_features.shape[1] < 260:
            audio_features = np.pad(audio_features, ((0, 0), (0, 260 - audio_features.shape[1])))
        elif audio_features.shape[1] > 260:
            audio_features = audio_features[:, :260]

        result = run_inference(audio_features, timestamps, lrc_path, emotional_lag)
        result["filename"] = audio_file.filename
        result["has_lyrics"] = lrc_path != "NO_LYRICS"

        basename = os.path.splitext(audio_file.filename)[0]
        csv_path = os.path.join(RESULTS_DIR, f"prediction_{basename}.csv")
        np.savetxt(
            csv_path,
            np.column_stack([result["timestamps"], result["valence"], result["arousal"]]),
            delimiter=",", header="timestamp,valence,arousal", comments="", fmt="%.6f",
        )
        result["csv_file"] = f"prediction_{basename}.csv"

        return jsonify(result)

    except ImportError:
        return jsonify({
            "error": "opensmile Python package not installed. Install with: pip install opensmile"
        }), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up all temporary files
        orig_upload = os.path.join(RESULTS_DIR, "upload_" + audio_file.filename)
        for tmp in [orig_upload, extracted_wav]:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)
        if lrc_path != "NO_LYRICS" and os.path.exists(lrc_path):
            os.remove(lrc_path)


@app.route("/api/download/<filename>")
def download_result(filename):
    return send_from_directory(RESULTS_DIR, filename, as_attachment=True)


@app.route("/api/status")
def status():
    """Check what's loaded and ready."""
    return jsonify({
        "device": DEVICE,
        "model_loaded": _model is not None,
        "encoder_loaded": _encoder is not None,
        "stats_exists": os.path.exists(PATHS["stats"]),
        "model_exists": os.path.exists(PATHS["model"]),
    })


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  MambaMER Web Interface")
    print(f"  Device: {DEVICE}")
    print(f"  Open: http://localhost:5000")
    print(f"{'='*60}\n")

    app.run(host="0.0.0.0", port=5000, debug=False)
