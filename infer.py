"""
infer.py — MambaMER End-to-End Inference Pipeline
===================================================
Predicts frame-level Valence and Arousal curves for unseen songs.

Inputs required per song:
    1. Audio features  — (T, 260) numpy array (openSMILE LLDs from dynamic_features.csv
                          format, or extracted via openSMILE with matching config)
    2. Lyrics          — .lrc file (timestamped lyrics in LRC format), OR none
    3. Model weights   — best_model.pt (trained MambaMER checkpoint)
    4. Global stats    — global_stats.npz (from compute_global_stats.py)
    5. NRC-VAD Lexicon — NRC-VAD-Lexicon-v2.1.txt

Usage:
    # From PMEmo CSV (for validation / known songs):
    python infer.py --mode csv --music-id 1

    # From raw audio file (for truly unseen songs):
    python infer.py --mode audio --audio-path song.wav --lrc-path song.lrc

    # Batch all PMEmo songs:
    python infer.py --mode csv --all
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import MultimodalEmotionModel
from lyrics import LyricsWindowEncoder


# ============================================================
# Global Configuration — update these paths for your setup
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_PATHS = {
    "model":       os.path.join(BASE_DIR, "best_model.pt"),
    "stats":       os.path.join(BASE_DIR, "global_stats.npz"),
    "lexicon":     os.path.join(BASE_DIR, "NRC-VAD-Lexicon-v2.1.txt"),
    "features":    "/home/yashkale/MER_VER3/PMEmo2019/features/dynamic_features.csv",
    "annotations": "/home/yashkale/MER_VER3/PMEmo2019/annotations/dynamic_annotations.csv",
    "lyrics_dir":  "/home/yashkale/MER_VER3/PMEmo2019/lyrics",
}


# ============================================================
# 1. Lyrics Encoder — singleton loader
# ============================================================

_lyrics_encoder = None

def get_lyrics_encoder(lexicon_path: str, device: str = "cuda") -> LyricsWindowEncoder:
    """Lazy-load the DeBERTa lyrics encoder (expensive, reuse across songs)."""
    global _lyrics_encoder
    if _lyrics_encoder is None:
        print("Loading DeBERTa lyrics encoder...")
        _lyrics_encoder = LyricsWindowEncoder(
            lexicon_path=lexicon_path,
            device=device,
        )
        _lyrics_encoder.eval()
        print("Lyrics encoder ready.")
    return _lyrics_encoder


# ============================================================
# 2. Model — singleton loader
# ============================================================

_model = None

def get_model():
    global _model
    if _model is None:
        print("Loading MambaMER model...")
        _model = MultimodalEmotionModel().to(DEVICE)
        
        # Load the checkpoint
        state_dict = torch.load(PATHS["model"], map_location=DEVICE, weights_only=True)
        
        # Use strict=False to handle the PE buffer shape change (500 -> 2000).
        # The sinusoidal PE is deterministic, so the regenerated buffer is correct
        # for positions 0..499 and simply extends correctly for 500..1999.
        _model.load_state_dict(state_dict, strict=False)
        
        _model.eval()
        print("Model loaded successfully with extended sequence length.")
    return _model


# ============================================================
# 3. Audio Feature Extraction
# ============================================================

def load_audio_features_from_csv(music_id: int, feature_csv_path: str) -> tuple:
    """
    Load pre-extracted openSMILE features for a PMEmo song from the CSV.

    Returns:
        audio_features: (T, 260) numpy float32 array
        timestamps:     (T,)    numpy float64 array (frameTime in seconds)
    """
    import pandas as pd

    print(f"Loading audio features for musicId={music_id} from CSV...")
    df = pd.read_csv(feature_csv_path)
    df = df.sort_values(["musicId", "frameTime"])
    df["musicId"] = df["musicId"].astype(int)

    song_data = df[df["musicId"] == music_id]
    if len(song_data) == 0:
        raise ValueError(f"musicId {music_id} not found in {feature_csv_path}")

    timestamps = song_data["frameTime"].values.astype(np.float64)
    audio_features = song_data.iloc[:, 2:].values.astype(np.float32)

    print(f"  Loaded {len(timestamps)} frames, feature dim = {audio_features.shape[1]}")
    return audio_features, timestamps


def extract_audio_features_opensmile(audio_path: str) -> tuple:
    """
    Extract openSMILE eGeMAPS LLD features from a raw audio or video file.

    Supports: .wav, .mp3, .flac, .ogg, .m4a (direct)
              .mp4, .mkv, .avi, .webm, .mov (extracts audio via ffmpeg first)

    Requires: pip install opensmile
              ffmpeg (for video files)

    Returns:
        audio_features: (T, 260) numpy float32 array
        timestamps:     (T,)    numpy float64 array
    """
    try:
        import opensmile
    except ImportError:
        raise ImportError(
            "opensmile Python package not installed.\n"
            "Install it with: pip install opensmile\n"
            "Or use --mode csv with pre-extracted features."
        )

    # Handle video files — extract audio track with ffmpeg
    VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".webm", ".mov", ".flv"}
    ext = os.path.splitext(audio_path)[1].lower()
    temp_wav = None

    if ext in VIDEO_EXTENSIONS:
        import subprocess
        temp_wav = audio_path + ".extracted.wav"
        print(f"Extracting audio from video file: {audio_path}")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path, "-vn", "-acodec", "pcm_s16le",
                 "-ar", "16000", "-ac", "1", temp_wav],
                capture_output=True, text=True, check=True, timeout=120,
            )
            audio_path = temp_wav
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg is required for video files (MP4, MKV, etc).\n"
                "Install it with: sudo apt install ffmpeg"
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed to extract audio: {e.stderr}")

    print(f"Extracting openSMILE features from: {audio_path}")

    try:
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )

        features_df = smile.process_file(audio_path)

        # opensmile returns a multi-index DataFrame with (file, start, end)
        # We need to aggregate into 0.5s windows to match PMEmo's format
        features_np = features_df.values.astype(np.float32)

        # Get timestamps from the index (midpoint of each frame)
        starts = features_df.index.get_level_values("start").total_seconds().values
        ends = features_df.index.get_level_values("end").total_seconds().values
        frame_times = (starts + ends) / 2.0

        # PMEmo uses 0.5s windows — aggregate by binning
        max_time = frame_times[-1]
        bin_edges = np.arange(0.5, max_time + 0.5, 0.5)
        bin_indices = np.digitize(frame_times, bin_edges)

        aggregated_features = []
        aggregated_times = []

        for b in range(len(bin_edges)):
            mask = bin_indices == b
            if mask.sum() > 0:
                aggregated_features.append(features_np[mask].mean(axis=0))
                aggregated_times.append(bin_edges[b] if b < len(bin_edges) else max_time)

        if len(aggregated_features) == 0:
            raise ValueError("No features extracted — audio file may be too short or corrupted.")

        audio_features = np.stack(aggregated_features, axis=0)
        timestamps = np.array(aggregated_times)

        # Pad/truncate to 260 dims if opensmile config differs slightly
        if audio_features.shape[1] < 260:
            pad_width = 260 - audio_features.shape[1]
            audio_features = np.pad(audio_features, ((0, 0), (0, pad_width)), mode="constant")
            print(f"  Warning: Padded features from {audio_features.shape[1]-pad_width} to 260 dims")
        elif audio_features.shape[1] > 260:
            audio_features = audio_features[:, :260]
            print(f"  Warning: Truncated features to 260 dims")

        print(f"  Extracted {len(timestamps)} frames, feature dim = {audio_features.shape[1]}")
        return audio_features, timestamps
    finally:
        # Clean up temporary extracted wav
        if temp_wav and os.path.exists(temp_wav):
            os.remove(temp_wav)


# ============================================================
# 4. Core Inference Function
# ============================================================

def predict_emotion(
    audio_features: np.ndarray,
    timestamps: np.ndarray,
    lrc_path: str,
    model_path: str,
    stats_path: str,
    lexicon_path: str,
    device: str = "cuda",
    emotional_lag_sec: float = 2.0,
) -> dict:
    """
    Run full MambaMER inference on a single song.

    Args:
        audio_features:    (T, 260) numpy float32 — openSMILE LLD features
        timestamps:        (T,)    numpy — frame timestamps in seconds
        lrc_path:          path to .lrc file (non-existent path = no lyrics)
        model_path:        path to best_model.pt
        stats_path:        path to global_stats.npz
        lexicon_path:      path to NRC-VAD-Lexicon-v2.1.txt
        device:            'cuda' or 'cpu'
        emotional_lag_sec: emotional lag in seconds (must match training, default 2.0s)

    Returns:
        dict with keys:
            'valence':    (T',) numpy array in [0, 1]
            'arousal':    (T',) numpy array in [0, 1]
            'timestamps': (T',) numpy array of frame times in seconds
            'valence_z':  (T',) raw z-scored valence (before de-normalization)
            'arousal_z':  (T',) raw z-scored arousal (before de-normalization)
    """

    T_orig = audio_features.shape[0]
    assert audio_features.shape[1] == 260, (
        f"Expected 260-dim audio features, got {audio_features.shape[1]}"
    )

    # ── Apply emotional lag (same as training) ─────────────────
    # During training, audio is truncated at the end and VA is shifted forward.
    # At inference, we apply the same truncation to audio timestamps for lyrics alignment.
    if emotional_lag_sec > 0 and len(timestamps) > 1:
        dt = float(np.median(np.diff(timestamps)))
        lag_frames = max(1, round(emotional_lag_sec / dt))
        lag_frames = min(lag_frames, len(timestamps) - 1)

        audio_features = audio_features[:-lag_frames]
        timestamps = timestamps[:-lag_frames]

    T = audio_features.shape[0]

    # ── Step 1: Encode lyrics ──────────────────────────────────
    encoder = get_lyrics_encoder(lexicon_path, device)
    with torch.no_grad():
        lyrics_emb = encoder(
            audio_timestamps=timestamps,
            lrc_path=lrc_path,
        )  # (T, 771)

    # Match lengths (lyrics encoder may produce slightly different T)
    min_len = min(T, lyrics_emb.shape[0])
    audio_features = audio_features[:min_len]
    lyrics_emb = lyrics_emb[:min_len]
    timestamps = timestamps[:min_len]
    T = min_len

    # ── Step 2: Load model ─────────────────────────────────────
    model = get_model(model_path, device)

    # ── Step 3: Prepare tensors ────────────────────────────────
    audio_t = torch.from_numpy(audio_features.astype(np.float32)).unsqueeze(0).to(device)  # (1, T, 260)
    lyrics_t = lyrics_emb.unsqueeze(0).to(device)                                           # (1, T, 771)
    mask = torch.ones(1, T, device=device)                                                   # (1, T)

    # ── Step 4: Forward pass ───────────────────────────────────
    with torch.no_grad():
        output = model(audio_t, lyrics_t, mask=mask)
        va_z = output["va_pred"].squeeze(0)  # (T, 2)

    # ── Step 5: De-normalize → [0, 1] VA scale ────────────────
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"Global stats file not found: {stats_path}\n"
            f"Run compute_global_stats.py first!"
        )

    stats = np.load(stats_path)
    g_mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device)  # (2,)
    g_std = torch.tensor(stats["std"], dtype=torch.float32, device=device)    # (2,)

    va_pred = (va_z * g_std + g_mean).clamp(0.0, 1.0)  # (T, 2)

    # ── Return results ─────────────────────────────────────────
    va_pred_np = va_pred.cpu().numpy()
    va_z_np = va_z.cpu().numpy()

    return {
        "valence":    va_pred_np[:, 0],
        "arousal":    va_pred_np[:, 1],
        "timestamps": timestamps[:T],
        "valence_z":  va_z_np[:, 0],
        "arousal_z":  va_z_np[:, 1],
    }


# ============================================================
# 5. Visualization
# ============================================================

def plot_predictions(
    result: dict,
    title: str = "MambaMER Emotion Prediction",
    save_path: str = None,
    ground_truth: dict = None,
):
    """
    Plot predicted valence and arousal curves over time.

    Args:
        result:       output dict from predict_emotion()
        title:        plot title
        save_path:    if provided, saves the figure instead of showing
        ground_truth: optional dict with 'valence' and 'arousal' keys for comparison
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    timestamps = result["timestamps"]

    # ── Valence ──
    axes[0].plot(timestamps, result["valence"], color="#2196F3", linewidth=1.5,
                 label="Predicted", alpha=0.9)
    if ground_truth is not None:
        axes[0].plot(timestamps[:len(ground_truth["valence"])],
                     ground_truth["valence"][:len(timestamps)],
                     color="#FF5722", linewidth=1.2, alpha=0.7, linestyle="--",
                     label="Ground Truth")
    axes[0].set_ylabel("Valence", fontsize=12)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(title, fontsize=14, fontweight="bold")

    # ── Arousal ──
    axes[1].plot(timestamps, result["arousal"], color="#4CAF50", linewidth=1.5,
                 label="Predicted", alpha=0.9)
    if ground_truth is not None:
        axes[1].plot(timestamps[:len(ground_truth["arousal"])],
                     ground_truth["arousal"][:len(timestamps)],
                     color="#FF5722", linewidth=1.2, alpha=0.7, linestyle="--",
                     label="Ground Truth")
    axes[1].set_ylabel("Arousal", fontsize=12)
    axes[1].set_xlabel("Time (seconds)", fontsize=12)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    else:
        plt.savefig("prediction_plot.png", dpi=150, bbox_inches="tight")
        print("Plot saved to: prediction_plot.png")

    plt.close()


# ============================================================
# 6. Load Ground Truth (for PMEmo validation)
# ============================================================

def load_ground_truth(music_id: int, va_csv_path: str, timestamps: np.ndarray) -> dict:
    """
    Load ground-truth VA annotations from PMEmo CSV for a specific song.
    Aligns to the provided timestamps.
    """
    import pandas as pd

    va_df = pd.read_csv(va_csv_path)
    va_df = va_df.sort_values(["musicId", "frameTime"])
    va_df["musicId"] = va_df["musicId"].astype(int)

    song_va = va_df[va_df["musicId"] == music_id]
    if len(song_va) == 0:
        print(f"  No ground truth found for musicId={music_id}")
        return None

    va_times = np.round(song_va["frameTime"].values, 2)
    query_times = np.round(timestamps, 2)

    common = np.intersect1d(query_times, va_times)
    if len(common) == 0:
        print(f"  No overlapping timestamps for ground truth")
        return None

    mask = np.isin(va_times, common)
    valence = song_va["Valence(mean)"].values[mask]
    arousal = song_va["Arousal(mean)"].values[mask]

    return {
        "valence": valence,
        "arousal": arousal,
    }


# ============================================================
# 7. CLI Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="MambaMER Inference — Predict valence/arousal for music",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single PMEmo song by ID:
  python infer.py --mode csv --music-id 1

  # All PMEmo songs:
  python infer.py --mode csv --all

  # Raw audio file:
  python infer.py --mode audio --audio-path song.wav --lrc-path song.lrc

  # Custom model checkpoint:
  python infer.py --mode csv --music-id 1 --model best_model_seed_42.pt
        """,
    )

    parser.add_argument("--mode", choices=["csv", "audio"], default="csv",
                        help="'csv' = load from PMEmo CSV, 'audio' = extract from audio file")
    parser.add_argument("--music-id", type=int, default=None,
                        help="PMEmo musicId (for --mode csv)")
    parser.add_argument("--all", action="store_true",
                        help="Process all songs in PMEmo (for --mode csv)")
    parser.add_argument("--audio-path", type=str, default=None,
                        help="Path to audio file (for --mode audio)")
    parser.add_argument("--lrc-path", type=str, default=None,
                        help="Path to .lrc lyrics file (optional)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model checkpoint (default: best_model.pt)")
    parser.add_argument("--stats", type=str, default=None,
                        help="Path to global_stats.npz")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cuda' or 'cpu' (auto-detected)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip generating plots")
    parser.add_argument("--output-dir", type=str, default="inference_results",
                        help="Directory to save results")
    parser.add_argument("--compare-gt", action="store_true",
                        help="Overlay ground-truth VA on plots (--mode csv only)")
    parser.add_argument("--emotional-lag", type=float, default=2.0,
                        help="Emotional lag in seconds (must match training, default 2.0)")

    args = parser.parse_args()

    # ── Resolve defaults ───────────────────────────────────────
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.model or DEFAULT_PATHS["model"]
    stats_path = args.stats or DEFAULT_PATHS["stats"]

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)
    if not os.path.exists(stats_path):
        print(f"ERROR: Global stats file not found: {stats_path}")
        print("Run 'python compute_global_stats.py' first!")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  MambaMER Inference Pipeline")
    print(f"{'='*60}")
    print(f"  Device:        {device}")
    print(f"  Model:         {model_path}")
    print(f"  Global Stats:  {stats_path}")
    print(f"  Emotional Lag: {args.emotional_lag}s")
    print(f"  Output Dir:    {args.output_dir}")
    print(f"{'='*60}\n")

    # ── MODE: CSV (PMEmo pre-extracted features) ───────────────
    if args.mode == "csv":
        import pandas as pd

        if args.all:
            feat_df = pd.read_csv(DEFAULT_PATHS["features"])
            music_ids = sorted(feat_df["musicId"].astype(int).unique())
            print(f"Processing ALL {len(music_ids)} songs...\n")
        elif args.music_id is not None:
            music_ids = [args.music_id]
        else:
            print("ERROR: Specify --music-id <ID> or --all for CSV mode")
            sys.exit(1)

        all_results = {}

        for i, mid in enumerate(music_ids):
            print(f"\n── Song {i+1}/{len(music_ids)}: musicId={mid} ──")

            try:
                audio_features, timestamps = load_audio_features_from_csv(
                    mid, DEFAULT_PATHS["features"]
                )
            except ValueError as e:
                print(f"  SKIP: {e}")
                continue

            lrc_path = args.lrc_path or os.path.join(
                DEFAULT_PATHS["lyrics_dir"], f"{mid}.lrc"
            )

            result = predict_emotion(
                audio_features=audio_features,
                timestamps=timestamps,
                lrc_path=lrc_path,
                model_path=model_path,
                stats_path=stats_path,
                lexicon_path=DEFAULT_PATHS["lexicon"],
                device=device,
                emotional_lag_sec=args.emotional_lag,
            )

            all_results[mid] = result

            # Print summary
            print(f"  Valence: [{result['valence'].min():.3f}, {result['valence'].max():.3f}] "
                  f"mean={result['valence'].mean():.3f}")
            print(f"  Arousal: [{result['arousal'].min():.3f}, {result['arousal'].max():.3f}] "
                  f"mean={result['arousal'].mean():.3f}")
            print(f"  Frames:  {len(result['timestamps'])}")

            # Save CSV
            csv_path = os.path.join(args.output_dir, f"prediction_{mid}.csv")
            np.savetxt(
                csv_path,
                np.column_stack([
                    result["timestamps"],
                    result["valence"],
                    result["arousal"],
                ]),
                delimiter=",",
                header="timestamp,valence,arousal",
                comments="",
                fmt="%.6f",
            )
            print(f"  Saved:   {csv_path}")

            # Plot
            if not args.no_plot:
                gt = None
                if args.compare_gt:
                    gt = load_ground_truth(
                        mid, DEFAULT_PATHS["annotations"],
                        result["timestamps"]
                    )

                plot_path = os.path.join(args.output_dir, f"plot_{mid}.png")
                plot_predictions(
                    result,
                    title=f"MambaMER Prediction — Song {mid}",
                    save_path=plot_path,
                    ground_truth=gt,
                )

        # Save all results as npz
        if len(all_results) > 1:
            npz_path = os.path.join(args.output_dir, "all_predictions.npz")
            np.savez(npz_path, **{
                f"song_{mid}_valence": r["valence"]
                for mid, r in all_results.items()
            }, **{
                f"song_{mid}_arousal": r["arousal"]
                for mid, r in all_results.items()
            })
            print(f"\nAll predictions saved to: {npz_path}")

    # ── MODE: AUDIO (raw audio file) ──────────────────────────
    elif args.mode == "audio":
        if args.audio_path is None:
            print("ERROR: --audio-path required for audio mode")
            sys.exit(1)

        if not os.path.exists(args.audio_path):
            print(f"ERROR: Audio file not found: {args.audio_path}")
            sys.exit(1)

        audio_features, timestamps = extract_audio_features_opensmile(args.audio_path)

        lrc_path = args.lrc_path or "NO_LYRICS"

        result = predict_emotion(
            audio_features=audio_features,
            timestamps=timestamps,
            lrc_path=lrc_path,
            model_path=model_path,
            stats_path=stats_path,
            lexicon_path=DEFAULT_PATHS["lexicon"],
            device=device,
            emotional_lag_sec=args.emotional_lag,
        )

        basename = os.path.splitext(os.path.basename(args.audio_path))[0]

        print(f"\n  Valence: [{result['valence'].min():.3f}, {result['valence'].max():.3f}] "
              f"mean={result['valence'].mean():.3f}")
        print(f"  Arousal: [{result['arousal'].min():.3f}, {result['arousal'].max():.3f}] "
              f"mean={result['arousal'].mean():.3f}")
        print(f"  Frames:  {len(result['timestamps'])}")

        # Save CSV
        csv_path = os.path.join(args.output_dir, f"prediction_{basename}.csv")
        np.savetxt(
            csv_path,
            np.column_stack([
                result["timestamps"],
                result["valence"],
                result["arousal"],
            ]),
            delimiter=",",
            header="timestamp,valence,arousal",
            comments="",
            fmt="%.6f",
        )
        print(f"  Saved: {csv_path}")

        if not args.no_plot:
            plot_path = os.path.join(args.output_dir, f"plot_{basename}.png")
            plot_predictions(
                result,
                title=f"MambaMER Prediction — {basename}",
                save_path=plot_path,
            )

    print(f"\n{'='*60}")
    print(f"  Inference complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
