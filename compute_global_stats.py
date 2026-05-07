"""
compute_global_stats.py — One-time corpus-level VA statistics computation
==========================================================================
Run this ONCE after training to compute the global mean & std of VA values
across the full corpus.  The resulting .npz file is used by infer.py to
de-normalize z-scored model predictions back to [0, 1] VA space.

The model was trained with per-song z-scoring:
    va_z = (va_raw - song_mean) / song_std

At inference we don't have per-song stats, so we approximate:
    va_pred ≈ va_z * global_std + global_mean

where global_mean and global_std are computed from ALL raw VA frames
pooled across the entire corpus.

Usage:
    python compute_global_stats.py

Output:
    global_stats.npz  (contains 'mean' and 'std', each shape (2,))
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def compute_and_save(
    va_path: str,
    output_path: str = "global_stats.npz",
):
    """
    Pool ALL raw VA frames across the entire corpus and compute
    corpus-wide mean and std.

    This gives the correct scale for de-normalization — the std reflects
    the true spread of VA values across all songs, not just the tiny
    within-song variation.
    """
    import pandas as pd

    print("Loading VA annotations...")
    va_df = pd.read_csv(va_path)
    va_df = va_df.sort_values(["musicId", "frameTime"])

    valence = va_df["Valence(mean)"].values.astype(np.float64)
    arousal = va_df["Arousal(mean)"].values.astype(np.float64)

    global_va_mean = np.array([valence.mean(), arousal.mean()], dtype=np.float32)
    global_va_std  = np.array([valence.std(),  arousal.std()],  dtype=np.float32)

    np.savez(
        output_path,
        mean=global_va_mean,
        std=global_va_std,
    )

    n_songs = va_df["musicId"].nunique()
    n_frames = len(va_df)

    print(f"\n{'='*50}")
    print(f"Global VA Statistics (corpus-level)")
    print(f"{'='*50}")
    print(f"  Songs:  {n_songs}")
    print(f"  Frames: {n_frames}")
    print(f"  Global Mean (Valence, Arousal): {global_va_mean}")
    print(f"  Global Std  (Valence, Arousal): {global_va_std}")
    print(f"\nSaved to: {os.path.abspath(output_path)}")
    print(f"{'='*50}")

    return global_va_mean, global_va_std


if __name__ == "__main__":

    # ── Paths (same as train1.py) ────────────────────────────
    VA_PATH = "/home/yashkale/MER_VER3/PMEmo2019/annotations/dynamic_annotations.csv"

    compute_and_save(
        va_path=VA_PATH,
        output_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "global_stats.npz"),
    )
