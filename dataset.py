"""
dataset.py — Window-level PMEmo Multimodal Dataset
====================================================
Disk cache added:
  - On first run, after all songs are pre-computed, the full song_cache
    is saved to `cache_path` (default: same dir as feature_path).
  - On subsequent runs, if the cache file exists and the config matches
    (window_sec, stride_sec, skip_seconds, deberta_model), the cache is
    loaded directly — DeBERTa never runs.
  - Cache is invalidated automatically if any of the above params change.
  - Set force_recompute=True to ignore existing cache and recompute.

All other fixes from previous version preserved:
  - Pre-compute once in __init__, __getitem__ is pure cache lookup.
  - Pretrained DeBERTa weights loaded, attn_pool skipped (768 vs 771).
  - win_ends + song_duration passed to lyric_encoder for correct alignment.
  - skip_seconds=15.0 removes PMEmo annotation delay artifact.
  - NaN guard drops poisoned songs.
"""

import os
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from lyrics import LyricsWindowEncoder


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_windows(
    timestamps : np.ndarray,
    window_sec : float,
    stride_sec : float,
) -> list[tuple[float, float]]:
    t_start = timestamps[0]
    t_end   = timestamps[-1]
    windows = []
    w_start = t_start
    while w_start + window_sec <= t_end + 1e-6:
        windows.append((w_start, w_start + window_sec))
        w_start += stride_sec
    return windows


def _frames_in_window(
    timestamps : np.ndarray,
    win_start  : float,
    win_end    : float,
) -> np.ndarray:
    return (timestamps >= win_start) & (timestamps < win_end)


def _has_nan(*tensors: torch.Tensor) -> bool:
    return any(torch.isnan(t).any().item() for t in tensors)


def _cache_key(
    window_sec   : float,
    stride_sec   : float,
    skip_seconds : float,
    deberta_model: str,
) -> str:
    """Short hash identifying the cache config. Changes if any param changes."""
    s = f"w{window_sec}_s{stride_sec}_sk{skip_seconds}_m{deberta_model}"
    return hashlib.md5(s.encode()).hexdigest()[:10]


# ─────────────────────────────────────────────────────────────────────────────
# PMEmoWindowDataset
# ─────────────────────────────────────────────────────────────────────────────

class PMEmoWindowDataset(Dataset):
    """
    Window-level multimodal dataset for PMEmo2019.

    Parameters
    ----------
    feature_path            : path to dynamic_features.csv
    va_path                 : path to dynamic_annotations.csv
    lyrics_folder           : folder containing {music_id}.lrc files
    lexicon_path            : path to NRC-VAD-Lexicon-v2.1.txt
    window_sec              : window size in seconds (default 3.0)
    stride_sec              : stride between windows in seconds (default 0.5)
    deberta_model           : HuggingFace model name
    freeze_deberta          : freeze DeBERTa weights during caching
    device                  : torch device string
    pretrained_encoder_path : optional path to best_encoder.pt
    skip_seconds            : seconds to discard from song start (default 15.0)
    cache_path              : directory to save/load the disk cache.
                              Defaults to same directory as feature_path.
                              Set to None to disable disk caching.
    force_recompute         : if True, ignore existing cache and recompute.
    """

    def __init__(
        self,
        feature_path            : str,
        va_path                 : str,
        lyrics_folder           : str,
        lexicon_path            : str,
        window_sec              : float = 3.0,
        stride_sec              : float = 0.5,
        deberta_model           : str   = "microsoft/deberta-v3-base",
        freeze_deberta          : bool  = True,
        device                  : str   = "cuda",
        pretrained_encoder_path : str   = None,
        skip_seconds            : float = 15.0,
        cache_path              : str   = None,
        force_recompute         : bool  = False,
    ):
        self.window_sec    = window_sec
        self.stride_sec    = stride_sec
        self.device        = device
        self.lyrics_folder = lyrics_folder
        self.skip_seconds  = skip_seconds

        # ── Disk cache path ───────────────────────────────────────────────
        if cache_path is None:
            cache_path = os.path.dirname(os.path.abspath(feature_path))

        key        = _cache_key(window_sec, stride_sec, skip_seconds, deberta_model)
        cache_file = os.path.join(cache_path, f"pmemo_cache_{key}.pt")

        # ── Try loading existing cache ────────────────────────────────────
        if not force_recompute and os.path.exists(cache_file):
            print(f"[dataset] Loading cache from: {cache_file}")
            payload = torch.load(cache_file, map_location="cpu",
                                 weights_only=False)
            self.song_cache = payload["song_cache"]
            print(f"[dataset] Cache loaded — {len(self.song_cache)} songs | "
                  f"window={window_sec}s  stride={stride_sec}s  skip={skip_seconds}s")
            return   # DeBERTa never instantiated

        # ── Build cache from scratch ──────────────────────────────────────
        lyric_encoder = LyricsWindowEncoder(
            lexicon_path   = lexicon_path,
            deberta_model  = deberta_model,
            window_sec     = window_sec,
            freeze_deberta = freeze_deberta,
            device         = device,
        )

        if pretrained_encoder_path is not None:
            if os.path.exists(pretrained_encoder_path):
                state = torch.load(pretrained_encoder_path, map_location=device,
                                   weights_only=False)

                if "deberta" in state:
                    missing, unexpected = lyric_encoder.deberta.load_state_dict(
                        state["deberta"], strict=False
                    )
                    if missing:
                        print(f"[dataset] pretrained DeBERTa — missing keys: {missing}")
                    if unexpected:
                        print(f"[dataset] pretrained DeBERTa — unexpected keys: {unexpected}")
                    print(f"[dataset] Loaded pretrained DeBERTa from: {pretrained_encoder_path}")
                else:
                    print(f"[dataset] WARNING: 'deberta' key not found in checkpoint.")

                # attn_pool intentionally NOT loaded (768 vs 771 dim mismatch)
                print("[dataset] attn_pool skipped (768 vs 771 dim mismatch — trains from scratch).")
            else:
                print(f"[dataset] WARNING: pretrained_encoder_path not found: "
                      f"{pretrained_encoder_path} — using random init.")

        feat_df = pd.read_csv(feature_path)
        feat_df = feat_df.sort_values(["musicId", "frameTime"])
        feat_df["musicId"] = feat_df["musicId"].astype(int)

        va_df = pd.read_csv(va_path)
        va_df = va_df.sort_values(["musicId", "frameTime"])
        va_df["musicId"] = va_df["musicId"].astype(int)

        common_ids = sorted(
            set(feat_df["musicId"].unique()) &
            set(va_df["musicId"].unique())
        )

        feat_groups = {k: v for k, v in
                       feat_df[feat_df["musicId"].isin(common_ids)].groupby("musicId")}
        va_groups   = {k: v for k, v in
                       va_df[va_df["musicId"].isin(common_ids)].groupby("musicId")}

        if skip_seconds > 0:
            print(f"[dataset] Skipping first {skip_seconds}s of each song "
                  f"(PMEmo annotation delay fix).")
        print(f"Pre-computing {len(common_ids)} songs "
              f"(DeBERTa runs once, then cached to disk) …")

        self.song_cache: list[dict] = []
        skipped_short = 0
        skipped_nan   = 0

        for i, music_id in enumerate(common_ids):
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(common_ids)}] processed …")

            feat_song  = feat_groups[music_id]
            feat_times = feat_song["frameTime"].values
            feat_vals  = feat_song.iloc[:, 2:].values.astype(np.float32)

            va_song  = va_groups[music_id]
            va_times = va_song["frameTime"].values
            valence  = va_song["Valence(mean)"].values.astype(np.float32)
            arousal  = va_song["Arousal(mean)"].values.astype(np.float32)

            all_windows  = _build_windows(feat_times, window_sec, stride_sec)
            t_song_start = feat_times[0]
            windows = [
                (ws, we) for ws, we in all_windows
                if ws >= t_song_start + skip_seconds
            ]

            if len(windows) == 0:
                skipped_short += 1
                continue

            # Audio aggregation
            audio_windows = []
            for ws, we in windows:
                m = _frames_in_window(feat_times, ws, we)
                audio_windows.append(
                    feat_vals[m].mean(axis=0) if m.any()
                    else np.zeros(feat_vals.shape[1], dtype=np.float32)
                )
            audio_tensor = torch.from_numpy(np.stack(audio_windows))

            # VA averaging
            va_windows = []
            for ws, we in windows:
                m = _frames_in_window(va_times, ws, we)
                if m.any():
                    va_windows.append(np.array(
                        [valence[m].mean(), arousal[m].mean()], dtype=np.float32
                    ))
                else:
                    va_windows.append(np.zeros(2, dtype=np.float32))
            va_tensor = torch.from_numpy(np.stack(va_windows))

            # Lyric encoding
            win_ends      = np.array([we for _, we in windows], dtype=np.float64)
            song_duration = float(feat_times[-1])
            lrc_path      = os.path.join(lyrics_folder, f"{music_id}.lrc")

            lyric_tensor = lyric_encoder(
                audio_timestamps = win_ends,
                lrc_path         = lrc_path,
                song_duration    = song_duration,
            ).detach().cpu()

            # Align lengths
            N = min(audio_tensor.shape[0], lyric_tensor.shape[0], va_tensor.shape[0])
            audio_tensor  = audio_tensor[:N]
            lyric_tensor  = lyric_tensor[:N]
            va_tensor     = va_tensor[:N]

            # NaN guard
            if _has_nan(audio_tensor, lyric_tensor, va_tensor):
                print(f"[dataset] WARNING: NaN in song {music_id} — skipping.")
                skipped_nan += 1
                continue

            self.song_cache.append({
                "audio"   : audio_tensor,
                "lyrics"  : lyric_tensor,
                "va"      : va_tensor,
                "music_id": str(music_id),
            })

        del lyric_encoder
        torch.cuda.empty_cache()

        if skipped_short:
            print(f"[dataset] {skipped_short} songs skipped — "
                  f"too short after {skip_seconds}s trim.")
        if skipped_nan:
            print(f"[dataset] {skipped_nan} songs skipped — NaN detected.")

        print(f"PMEmoWindowDataset: {len(self.song_cache)} songs cached | "
              f"window={window_sec}s  stride={stride_sec}s  skip={skip_seconds}s")

        # ── Save cache to disk ────────────────────────────────────────────
        print(f"[dataset] Saving cache to: {cache_file} …")
        torch.save({"song_cache": self.song_cache}, cache_file)
        size_mb = os.path.getsize(cache_file) / 1e6
        print(f"[dataset] Cache saved ({size_mb:.1f} MB) — "
              f"next run will load instantly.")

    def __len__(self) -> int:
        return len(self.song_cache)

    def __getitem__(self, idx: int) -> dict:
        return self.song_cache[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Kept for backward compatibility
# ─────────────────────────────────────────────────────────────────────────────

class FlatWindowDataset(Dataset):
    def __init__(self, song_dataset: PMEmoWindowDataset):
        self.song_ds = song_dataset
        self.index: list[tuple[int, int]] = []
        for song_idx in range(len(song_dataset)):
            n = song_dataset[song_idx]["audio"].shape[0]
            for w in range(n):
                self.index.append((song_idx, w))
        print(f"FlatWindowDataset: {len(self.index):,} total windows.")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, flat_idx: int) -> dict:
        song_idx, win_idx = self.index[flat_idx]
        s = self.song_ds[song_idx]
        return {
            "audio"  : s["audio"][win_idx],
            "lyrics" : s["lyrics"][win_idx],
            "va"     : s["va"][win_idx],
        }