# multimodal_datase2.py — Audio+Lyrics ablation dataset (mel removed)
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from lyrics import LyricsWindowEncoder

# =========================================================
# -------------- MULTIMODAL DATASET -----------------------
# =========================================================

class MultimodalPMEmoDataset(Dataset):
    """
    Z-SCORE NORMALISATION OF VA TARGETS
    ─────────────────────────────────────
    Fix: z-score each song's VA targets (mean=0, std=1 per song).
    The model now outputs unbounded linear values. At eval/inference
    time, predictions are rescaled back to [0,1] using the stored
    per-song mean and std.
    """

    def __init__(self,
                 feature_path,
                 va_path,
                 lyrics_folder,
                 device="cuda",
                 emotional_lag_sec=2.0):

        self.device = device
        self.lyrics_folder = lyrics_folder
        self.emotional_lag_sec = emotional_lag_sec

        # ---- Load audio features ----
        self.features = pd.read_csv(feature_path).values

        # ---- Load VA annotations ----
        va_df = pd.read_csv(va_path)
        va_df = va_df.sort_values(["musicId", "frameTime"])
        va_df["musicId"] = va_df["musicId"].astype(int)

        feature_ids = set(self.features[:, 0].astype(int))
        va_ids = set(va_df["musicId"].unique())

        self.music_ids = sorted(feature_ids.intersection(va_ids))
        self.va_groups = {k: v for k, v in va_df.groupby("musicId")}

        print(f"Multimodal dataset initialized with {len(self.music_ids)} songs.")
        if self.emotional_lag_sec > 0:
            print(f"Emotional lag applied: {self.emotional_lag_sec} s "
                  f"(VA labels shifted forward relative to audio).")
        print("VA targets will be z-scored per song (mean=0, std=1).")
        print("Mel modality: DISABLED (audio+lyrics ablation).")
        print("Predictions must be rescaled at eval time using va_mean/va_std.")

        # =====================================================
        # ---- PRECOMPUTE & CACHE DEBERTA LYRICS EMBEDDINGS ---
        # =====================================================
        cache_path = "/home/yashkale/MER_VER3/lyrics_deberta_cache.pt"
        
        if os.path.exists(cache_path):
            print(f"Loading cached DeBERTa lyrics from: {cache_path}")
            self.lyrics_cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        else:
            print(f"\nCache not found. Precomputing DeBERTa embeddings for all {len(self.music_ids)} songs...")
            print("This will take a few minutes, but only happens ONCE.")
            
            self.bert_encoder = LyricsWindowEncoder(
                lexicon_path="/home/yashkale/MER_VER3/NRC-VAD-Lexicon-v2.1.txt", 
                device=device
            )
            self.bert_encoder.eval()
            self.lyrics_cache = {}
            
            for i, music_id in enumerate(self.music_ids):
                if (i + 1) % 50 == 0:
                    print(f"  Processed {i + 1}/{len(self.music_ids)} songs...")
                
                # Calculate the aligned timestamps for this song
                mask = self.features[:, 0].astype(int) == music_id
                song_data = self.features[mask]
                feature_times = np.round(song_data[:, 1], 2)
                
                va_song = self.va_groups[music_id]
                va_times = np.round(va_song["frameTime"].values, 2)
                
                common_times = np.sort(np.intersect1d(feature_times, va_times))
                
                # Apply lag to timestamps to get the exact query points for lyrics
                if self.emotional_lag_sec > 0 and len(common_times) > 1:
                    dt = float(np.median(np.diff(common_times)))
                    lag_frames = max(1, round(self.emotional_lag_sec / dt))
                    lag_frames = min(lag_frames, len(common_times) - 1)
                    aligned_times = common_times[:-lag_frames]
                else:
                    aligned_times = common_times
                    
                # Run DeBERTa and store on CPU memory
                lrc_path = os.path.join(self.lyrics_folder, f"{music_id}.lrc")
                with torch.no_grad():
                    aligned_lyrics = self.bert_encoder(
                        audio_timestamps=aligned_times,
                        lrc_path=lrc_path,
                    )
                self.lyrics_cache[music_id] = aligned_lyrics.detach().cpu()
                
            print(f"Saving completed DeBERTa cache to {cache_path}...")
            torch.save(self.lyrics_cache, cache_path)
            
            # Delete encoder from GPU memory since we don't need it anymore
            del self.bert_encoder
            torch.cuda.empty_cache()
            print("Done! Ready to train.\n")

    def __len__(self):
        return len(self.music_ids)

    def __getitem__(self, idx):

        music_id = int(self.music_ids[idx])

        # =====================================================
        # AUDIO & VA
        # =====================================================

        mask = self.features[:, 0].astype(int) == music_id
        song_data = self.features[mask]
        song_data = song_data[np.argsort(song_data[:, 1])]

        feature_times = song_data[:, 1]
        audio_feat = song_data[:, 2:].astype(np.float32)

        va_song = self.va_groups[music_id]
        va_times = va_song["frameTime"].values
        valence  = va_song["Valence(mean)"].values
        arousal  = va_song["Arousal(mean)"].values

        va_values = np.stack([valence, arousal], axis=1).astype(np.float32)

        feature_times = np.round(feature_times, 2)
        va_times      = np.round(va_times, 2)
        common_times = np.intersect1d(feature_times, va_times)

        audio_indices = np.isin(feature_times, common_times)
        aligned_audio = audio_feat[audio_indices]

        va_indices = np.isin(va_times, common_times)
        aligned_va = va_values[va_indices]

        aligned_audio = aligned_audio[np.argsort(feature_times[audio_indices])]
        aligned_va    = aligned_va[np.argsort(va_times[va_indices])]
        aligned_times = np.sort(common_times)

        if self.emotional_lag_sec > 0 and len(aligned_times) > 1:
            dt = float(np.median(np.diff(aligned_times)))
            lag_frames = max(1, round(self.emotional_lag_sec / dt))
            lag_frames = min(lag_frames, len(aligned_times) - 1)

            aligned_audio = aligned_audio[:-lag_frames]
            aligned_va    = aligned_va[lag_frames:]
            aligned_times = aligned_times[:-lag_frames]

        T = len(aligned_times)

        # =====================================================
        # LYRICS (Instantly loaded from cache!)
        # =====================================================
        
        aligned_lyrics = self.lyrics_cache[music_id]

        # =====================================================
        # FINAL SAFE LENGTH MATCH
        # =====================================================

        min_len = min(aligned_audio.shape[0],
                      aligned_va.shape[0],
                      aligned_lyrics.shape[0])

        aligned_audio  = aligned_audio[:min_len]
        aligned_va     = aligned_va[:min_len]
        aligned_lyrics = aligned_lyrics[:min_len]
        aligned_times  = aligned_times[:min_len]

        # =====================================================
        # Z-SCORE VA PER SONG
        # =====================================================
        
        va_mean = aligned_va.mean(axis=0)           # (2,)  [V_mean, A_mean]
        va_std  = aligned_va.std(axis=0)            # (2,)  [V_std,  A_std]
        va_std  = np.maximum(va_std, 1e-3)          # guard against flat sequences

        aligned_va_norm = (aligned_va - va_mean) / va_std   # z-scored, unbounded

        # (Mel spectrogram removed — audio+lyrics ablation)
                
        return {
            "audio":   torch.from_numpy(aligned_audio),
            "lyrics":  aligned_lyrics,
            "va":      torch.from_numpy(aligned_va_norm),   # z-scored targets
            "va_mean": torch.from_numpy(va_mean.astype(np.float32)),  # (2,)
            "va_std":  torch.from_numpy(va_std.astype(np.float32)),   # (2,)
            "times":   torch.from_numpy(aligned_times.astype(np.float32)),
        }


# =========================================================
# ---------------- TEST -----------------------------------
# =========================================================

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = MultimodalPMEmoDataset(
        feature_path  = "/home/yashkale/MER_VER3/PMEmo2019/features/dynamic_features.csv",
        va_path       = "/home/yashkale/MER_VER3/PMEmo2019/annotations/dynamic_annotations.csv",
        lyrics_folder = "/home/yashkale/MER_VER3/PMEmo2019/lyrics",
        device        = device,
        emotional_lag_sec=2.0,
    )

    sample = dataset[0]

    audio   = sample["audio"]
    lyrics  = sample["lyrics"]
    va      = sample["va"]
    va_mean = sample["va_mean"]
    va_std  = sample["va_std"]
    times   = sample["times"]

    print("Audio shape  :", audio.shape)
    print("Lyrics shape :", lyrics.shape)
    print("VA shape     :", va.shape,   "  (z-scored)")
    print("VA mean      :", va_mean,    "  (original [0,1] scale)")
    print("VA std       :", va_std)
    print("Times shape  :", times.shape)

    assert len(audio) == len(lyrics) == len(va) == len(times)
    print("Alignment verified ✔ (audio+lyrics, no mel)")