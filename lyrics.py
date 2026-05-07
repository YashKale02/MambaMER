"""
lyrics.py — Token-level Lyric Preprocessing Module
====================================================
Changes from previous version:
  - BERT replaced with DeBERTa-v3-base (microsoft/deberta-v3-base).
    DeBERTa uses disentangled attention (content + position separately),
    which produces better contextualised token embeddings than BERT for
    emotion-related downstream tasks.
  - Tokenizer swapped to DebertaV2TokenizerFast (required for v3 models).
  - Model swapped to DebertaV2Model.
  - Hidden dim is still 768 (DeBERTa-base) so HIDDEN_DIM = 771 is unchanged.
    All downstream consumers (dataset.py, models.py) see the
    same (T, 771) output — no other files need modification.

NaN guards added:
  - DeBERTa hidden state checked after each line encode; NaN replaced with zeros.
  - VAD lookup result checked; NaN replaced with zeros.
  - attn_pool output checked; NaN replaced with no_lyric_embedding.
  - Final stacked output checked; NaN rows replaced with no_lyric_embedding.
  - Empty token sequences (only CLS/SEP) handled gracefully.

Alignment strategy:
  Each lyric line is encoded once by DeBERTa and attention-pooled into a
  single 771-dim sentence embedding (768 DeBERTa + 3 mean VAD). At each
  audio timestamp t, the embedding is linearly interpolated between the
  embeddings of the two consecutive lyric lines whose start times bracket t:

      alpha = (t - t_i) / (t_{i+1} - t_i)
      emb   = (1 - alpha) * emb_i  +  alpha * emb_{i+1}

  Timestamps before the first lyric line clamp to line 0; timestamps after
  the last line clamp to the last line.
"""

import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DebertaV2TokenizerFast, DebertaV2Model


# =========================================================
# 1.  NRC-VAD LEXICON
# =========================================================

class NRCVADLexicon:
    """
    Loads the NRC-VAD lexicon (tab-separated: term / valence / arousal / dominance).
    Provides O(1) token lookup; returns a zero vector for unknown words.
    """
    def __init__(self, lexicon_path: str):
        self.vad: dict[str, np.ndarray] = {}
        df = pd.read_csv(lexicon_path, sep="\t", header=0,
                         names=["term", "valence", "arousal", "dominance"],
                         skiprows=1)
        for _, row in df.iterrows():
            key = str(row["term"]).strip().lower()
            self.vad[key] = np.array(
                [float(row["valence"]), float(row["arousal"]), float(row["dominance"])],
                dtype=np.float32
            )

    def lookup(self, token: str) -> np.ndarray:
        return self.vad.get(token.strip().lower(), np.zeros(3, dtype=np.float32))

    def lookup_batch(self, tokens: list[str]) -> np.ndarray:
        return np.stack([self.lookup(t) for t in tokens], axis=0)


# =========================================================
# 2.  LRC PARSER
# =========================================================

def parse_lrc(path: str) -> list[dict]:
    """
    Parse an .lrc file into a list of dicts:
        {"start": float_seconds, "text": str}
    Returns [] on any error or if the file has no timestamped lines.
    """
    pattern = re.compile(r"\[(\d+):(\d+\.\d+)\](.*)")
    entries = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = pattern.match(line.strip())
                if m:
                    minutes = int(m.group(1))
                    seconds = float(m.group(2))
                    text    = m.group(3).strip()
                    if text:
                        entries.append({
                            "start": minutes * 60.0 + seconds,
                            "text":  text,
                        })
    except Exception:
        return []
    return entries


def assign_end_times(entries: list[dict], song_duration: float) -> list[dict]:
    """Assign end time to each lyric line (= start of next line, or song end)."""
    for i in range(len(entries) - 1):
        entries[i]["end"] = entries[i + 1]["start"]
    if entries:
        entries[-1]["end"] = song_duration
    return entries


# =========================================================
# 3.  ATTENTION-WEIGHTED POOLING
# =========================================================

class AttentionPool(nn.Module):
    """
    Soft attention over a variable-length set of token embeddings.
    Score = tanh(W·h) projected to scalar; softmax across tokens.
    Fully differentiable.

    Note: attribute is named 'attn' to match pretrain_lyric_encoder.py
    checkpoint key names ("attn.weight", "attn.bias").
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_embeddings: (N, hidden_dim)
        Returns:
            pooled:           (hidden_dim,)
        """
        scores  = self.attn(torch.tanh(token_embeddings))  # (N, 1)
        weights = F.softmax(scores, dim=0)                  # (N, 1)
        pooled  = (weights * token_embeddings).sum(dim=0)   # (hidden_dim,)
        return pooled


# =========================================================
# 4.  LYRICS ENCODER  — DeBERTa-v3-base backbone
# =========================================================

class LyricsWindowEncoder(nn.Module):
    """
    Encodes lyrics into per-window embeddings aligned to audio feature timestamps.

    Architecture
    ------------
    - DebertaV2Model (microsoft/deberta-v3-base, optionally fine-tunable)
        → token-level last hidden state (768-dim)
    - NRC-VAD lookup per token → VAD scores (3-dim)
    - Concatenation             → 771-dim token embedding
    - Attention-weighted pooling per window
    - Learned [NO_LYRIC] embedding for windows with no overlapping lyrics

    Output dim: HIDDEN_DIM = 771

    Parameters
    ----------
    lexicon_path    : path to NRC-VAD-Lexicon-v2_1.txt
    deberta_model   : HuggingFace model name (default: microsoft/deberta-v3-base)
    window_sec      : window size in seconds (default: 3.0)
    freeze_deberta  : if True, DeBERTa weights are frozen (default: True)
    device          : torch device string
    """

    DEBERTA_DIM = 768
    VAD_DIM     = 3
    HIDDEN_DIM  = DEBERTA_DIM + VAD_DIM   # 771

    def __init__(
        self,
        lexicon_path   : str,
        deberta_model  : str   = "microsoft/deberta-v3-base",
        window_sec     : float = 3.0,
        freeze_deberta : bool  = True,
        device         : str   = "cuda",
    ):
        super().__init__()
        self.device     = torch.device(device)
        self.window_sec = window_sec

        # ---- DeBERTa ----
        self.tokenizer = DebertaV2TokenizerFast.from_pretrained(deberta_model)
        self.deberta   = DebertaV2Model.from_pretrained(deberta_model)
        self.set_deberta_frozen(freeze_deberta)

        # ---- VAD Lexicon ----
        self.lexicon = NRCVADLexicon(lexicon_path)

        # ---- Attention Pool ----
        # hidden_dim=771: accepts VAD-augmented token embeddings
        self.attn_pool = AttentionPool(self.HIDDEN_DIM)

        # ---- Learned [NO_LYRIC] embedding (771-dim) ----
        self.no_lyric_embedding = nn.Parameter(torch.zeros(self.HIDDEN_DIM))
        nn.init.normal_(self.no_lyric_embedding, mean=0.0, std=0.02)

        self.to(device)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def set_deberta_frozen(self, freeze: bool):
        """Toggle DeBERTa weight freezing at any time."""
        for p in self.deberta.parameters():
            p.requires_grad = not freeze

    # ------------------------------------------------------------------
    # Internal: encode lyric lines → token embeddings + words
    # ------------------------------------------------------------------

    def _encode_lines(
        self,
        lines: list[str],
    ) -> tuple[list[torch.Tensor], list[list[str]]]:
        """
        Run DeBERTa on each lyric line individually.
        Skips [CLS] and [SEP] special tokens.
        NaN in hidden states replaced with zeros.
        Empty lines (zero real tokens) replaced with fallback embedding.

        Returns
        -------
        token_embs_per_line  : list of (N_tokens_i, 768) tensors
        token_words_per_line : list of token strings (for VAD lookup)
        """
        token_embs_per_line  = []
        token_words_per_line = []

        fallback_emb = self.no_lyric_embedding[:self.DEBERTA_DIM].unsqueeze(0).detach()

        for line in lines:
            enc = self.tokenizer(
                line,
                return_tensors="pt",
                truncation=True,
                max_length=128,
            ).to(self.device)

            output = self.deberta(**enc)
            hidden = output.last_hidden_state.squeeze(0)   # (seq_len, 768)

            # NaN guard on DeBERTa output
            if torch.isnan(hidden).any():
                print(f"[lyrics] NaN in DeBERTa hidden state for line: '{line[:60]}' — replacing with zeros")
                hidden = torch.zeros_like(hidden)

            token_ids  = enc["input_ids"].squeeze(0).tolist()
            tokens_str = self.tokenizer.convert_ids_to_tokens(token_ids)

            # Drop [CLS] at index 0 and [SEP] at index -1
            token_embs  = hidden[1:-1]       # (N, 768)
            token_words = tokens_str[1:-1]   # list[str]

            # Empty line guard
            if token_embs.shape[0] == 0:
                token_embs_per_line.append(fallback_emb)
                token_words_per_line.append([""])
                continue

            token_embs_per_line.append(token_embs)
            token_words_per_line.append(token_words)

        return token_embs_per_line, token_words_per_line

    # ------------------------------------------------------------------
    # Internal: attach VAD scores to token embeddings → 771-dim
    # ------------------------------------------------------------------

    def _attach_vad(
        self,
        token_embs  : torch.Tensor,   # (N, 768)
        token_words : list[str],
    ) -> torch.Tensor:                 # (N, 771)
        """
        Look up VAD for each token, concatenate → (N, 771).
        NaN in VAD values replaced with zeros.
        """
        vad_np = self.lexicon.lookup_batch(token_words)       # (N, 3)  numpy

        # NaN guard on VAD numpy array
        if np.isnan(vad_np).any():
            print(f"[lyrics] NaN in VAD lookup — replacing with zeros")
            vad_np = np.nan_to_num(vad_np, nan=0.0)

        vad_t  = torch.from_numpy(vad_np).to(self.device)     # (N, 3)
        result = torch.cat([token_embs, vad_t], dim=-1)        # (N, 771)

        # NaN guard on concat result
        if torch.isnan(result).any():
            print(f"[lyrics] NaN after VAD concat — replacing with zeros")
            result = torch.nan_to_num(result, nan=0.0)

        return result

    # ------------------------------------------------------------------
    # Core: build per-window embeddings aligned to audio timestamps
    # ------------------------------------------------------------------

    def forward(
        self,
        audio_timestamps : np.ndarray,   # (T,) window-end times in seconds
        lrc_path         : str,
        song_duration    : float = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        audio_timestamps : 1-D numpy array of window-end timestamps (seconds).
                           lyrics.py treats each t as window [t-window_sec, t].
        lrc_path         : path to .lrc file for this song.
        song_duration    : true duration of the song in seconds (feat_times[-1]).
                           Falls back to audio_timestamps[-1] if None.

        Returns
        -------
        lyric_embedding : torch.Tensor  (T, 771)
        """
        T = len(audio_timestamps)

        # ── Step A: Parse lyrics ──────────────────────────────────────────
        if not os.path.exists(lrc_path):
            return self.no_lyric_embedding.unsqueeze(0).expand(T, -1).detach()

        entries = parse_lrc(lrc_path)
        if len(entries) == 0:
            return self.no_lyric_embedding.unsqueeze(0).expand(T, -1).detach()

        true_duration = (
            float(song_duration)
            if song_duration is not None
            else float(audio_timestamps[-1])
        )
        entries = assign_end_times(entries, true_duration)

        # ── Step B: DeBERTa-encode every lyric line → one 771-dim vector ──
        lines = [e["text"] for e in entries]
        token_embs_per_line, token_words_per_line = self._encode_lines(lines)

        line_embs: list[torch.Tensor] = []
        for tok_embs, tok_words in zip(token_embs_per_line, token_words_per_line):
            tok_vad = self._attach_vad(tok_embs, tok_words)  # (N, 771)
            pooled  = self.attn_pool(tok_vad)                 # (771,)

            # NaN guard on pooled output
            if torch.isnan(pooled).any():
                print(f"[lyrics] NaN in attn_pool output — replacing with no_lyric_embedding")
                pooled = self.no_lyric_embedding.detach().clone()

            line_embs.append(pooled)

        line_start_times = [e["start"] for e in entries]
        L = len(line_embs)

        # ── Step C: Linear interpolation at every audio timestamp ─────────
        result: list[torch.Tensor] = []

        for t in audio_timestamps:
            if t <= line_start_times[0]:
                result.append(line_embs[0])

            elif t >= line_start_times[-1]:
                result.append(line_embs[-1])

            else:
                # Binary search for bracketing interval [t_i, t_{i+1})
                lo, hi = 0, L - 2
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if line_start_times[mid] <= t:
                        lo = mid
                    else:
                        hi = mid - 1
                i     = lo
                t0    = line_start_times[i]
                t1    = line_start_times[i + 1]
                alpha = (t - t0) / (t1 - t0 + 1e-8)
                emb   = (1.0 - alpha) * line_embs[i] + alpha * line_embs[i + 1]
                result.append(emb)

        output = torch.stack(result, dim=0)   # (T, 771)

        # Final NaN guard — replace any remaining NaN rows
        nan_rows = torch.isnan(output).any(dim=1)
        if nan_rows.any():
            print(f"[lyrics] NaN in final output for {lrc_path} "
                  f"— {nan_rows.sum().item()} rows replaced with no_lyric_embedding")
            output[nan_rows] = self.no_lyric_embedding.detach()

        return output


# =========================================================
# 5.  DATASET (drop-in replacement for old LyricsOnlyDataset)
# =========================================================

class LyricsWindowDataset(torch.utils.data.Dataset):
    """
    Dataset that returns per-song window lyric embeddings aligned to audio timestamps.

    Each item:
        {
          "lyrics": Tensor (T, 771)   — window lyric embeddings
          "va":     Tensor (T, 2)     — valence / arousal targets
        }
    """

    def __init__(
        self,
        va_path        : str,
        lyrics_folder  : str,
        lexicon_path   : str,
        deberta_model  : str   = "microsoft/deberta-v3-base",
        window_sec     : float = 3.0,
        freeze_deberta : bool  = True,
        device         : str   = "cuda",
    ):
        self.lyrics_folder = lyrics_folder
        self.device        = device

        self.encoder = LyricsWindowEncoder(
            lexicon_path   = lexicon_path,
            deberta_model  = deberta_model,
            window_sec     = window_sec,
            freeze_deberta = freeze_deberta,
            device         = device,
        )

        va_df = pd.read_csv(va_path)
        va_df = va_df.sort_values(["musicId", "frameTime"])
        va_df["musicId"] = va_df["musicId"].astype(int)

        self.music_ids = sorted(va_df["musicId"].unique())
        self.va_groups = {k: v for k, v in va_df.groupby("musicId")}

        print(f"LyricsWindowDataset: {len(self.music_ids)} songs loaded.")

    def __len__(self):
        return len(self.music_ids)

    def __getitem__(self, idx: int) -> dict:
        music_id   = int(self.music_ids[idx])
        va_song    = self.va_groups[music_id]
        timestamps = va_song["frameTime"].values
        valence    = va_song["Valence(mean)"].values
        arousal    = va_song["Arousal(mean)"].values

        va_tensor = torch.from_numpy(
            np.stack([valence, arousal], axis=1).astype(np.float32)
        ).to(self.device)

        lrc_path  = os.path.join(self.lyrics_folder, f"{music_id}.lrc")
        lyric_emb = self.encoder(
            audio_timestamps = timestamps,
            lrc_path         = lrc_path,
        )   # (T, 771)

        return {"lyrics": lyric_emb, "va": va_tensor}


# =========================================================
# 6.  QUICK SANITY CHECK
# =========================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    LEXICON_PATH  = "/home/tanishkaunix/PMEmo2019/NRC-VAD-Lexicon-v2.1.txt"
    VA_PATH       = "/home/tanishkaunix/PMEmo2019/PMEmo2019/annotations/dynamic_annotations.csv"
    LYRICS_FOLDER = "/home/tanishkaunix/PMEmo2019/PMEmo2019/lyrics"

    dataset = LyricsWindowDataset(
        va_path        = VA_PATH,
        lyrics_folder  = LYRICS_FOLDER,
        lexicon_path   = LEXICON_PATH,
        window_sec     = 3.0,
        freeze_deberta = True,
        device         = device,
    )

    sample = dataset[0]
    print("lyrics shape :", sample["lyrics"].shape)   # (T, 771)
    print("va shape     :", sample["va"].shape)        # (T, 2)
    print("Hidden dim   :", LyricsWindowEncoder.HIDDEN_DIM)   # 771