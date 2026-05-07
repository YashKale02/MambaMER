"""
pretrain_lyric_encoder.py
=========================
Pretraining pipeline for the lyric encoder.

Goal  : Teach the DeBERTa-v3-base encoder to regress sentence-level VAD scores
        before plugging it into the full multimodal emotion model.

Data sources supported
----------------------
  1. NRC-VAD-Lexicon  — every lexicon entry is treated as a (term → VAD) sample.
                        Sentence-level VAD is the mean of constituent token VADs,
                        keeping only tokens that exist in the lexicon.

Architecture
------------
  DebertaV2Model (microsoft/deberta-v3-base, optionally frozen)
  └─ AttentionPool  (learned soft attention over token embeddings)
  └─ VADRegressionHead
       Linear(768 → 128)   # DeBERTa hidden dim
       GELU
       Linear(128 → 3)     # predict V, A, D

  NOTE: VAD-from-lexicon concatenation (→ 771-dim) is done in the downstream
  lyrics.py. Here we pretrain the *encoder + pooler* so that the pooled
  representation already captures sentence-level VAD structure.

Loss
----
  final = 0.7 * mean_CCC(V, A, D) + 0.3 * MSE(V, A, D)

Training
--------
  AdamW + linear warmup + cosine decay
  Gradient clipping (max norm 1.0)
  Best encoder checkpoint saved by validation mean CCC.

Checkpoint format (best_encoder.pt)
------------------------------------
  {
    "deberta"  : model.deberta.state_dict(),
    "attn_pool": model.attn_pool.state_dict(),
  }
  This is the format expected by load_pretrained_lyric_encoder() in models.py.
"""

import os
import math
import random
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import DebertaV2TokenizerFast, DebertaV2Model
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ═════════════════════════════════════════════════════════════
# 1.  NRC-VAD LEXICON  (shared with lyrics.py)
# ═════════════════════════════════════════════════════════════

class NRCVADLexicon:
    """Tab-separated NRC-VAD file → {term: np.array([V, A, D])}."""

    def __init__(self, path: str):
        self.vad: dict[str, np.ndarray] = {}
        df = pd.read_csv(path, sep="\t", header=0,
                         names=["term", "valence", "arousal", "dominance"],
                         skiprows=1)
        for _, row in df.iterrows():
            key = str(row["term"]).strip().lower()
            self.vad[key] = np.array(
                [float(row["valence"]), float(row["arousal"]), float(row["dominance"])],
                dtype=np.float32
            )

    def lookup(self, token: str) -> np.ndarray | None:
        return self.vad.get(token.strip().lower(), None)

    def lookup_batch(self, tokens: list[str]) -> np.ndarray:
        return np.stack(
            [self.vad.get(t.strip().lower(), np.zeros(3, dtype=np.float32))
             for t in tokens],
            axis=0
        )

    def sentence_vad(self, tokens: list[str]) -> np.ndarray | None:
        """Mean VAD of tokens that exist in lexicon; None if none found."""
        hits = [self.vad[t.lower()] for t in tokens if t.lower() in self.vad]
        return np.mean(hits, axis=0).astype(np.float32) if hits else None


# ═════════════════════════════════════════════════════════════
# 2.  DATASET
# ═════════════════════════════════════════════════════════════

class NRCVADSentenceDataset(Dataset):
    """
    Constructs sentence-level samples directly from the NRC-VAD lexicon.
    Each multi-word entry is used as a phrase; single-word entries are kept too.
    Target VAD = the lexicon entry's own scores.
    This gives ~54 k training sentences with ground-truth VAD.
    """

    def __init__(self, lexicon: NRCVADLexicon, min_tokens: int = 1):
        self.samples: list[tuple[str, np.ndarray]] = []
        for term, vad in lexicon.vad.items():
            if len(term.split()) >= min_tokens:
                self.samples.append((term, vad))
        log.info("NRCVADSentenceDataset: %d samples", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        text, vad = self.samples[idx]
        return {"text": text, "vad": torch.from_numpy(vad)}


# ═════════════════════════════════════════════════════════════
# 3.  COLLATE
# ═════════════════════════════════════════════════════════════

def make_collate_fn(tokenizer, max_length: int = 128):
    """Returns a collate function that tokenises on-the-fly (GPU-friendly)."""

    def collate(batch: list[dict]) -> dict:
        texts = [item["text"] for item in batch]
        vads  = torch.stack([item["vad"] for item in batch])   # (B, 3)

        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "token_type_ids": enc.get("token_type_ids", None),  # DeBERTa-v3 omits this
            "vad":            vads,
        }

    return collate


# ═════════════════════════════════════════════════════════════
# 4.  MODEL
# ═════════════════════════════════════════════════════════════

class AttentionPool(nn.Module):
    """
    Soft attention pooling over token embeddings.
    score_i = tanh(W · h_i);  weight_i = softmax(score_i)
    output  = Σ weight_i · h_i
    Respects padding via attention_mask.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1, bias=True)

    def forward(
        self,
        hidden         : torch.Tensor,   # (B, T, H)
        attention_mask : torch.Tensor,   # (B, T)  1=real, 0=pad
    ) -> torch.Tensor:                   # (B, H)
        scores  = self.proj(torch.tanh(hidden)).squeeze(-1)         # (B, T)
        scores  = scores.masked_fill(attention_mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)           # (B, T, 1)
        return (weights * hidden).sum(dim=1)                         # (B, H)


class VADRegressionHead(nn.Module):
    """Linear → GELU → Linear  mapping hidden_dim → 3."""

    def __init__(self, hidden_dim: int, mid_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, 3)


class LyricVADEncoder(nn.Module):
    """
    Full pretraining model:
        DeBERTa-v3-base → AttentionPool → VADRegressionHead → VAD predictions

    The hidden dim is 768 (DeBERTa-base). VAD-from-lexicon concatenation
    (making it 771) is handled downstream in lyrics.py. Here we pretrain
    the encoder + pooler so the pooled representation already captures
    sentence-level VAD structure before the full multimodal model is trained.

    Parameters
    ----------
    deberta_model   : HuggingFace model name (default: microsoft/deberta-v3-base)
    freeze_deberta  : freeze DeBERTa weights during pretraining (default False —
                      we want to fine-tune the encoder here)
    """

    DEBERTA_DIM = 768

    def __init__(
        self,
        deberta_model  : str  = "microsoft/deberta-v3-base",
        freeze_deberta : bool = False,
    ):
        super().__init__()
        self.deberta   = DebertaV2Model.from_pretrained(deberta_model)
        self.attn_pool = AttentionPool(self.DEBERTA_DIM)
        self.head      = VADRegressionHead(self.DEBERTA_DIM, mid_dim=128)

        if freeze_deberta:
            for p in self.deberta.parameters():
                p.requires_grad = False

    def encode(
        self,
        input_ids      : torch.Tensor,
        attention_mask : torch.Tensor,
    ) -> torch.Tensor:
        """Return pooled sentence embedding (B, 768). Differentiable."""
        # DeBERTa-v3 does not use token_type_ids — passing None is safe
        out    = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state                      # (B, T, 768)
        return self.attn_pool(hidden, attention_mask)       # (B, 768)

    def forward(
        self,
        input_ids      : torch.Tensor,
        attention_mask : torch.Tensor,
    ) -> torch.Tensor:
        """Return predicted VAD (B, 3)."""
        pooled = self.encode(input_ids, attention_mask)
        return self.head(pooled)


# ═════════════════════════════════════════════════════════════
# 5.  LOSSES
# ═════════════════════════════════════════════════════════════

def ccc_loss_single(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """
    Concordance Correlation Coefficient loss for one dimension.
    CCC ∈ [-1, 1]; loss = 1 - CCC  so lower is better.
    """
    pred = pred.view(-1)
    true = true.view(-1)

    mu_p  = pred.mean()
    mu_t  = true.mean()
    var_p = pred.var(unbiased=False)
    var_t = true.var(unbiased=False)
    cov   = ((pred - mu_p) * (true - mu_t)).mean()

    ccc = (2.0 * cov) / (var_p + var_t + (mu_p - mu_t) ** 2 + 1e-8)
    return 1.0 - ccc


def combined_loss(
    pred   : torch.Tensor,   # (B, 3)
    target : torch.Tensor,   # (B, 3)
    alpha  : float = 0.7,
) -> tuple[torch.Tensor, dict]:
    """
    final = alpha * mean_CCC(V,A,D)  +  (1-alpha) * MSE(V,A,D)
    Returns (loss_tensor, metrics_dict).
    """
    ccc_v = ccc_loss_single(pred[:, 0], target[:, 0])
    ccc_a = ccc_loss_single(pred[:, 1], target[:, 1])
    ccc_d = ccc_loss_single(pred[:, 2], target[:, 2])
    mean_ccc_loss = (ccc_v + ccc_a + ccc_d) / 3.0

    mse  = F.mse_loss(pred, target)
    loss = alpha * mean_ccc_loss + (1.0 - alpha) * mse

    metrics = {
        "loss":     loss.item(),
        "ccc_loss": mean_ccc_loss.item(),
        "mse":      mse.item(),
        "ccc_v":    (1.0 - ccc_v).item(),
        "ccc_a":    (1.0 - ccc_a).item(),
        "ccc_d":    (1.0 - ccc_d).item(),
    }
    return loss, metrics


# ═════════════════════════════════════════════════════════════
# 6.  SCHEDULER  — linear warmup + cosine decay
# ═════════════════════════════════════════════════════════════

def get_warmup_cosine_scheduler(
    optimizer    : torch.optim.Optimizer,
    warmup_steps : int,
    total_steps  : int,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(warmup_steps, 1)
        progress = float(step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ═════════════════════════════════════════════════════════════
# 7.  TRAIN / VALIDATE
# ═════════════════════════════════════════════════════════════

def train_one_epoch(
    model     : LyricVADEncoder,
    loader    : DataLoader,
    optimizer : torch.optim.Optimizer,
    scheduler : LambdaLR,
    device    : torch.device,
    clip_norm : float = 1.0,
    alpha     : float = 0.7,
) -> dict:
    model.train()
    totals: dict[str, float] = {}
    n_batches = 0

    for batch in loader:
        input_ids  = batch["input_ids"].to(device)
        attn_mask  = batch["attention_mask"].to(device)
        vad_true   = batch["vad"].to(device)

        optimizer.zero_grad()

        vad_pred = model(input_ids, attn_mask)
        loss, metrics = combined_loss(vad_pred, vad_true, alpha=alpha)

        if torch.isnan(loss):
            log.warning("NaN loss — skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        scheduler.step()

        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def validate(
    model  : LyricVADEncoder,
    loader : DataLoader,
    device : torch.device,
    alpha  : float = 0.7,
) -> dict:
    model.eval()

    all_pred: list[torch.Tensor] = []
    all_true: list[torch.Tensor] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        vad_true  = batch["vad"].to(device)

        vad_pred  = model(input_ids, attn_mask)
        all_pred.append(vad_pred.cpu())
        all_true.append(vad_true.cpu())

    pred = torch.cat(all_pred, dim=0)   # (N, 3)
    true = torch.cat(all_true, dim=0)

    _, metrics = combined_loss(pred, true, alpha=alpha)
    return metrics


# ═════════════════════════════════════════════════════════════
# 8.  MAIN  TRAINING LOOP
# ═════════════════════════════════════════════════════════════

def build_datasets(args) -> Dataset:
    """Build dataset using only NRC-VAD lexicon."""
    if args.nrc_vad_path and os.path.exists(args.nrc_vad_path):
        lexicon = NRCVADLexicon(args.nrc_vad_path)
        return NRCVADSentenceDataset(lexicon, min_tokens=1)
    else:
        raise RuntimeError(f"NRC-VAD lexicon not found at '{args.nrc_vad_path}'. Provide a valid path.")


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Tokeniser ────────────────────────────────────────────
    # DebertaV2TokenizerFast is required for deberta-v3 models;
    # it uses SentencePiece and does NOT produce token_type_ids.
    tokenizer = DebertaV2TokenizerFast.from_pretrained(args.deberta_model)
    collate   = make_collate_fn(tokenizer, max_length=args.max_length)

    # ── Data ─────────────────────────────────────────────────
    full_ds = build_datasets(args)
    n_total = len(full_ds)
    n_val   = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    log.info("Train: %d  |  Val: %d", n_train, n_val)

    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        collate_fn  = collate,
        pin_memory  = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = args.batch_size * 2,
        shuffle     = False,
        num_workers = args.num_workers,
        collate_fn  = collate,
        pin_memory  = True,
    )

    # ── Model ────────────────────────────────────────────────
    model = LyricVADEncoder(
        deberta_model  = args.deberta_model,
        freeze_deberta = args.freeze_deberta,
    ).to(device).float()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Trainable parameters: %s", f"{n_params:,}")

    # ── Optimiser ────────────────────────────────────────────
    # DeBERTa benefits from separate LR for the encoder vs. the head.
    # We also apply no weight decay to bias and LayerNorm parameters.
    encoder_params, head_params, no_decay = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "LayerNorm" in name or "layer_norm" in name:
            no_decay.append(param)
        elif name.startswith("deberta"):
            encoder_params.append(param)
        else:
            head_params.append(param)

    optimizer = AdamW(
        [
            {"params": encoder_params, "lr": args.lr,            "weight_decay": args.weight_decay},
            {"params": head_params,    "lr": args.lr * 10,        "weight_decay": args.weight_decay},
            {"params": no_decay,       "lr": args.lr,             "weight_decay": 0.0},
        ],
        eps=1e-8,
    )

    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler    = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)
    log.info("Total steps: %d  |  Warmup: %d", total_steps, warmup_steps)

    # ── Output dir ───────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Training Loop ────────────────────────────────────────
    best_val_ccc  = -float("inf")
    best_val_loss =  float("inf")

    log.info("Starting pretraining for %d epochs …", args.epochs)

    for epoch in range(1, args.epochs + 1):

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            clip_norm=args.clip_norm, alpha=args.alpha,
        )
        val_metrics = validate(model, val_loader, device, alpha=args.alpha)

        mean_ccc = (val_metrics["ccc_v"] + val_metrics["ccc_a"] + val_metrics["ccc_d"]) / 3.0

        log.info(
            "Epoch %3d/%d | "
            "Train loss %.4f (CCC %.4f | MSE %.4f) | "
            "Val loss %.4f | CCC V/A/D %.4f/%.4f/%.4f | mean %.4f",
            epoch, args.epochs,
            train_metrics["loss"], train_metrics["ccc_loss"], train_metrics["mse"],
            val_metrics["loss"],
            val_metrics["ccc_v"], val_metrics["ccc_a"], val_metrics["ccc_d"],
            mean_ccc,
        )

        # ── Checkpoint: encoder + pooler weights only ────────
        # Key names match what lyrics.py / models.py expect when loading.
        if mean_ccc > best_val_ccc:
            best_val_ccc  = mean_ccc
            best_val_loss = val_metrics["loss"]

            encoder_state = {
                "deberta"  : model.deberta.state_dict(),
                "attn_pool": model.attn_pool.state_dict(),
                # head weights intentionally omitted — not needed downstream
            }
            torch.save(encoder_state, out_dir / "best_encoder.pt")
            log.info(
                "  ✓ Best encoder saved  (val mean CCC = %.4f, loss = %.4f)",
                best_val_ccc, best_val_loss,
            )

        # ── Periodic full checkpoint ─────────────────────────
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(
                {
                    "epoch":      epoch,
                    "model":      model.state_dict(),
                    "optimizer":  optimizer.state_dict(),
                    "scheduler":  scheduler.state_dict(),
                    "best_ccc":   best_val_ccc,
                },
                out_dir / f"checkpoint_epoch{epoch:03d}.pt",
            )

    log.info(
        "Pretraining complete. Best val mean CCC = %.4f  |  loss = %.4f",
        best_val_ccc, best_val_loss,
    )
    log.info("Encoder weights saved to: %s", out_dir / "best_encoder.pt")


# ═════════════════════════════════════════════════════════════
# 9.  LOADING HELPER  (for downstream use in lyrics.py)
# ═════════════════════════════════════════════════════════════

def load_pretrained_encoder(
    checkpoint_path : str,
    deberta_model   : str = "microsoft/deberta-v3-base",
    device          : str = "cuda",
) -> LyricVADEncoder:
    """
    Instantiate LyricVADEncoder and load pretrained encoder weights.
    The regression head is re-initialised (not used downstream).

    Usage in lyrics.py
    ------------------
        from pretrain_lyric_encoder import load_pretrained_encoder
        encoder_model = load_pretrained_encoder("checkpoints/best_encoder.pt")
        # encoder_model.deberta  and  encoder_model.attn_pool  are warm-started
    """
    model = LyricVADEncoder(deberta_model=deberta_model, freeze_deberta=False)
    state = torch.load(checkpoint_path, map_location=device)
    model.deberta.load_state_dict(state["deberta"])
    model.attn_pool.load_state_dict(state["attn_pool"])
    model.to(device)
    log.info("Loaded pretrained DeBERTa encoder from %s", checkpoint_path)
    return model


# ═════════════════════════════════════════════════════════════
# 10. ENTRY POINT
# ═════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Pretrain DeBERTa lyric encoder on VAD regression")

    # Data
    p.add_argument("--nrc_vad_path",      default="/home/yashkale/MER_VER3/NRC-VAD-Lexicon-v2.1.txt",
                   help="Path to NRC-VAD lexicon (.txt)")
    p.add_argument("--val_split",         type=float, default=0.1)

    # Model
    p.add_argument("--deberta_model",     default="microsoft/deberta-v3-base",
                   help="HuggingFace DeBERTa model name")
    p.add_argument("--freeze_deberta",    action="store_true", default=False,
                   help="Freeze DeBERTa weights (train head + pool only)")
    p.add_argument("--max_length",        type=int, default=128)

    # Training
    p.add_argument("--epochs",            type=int,   default=10)
    p.add_argument("--batch_size",        type=int,   default=64)
    p.add_argument("--lr",                type=float, default=2e-5,
                   help="Base LR for DeBERTa encoder; head uses lr*10")
    p.add_argument("--weight_decay",      type=float, default=1e-2)
    p.add_argument("--warmup_ratio",      type=float, default=0.06,
                   help="Fraction of total steps used for LR warmup")
    p.add_argument("--clip_norm",         type=float, default=1.0)
    p.add_argument("--alpha",             type=float, default=0.7,
                   help="CCC loss weight (final = alpha*CCC + (1-alpha)*MSE)")
    p.add_argument("--num_workers",       type=int,   default=4)

    # Output
    p.add_argument("--output_dir",        default="checkpoints/lyric_pretrain")
    p.add_argument("--save_every",        type=int,   default=5)
    p.add_argument("--seed",              type=int,   default=42)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)