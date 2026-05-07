#train1.py — Audio+Lyrics ablation training (no mel modality)
import os
import sys
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.getcwd())

from multimodal_datase2 import MultimodalPMEmoDataset
from models import MultimodalEmotionModel

print("Using dataset file:", MultimodalPMEmoDataset.__module__)

# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

# ============================================================
# Collate Function
# ============================================================

def collate_fn(batch):
    audio_list       = [item["audio"]   for item in batch]
    lyrics_list      = [item["lyrics"]  for item in batch]
    va_list          = [item["va"]      for item in batch]
    va_mean_list     = [item["va_mean"] for item in batch]
    va_std_list      = [item["va_std"]  for item in batch]
    
    # va_std is already clamped (min 1e-3) in the dataset — reuse as va_std_safe
    va_std_safe_list = va_std_list

    lengths = [a.shape[0] for a in audio_list]
    max_len = max(lengths)

    padded_audio, padded_lyrics, padded_va, masks = [], [], [], []

    for audio, lyrics, va in zip(audio_list, lyrics_list, va_list):
        T   = audio.shape[0]
        pad = max_len - T
        padded_audio.append(F.pad(audio,   (0, 0, 0, pad)))
        padded_lyrics.append(F.pad(lyrics, (0, 0, 0, pad)))
        padded_va.append(F.pad(va,         (0, 0, 0, pad)))
        mask = torch.zeros(max_len)
        mask[:T] = 1
        masks.append(mask)

    return {
        "audio":       torch.stack(padded_audio),
        "lyrics":      torch.stack(padded_lyrics),
        "va":          torch.stack(padded_va),        # z-scored, (B, T, 2)
        "va_mean":     torch.stack(va_mean_list),     # (B, 2)
        "va_std":      torch.stack(va_std_list),      # (B, 2)
        "va_std_safe": torch.stack(va_std_safe_list), # (B, 2) clamped
        "mask":        torch.stack(masks),
    }

# ============================================================
# Loss Functions
# ============================================================

def masked_mse(pred, target, mask):
    mask = mask.unsqueeze(-1)
    return ((pred - target) ** 2 * mask).sum() / (mask.sum() + 1e-8)

def smoothness_loss(pred, mask):
    diff      = (pred[:, 1:, :] - pred[:, :-1, :]).abs()
    mask_pair = (mask[:, :-1] * mask[:, 1:]).unsqueeze(-1)
    return (diff * mask_pair).sum() / (mask_pair.sum() + 1e-8)

def direction_loss(pred, true, mask):
    pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
    true_diff = true[:, 1:, :] - true[:, :-1, :]
    mask_pair = (mask[:, :-1] * mask[:, 1:]).unsqueeze(-1)
    return (torch.relu(-pred_diff * true_diff) * mask_pair).sum() / (mask_pair.sum() + 1e-8)

def per_song_ccc_loss(pred_dim, true_dim, mask):
    losses = []
    for b in range(pred_dim.shape[0]):
        valid = mask[b].bool()
        p, t  = pred_dim[b][valid], true_dim[b][valid]
        if len(p) < 2:
            continue
        mean_p, mean_t = p.mean(), t.mean()
        var_p  = p.var(unbiased=False)
        var_t  = t.var(unbiased=False)
        cov    = ((p - mean_p) * (t - mean_t)).mean()
        ccc    = (2 * cov) / (var_p + var_t + (mean_p - mean_t) ** 2 + 1e-8)
        losses.append(1 - ccc)
    if not losses:
        return torch.tensor(0.0, device=pred_dim.device)
    return torch.stack(losses).mean()

def sliding_window_ccc_loss(pred, true, mask, window_size=8):
    B, T, C = pred.shape
    W = window_size
    if T < W:
        return torch.tensor(0.0, device=pred.device)
    pred_w = pred.unfold(1, W, 1).permute(0, 1, 3, 2)
    true_w = true.unfold(1, W, 1).permute(0, 1, 3, 2)
    mask_w = mask.unfold(1, W, 1).unsqueeze(-1)
    denom  = mask_w.sum(dim=2) + 1e-8
    mean_p = (pred_w * mask_w).sum(dim=2) / denom
    mean_t = (true_w * mask_w).sum(dim=2) / denom
    var_p  = ((pred_w - mean_p.unsqueeze(2)) ** 2 * mask_w).sum(dim=2) / denom
    var_t  = ((true_w - mean_t.unsqueeze(2)) ** 2 * mask_w).sum(dim=2) / denom
    cov    = ((pred_w - mean_p.unsqueeze(2)) * (true_w - mean_t.unsqueeze(2)) * mask_w).sum(dim=2) / denom
    ccc    = (2 * cov) / (var_p + var_t + (mean_p - mean_t) ** 2 + 1e-8)
    valid  = mask_w.sum(dim=2).squeeze(-1) >= W * 0.5
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return (1 - ccc[valid.unsqueeze(-1).expand_as(ccc)]).mean()

def variance_floor_loss(pred, true, mask):
    mask3  = mask.unsqueeze(-1)
    denom  = mask3.sum(dim=1, keepdim=True) + 1e-8
    mean_p = (pred * mask3).sum(dim=1, keepdim=True) / denom
    mean_t = (true * mask3).sum(dim=1, keepdim=True) / denom
    var_p  = ((pred - mean_p) ** 2 * mask3).sum(dim=1) / denom.squeeze(1)
    var_t  = ((true - mean_t) ** 2 * mask3).sum(dim=1) / denom.squeeze(1)
    return torch.clamp(var_t - var_p, min=0).mean()

def pearson_loss(pred, true, mask):
    losses = []
    for b in range(pred.shape[0]):
        valid = mask[b].bool()
        p = pred[b][valid]
        t = true[b][valid]
        if len(p) < 2:
            continue
        for dim in range(2):
            pd = p[:, dim] - p[:, dim].mean()
            td = t[:, dim] - t[:, dim].mean()
            pcc = torch.sum(pd * td) / (
                torch.sqrt(torch.sum(pd ** 2)) *
                torch.sqrt(torch.sum(td ** 2)) + 1e-8
            )
            losses.append(1 - pcc)
    if not losses:
        return torch.tensor(0.0, device=pred.device)
    return torch.stack(losses).mean()

def mean_bias_loss(pred, true, mask):
    losses = []
    for b in range(pred.shape[0]):
        valid = mask[b].bool()
        if valid.sum() < 2:
            continue
        p = pred[b][valid]
        t = true[b][valid]
        for dim in range(2):
            offset = p[:, dim].mean() - t[:, dim].mean()
            losses.append(offset ** 2)
    if not losses:
        return torch.tensor(0.0, device=pred.device)
    return torch.stack(losses).mean()

# ============================================================
# Composite loss
# ============================================================

LOSS_WEIGHTS = {
    "mse":       0.015,
    "ccc_v":     1.5,
    "ccc_a":     3.5,   
    "local_ccc": 1.2,   
    "pearson":   2.5,
    "mean_bias": 1.5,   
    "var_floor": 1.4,   
    "smooth":    0.01,  
    "dir":       0.08,  
}

MSE_ONLY_EPOCHS = 5

def compute_loss(va_pred, va_true, mask, epoch=0):
    mse       = masked_mse(va_pred, va_true, mask)
    ccc_v     = per_song_ccc_loss(va_pred[..., 0], va_true[..., 0], mask)
    ccc_a     = per_song_ccc_loss(va_pred[..., 1], va_true[..., 1], mask)
    local_ccc = sliding_window_ccc_loss(va_pred, va_true, mask)
    pearson   = pearson_loss(va_pred, va_true, mask)
    mean_bias = mean_bias_loss(va_pred, va_true, mask)
    var_floor = variance_floor_loss(va_pred, va_true, mask)
    smooth    = smoothness_loss(va_pred, mask)
    dir_loss  = direction_loss(va_pred, va_true, mask)

    if epoch < MSE_ONLY_EPOCHS:
        total = mse
    else:
        total = (
            LOSS_WEIGHTS["mse"]       * mse
            + LOSS_WEIGHTS["ccc_v"]     * ccc_v
            + LOSS_WEIGHTS["ccc_a"]     * ccc_a
            + LOSS_WEIGHTS["local_ccc"] * local_ccc
            + LOSS_WEIGHTS["pearson"]   * pearson
            + LOSS_WEIGHTS["mean_bias"] * mean_bias
            + LOSS_WEIGHTS["var_floor"] * var_floor
            + LOSS_WEIGHTS["smooth"]    * smooth
            + LOSS_WEIGHTS["dir"]       * dir_loss
        )

    terms = dict(
        mse=mse.item(), ccc_v=ccc_v.item(), ccc_a=ccc_a.item(),
        local_ccc=local_ccc.item(), pearson=pearson.item(),
        mean_bias=mean_bias.item(), var_floor=var_floor.item(),
        smooth=smooth.item(), dir=dir_loss.item(),
        phase=1 if epoch < MSE_ONLY_EPOCHS else 2,
    )
    return total, terms

# ============================================================
# Metric functions (flat, reference-style)
# ============================================================

def ccc_metric(x, y):
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    sxy    = torch.sum((x - x_mean) * (y - y_mean)) / x.shape[0]
    return (2 * sxy) / (
        torch.var(x, unbiased=False)
        + torch.var(y, unbiased=False)
        + (x_mean - y_mean) ** 2
    )

def pcc_metric(p, y):
    p = p - torch.mean(p)
    y = y - torch.mean(y)
    return torch.sum(p * y) / (
        torch.sqrt(torch.sum(p ** 2)) * torch.sqrt(torch.sum(y ** 2)) + 1e-8
    )

def rmse_metric(p, y):
    return torch.sqrt(torch.mean((p - y) ** 2))

# ============================================================
# Scheduler
# ============================================================

def build_scheduler(optimizer, warmup_epochs, total_epochs):
    T_0 = 25  

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        e            = epoch - warmup_epochs
        cycle_num    = e // T_0
        pos_in_cycle = e % T_0
        peak         = 0.5 ** cycle_num
        return peak * 0.5 * (1.0 + math.cos(math.pi * pos_in_cycle / T_0))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ============================================================
# NaN diagnostic
# ============================================================

def check_nan_in_model(model, tag=""):
    for name, p in model.named_parameters():
        if p.grad is not None and torch.isnan(p.grad).any():
            print(f"  [NaN grad]   {tag} → {name}")
            return True
        if torch.isnan(p).any():
            print(f"  [NaN weight] {tag} → {name}")
            return True
    return False

# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(model, loader, optimizer, device, epoch=0, GRAD_CLIP=2.0):
    model.train()
    total_loss = 0.0
    term_accum = {k: 0.0 for k in list(LOSS_WEIGHTS.keys()) + ["phase"]}
    nan_batches = 0

    for p in model.head.smoother.parameters():
        p.requires_grad = (epoch >= MSE_ONLY_EPOCHS)

    for batch_idx, batch in enumerate(loader):
        audio   = batch["audio"].to(device)
        lyrics  = batch["lyrics"].to(device)
        va_true = batch["va"].to(device)
        mask    = batch["mask"].to(device)

        optimizer.zero_grad()

        out     = model(audio, lyrics, mask=mask)
        va_pred = out["va_pred"]

        loss, terms = compute_loss(va_pred, va_true, mask, epoch=epoch)

        if torch.isnan(loss):
            nan_batches += 1
            if batch_idx < 3:
                print(f"  [WARN] NaN loss at batch {batch_idx}")
                check_nan_in_model(model, tag=f"batch {batch_idx}")
            optimizer.zero_grad()
            continue

        loss.backward()

        mamba_grads = [p for n, p in model.named_parameters()
                       if "mamba" in n.lower() and p.grad is not None]
        other_grads = [p for n, p in model.named_parameters()
                       if "mamba" not in n.lower() and p.grad is not None]
        torch.nn.utils.clip_grad_norm_(mamba_grads, 0.5)
        torch.nn.utils.clip_grad_norm_(other_grads, GRAD_CLIP)

        for name, p in model.named_parameters():
            if "a_log" in name.lower() and p.grad is not None:
                p.grad.data.clamp_(-0.1, 0.1)

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        if grad_norm > 20.0:
            pass # Suppress warning to keep console clean

        optimizer.step()

        total_loss += loss.item()
        for k, v in terms.items():
            term_accum[k] += v

    n_valid = len(loader) - nan_batches
    if n_valid == 0:
        print("  [ERROR] ALL batches NaN.")
        return float("nan"), {k: float("nan") for k in LOSS_WEIGHTS}

    return total_loss / n_valid, {k: v / n_valid for k, v in term_accum.items()}

# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader, device, epoch=MSE_ONLY_EPOCHS):
    model.eval()
    total_loss = 0.0
    term_accum = {k: 0.0 for k in list(LOSS_WEIGHTS.keys()) + ["phase"]}

    ccc_v_list,  ccc_a_list  = [], []
    pcc_v_list,  pcc_a_list  = [], []
    rmse_v_list, rmse_a_list = [], []

    with torch.no_grad():
        for batch in loader:
            audio   = batch["audio"].to(device)
            lyrics  = batch["lyrics"].to(device)
            va_true = batch["va"].to(device)
            va_mean = batch["va_mean"].to(device)   
            va_std  = batch["va_std"].to(device)    
            mask    = batch["mask"].to(device)

            out     = model(audio, lyrics, mask=mask)
            va_pred = out["va_pred"]

            loss, terms = compute_loss(va_pred, va_true, mask, epoch=epoch)
            total_loss += loss.item()
            for k, v in terms.items():
                term_accum[k] += v

            for b in range(audio.shape[0]):
                valid = mask[b].bool()
                if valid.sum() < 2:
                    continue

                pred_z = va_pred[b][valid].cpu()    
                true_z = va_true[b][valid].cpu()    

                mean_b = va_mean[b].cpu()           
                std_b  = va_std[b].cpu()            
                pred_orig = pred_z * std_b + mean_b
                true_orig = true_z * std_b + mean_b

                pv, pa = pred_orig[:, 0], pred_orig[:, 1]
                tv, ta = true_orig[:, 0], true_orig[:, 1]

                ccc_v_list.append(ccc_metric(pv, tv).item())
                ccc_a_list.append(ccc_metric(pa, ta).item())
                pcc_v_list.append(pcc_metric(pv, tv).item())
                pcc_a_list.append(pcc_metric(pa, ta).item())
                rmse_v_list.append(rmse_metric(pv, tv).item())
                rmse_a_list.append(rmse_metric(pa, ta).item())

    ccc_v  = float(np.mean(ccc_v_list))  if ccc_v_list  else 0.0
    ccc_a  = float(np.mean(ccc_a_list))  if ccc_a_list  else 0.0
    pcc_v  = float(np.mean(pcc_v_list))  if pcc_v_list  else 0.0
    pcc_a  = float(np.mean(pcc_a_list))  if pcc_a_list  else 0.0
    rmse_v = float(np.mean(rmse_v_list)) if rmse_v_list else 0.0
    rmse_a = float(np.mean(rmse_a_list)) if rmse_a_list else 0.0

    n = len(loader)
    return (
        total_loss / n,
        {k: v / n for k, v in term_accum.items()},
        ccc_v, ccc_a, pcc_v, pcc_a, rmse_v, rmse_a,
    )


# ============================================================
# EXPERIMENT EXECUTION (ONE SEED)
# ============================================================

TOTAL_EPOCHS  = 60
WARMUP_EPOCHS = 5
BATCH_SIZE    = 4
LR            = 5e-5
LR_MAMBA      = 1e-5
GRAD_CLIP     = 2.0

def run_experiment(seed, dataset, device):
    """Runs the full training/val pipeline and tests on a completely held-out 10% split."""
    print(f"\n{'='*50}")
    print(f"🚀 STARTING RUN WITH SEED: {seed}")
    print(f"{'='*50}")
    set_seed(seed)

    # 1. NEW 80/10/10 Split Logic
    total_len = len(dataset)
    test_size  = int(0.1 * total_len)  # 10% Test
    val_size   = int(0.1 * total_len)  # 10% Validation
    train_size = total_len - val_size - test_size # 80% Train
    
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn) # The Untouched Set!

    model = MultimodalEmotionModel().to(device)

    refiner1_mamba, refiner2_mamba, other_params = [], [], []
    for name, p in model.named_parameters():
        if "refiner1" in name and "mamba" in name.lower():
            refiner1_mamba.append(p)
        elif "refiner2" in name and "mamba" in name.lower():
            refiner2_mamba.append(p)
        else:
            other_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": other_params,   "lr": LR,           "betas": (0.9, 0.98), "eps": 1e-6, "weight_decay": 1e-4},
        {"params": refiner2_mamba, "lr": LR_MAMBA,     "betas": (0.95, 0.98), "eps": 1e-6, "weight_decay": 1e-4},
        {"params": refiner1_mamba, "lr": LR_MAMBA*0.5, "betas": (0.95, 0.98), "eps": 1e-6, "weight_decay": 2e-4},
    ])

    scheduler = build_scheduler(optimizer, WARMUP_EPOCHS, TOTAL_EPOCHS)

    best_ccc_a = -1.0
    best_model_path = f"best_model_seed_{seed}.pt"

    for epoch in range(TOTAL_EPOCHS):

        if epoch == MSE_ONLY_EPOCHS:
            with torch.no_grad():
                model.head.smoother.weight.zero_()
                model.head.smoother.weight[:, :, 2] = 1.0
                if model.head.smoother.bias is not None:
                    model.head.smoother.bias.zero_()
            print("  [INFO] Smoother reset to identity")

        train_loss, train_terms = train_one_epoch(
            model, train_loader, optimizer, device, epoch=epoch, GRAD_CLIP=GRAD_CLIP
        )

        val_loss, val_terms, ccc_v, ccc_a, pcc_v, pcc_a, rmse_v, rmse_a = evaluate(
            model, val_loader, device, epoch=epoch
        )

        scheduler.step()

        # Save based on best Validation CCC-A
        if ccc_a > best_ccc_a:
            best_ccc_a = ccc_a
            torch.save(model.state_dict(), best_model_path)
            
        # Print progress sparingly
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"Seed {seed} | Epoch {epoch+1:02d}/{TOTAL_EPOCHS} | Val CCC-V: {ccc_v:.4f} | Val CCC-A: {ccc_a:.4f} "
                  f"(Best Val CCC-A: {best_ccc_a:.4f})")

    # ---------------------------------------------------------
    # 2. FINAL TEST EVALUATION (On the completely hidden 10% split)
    # ---------------------------------------------------------
    print(f"\n--- Loading Best Model for Seed {seed} ---")
    model.load_state_dict(torch.load(best_model_path))
    
    # Evaluate the best model on the untouched TEST set
    _, _, test_ccc_v, test_ccc_a, test_pcc_v, test_pcc_a, test_rmse_v, test_rmse_a = evaluate(model, test_loader, device, epoch=TOTAL_EPOCHS)
    
    print(f"🏁 TEST RESULTS (SEED {seed}):")
    print(f"   CCC-V: {test_ccc_v:.4f} | CCC-A: {test_ccc_a:.4f}")
    
    return {
        "ccc_v": test_ccc_v, "ccc_a": test_ccc_a,
        "pcc_v": test_pcc_v, "pcc_a": test_pcc_a,
        "rmse_v": test_rmse_v, "rmse_a": test_rmse_a
    }

# ============================================================
# MAIN ORCHESTRATOR (Multiple Seeds)
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load dataset ONCE
    dataset = MultimodalPMEmoDataset(
        feature_path  = "/home/yashkale/MER_VER3/PMEmo2019/features/dynamic_features.csv",
        va_path       = "/home/yashkale/MER_VER3/PMEmo2019/annotations/dynamic_annotations.csv",
        lyrics_folder = "/home/yashkale/MER_VER3/PMEmo2019/lyrics",
        device        = device,
    )

    # ---------------------------------------------------------
    # 3. RUN MULTIPLE SEEDS
    # ---------------------------------------------------------
    seeds = [42, 123, 2024]
    all_results = []

    for seed in seeds:
        res = run_experiment(seed, dataset, device)
        all_results.append(res)

    # Calculate Mean & Standard Deviation
    print(f"\n{'='*50}")
    print("🏆 FINAL TEST SET RESULTS ACROSS ALL SEEDS")
    print(f"{'='*50}")
    
    metrics = ["ccc_v", "ccc_a", "pcc_v", "pcc_a", "rmse_v", "rmse_a"]
    
    for metric in metrics:
        values = [r[metric] for r in all_results]
        mean_val = np.mean(values)
        std_val  = np.std(values)
        print(f"{metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")

if __name__ == "__main__":
    main()