import torch
import torch.nn as nn
import sys

class CoAttentionFusion(nn.Module):
    def __init__(self, d_model=256, n_heads=4, dropout=0.1):
        """
        ViLBERT-style co-attentional transformer layer.

        KEY FIXES vs previous version
        ──────────────────────────────
        1. Mask polarity: collate_fn produces mask where 1=valid, 0=pad.
           PyTorch MHA key_padding_mask expects True=IGNORE (pad).
           Fixed by inverting: key_pad_mask = ~mask.bool()

        2. FFN sublayers added per ViLBERT Co-TRM spec.
           Previous version had only: cross-attn → Add&Norm
           Correct structure:         cross-attn → Add&Norm → FFN → Add&Norm
           Each stream gets its own FFN so V and A can diverge.
        """
        super().__init__()
        self.d_model = d_model

        # ── Co-attention (cross-modal MHA) ───────────────────────
        self.attn_A2L = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            batch_first=True, dropout=dropout
        )
        self.attn_L2A = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            batch_first=True, dropout=dropout
        )

        # ── Post-attention norms (one per stream) ─────────────────
        self.norm_A1 = nn.LayerNorm(d_model)
        self.norm_L1 = nn.LayerNorm(d_model)

        # ── FFN sublayers (ViLBERT Co-TRM second half) ───────────
        # Each stream: Linear(d→2d) → GELU → Dropout → Linear(2d→d)
        self.ff_A = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.ff_L = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

        # ── Post-FFN norms ────────────────────────────────────────
        self.norm_A2 = nn.LayerNorm(d_model)
        self.norm_L2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        # ── Final gated merge ─────────────────────────────────────
        self.fusion_proj = nn.Linear(d_model * 2, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)

    def forward(self, A: torch.Tensor, L: torch.Tensor, mask: torch.Tensor = None):
        """
        A    : (B, T, d_model) — audio+mel features
        L    : (B, T, d_model) — lyrics features
        mask : (B, T) float — 1=valid frame, 0=padding  (collate_fn convention)

        Returns: (B, T, d_model)
        """
        # ── Convert mask polarity ─────────────────────────────────
        # collate_fn: 1=valid, 0=pad
        # MHA key_padding_mask: True=IGNORE(pad), False=attend(valid)
        key_pad_mask = (~mask.bool()) if mask is not None else None

        # ── Cross-attention block (audio → lyrics) ────────────────
        attn_A2L, _ = self.attn_A2L(
            query=A, key=L, value=L,
            key_padding_mask=key_pad_mask
        )
        out_A = self.norm_A1(A + self.dropout(attn_A2L))   # Add & Norm

        # ── Cross-attention block (lyrics → audio) ────────────────
        attn_L2A, _ = self.attn_L2A(
            query=L, key=A, value=A,
            key_padding_mask=key_pad_mask
        )
        out_L = self.norm_L1(L + self.dropout(attn_L2A))   # Add & Norm

        # ── FFN sublayer per stream ───────────────────────────────
        out_A = self.norm_A2(out_A + self.ff_A(out_A))     # Add & Norm
        out_L = self.norm_L2(out_L + self.ff_L(out_L))     # Add & Norm

        # ── Gated merge → single stream ───────────────────────────
        fused = torch.cat([out_A, out_L], dim=-1)           # (B, T, 2d)
        F = self.fusion_norm(self.dropout(self.fusion_proj(fused)))

        return F

# ============================================================
# SANITY TESTS
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Co-Attention Fusion Sanity Tests  [{device}]")
    print(f"{'='*60}\n")
    
    PASS = "✅ PASS"
    FAIL = "❌ FAIL"
    
    def check(name, cond, detail=""):
        status = PASS if cond else FAIL
        print(f"  {status}  {name}" + (f"  [{detail}]" if detail else ""))
        if not cond:
            sys.exit(1)

    batch_size = 4
    seq_len = 33
    d_model = 256
    
    model = CoAttentionFusion(d_model=d_model, n_heads=4).to(device)
    
    # ── TEST 1: Forward Pass Shape ────────────────────
    print("── TEST 1: Forward Pass Shape ─────────────────────────")
    A_mock = torch.randn(batch_size, seq_len, d_model).to(device)
    L_mock = torch.randn(batch_size, seq_len, d_model).to(device)
    
    with torch.no_grad():
        F_out = model(A_mock, L_mock)
        
    check("Output shape (Batch, T, d_model)", F_out.shape == (batch_size, seq_len, d_model))
    check("No NaN in output", not torch.isnan(F_out).any())

    # ── TEST 2: Mask Handling ─────────────────────────
    print("\n── TEST 2: Mask Handling ──────────────────────────────")
    # Use collate_fn convention: 1=valid, 0=pad (float, NOT bool)
    mask = torch.ones(batch_size, seq_len).to(device)
    mask[:, -3:] = 0   # last 3 frames are padding
    
    try:
        with torch.no_grad():
            F_masked = model(A_mock, L_mock, mask=mask)
        check("Masked forward pass executes", True)
    except Exception as e:
        check("Masked forward pass executes", False, str(e))

    # ── TEST 3: Gradient Flow ─────────────────────────
    print("\n── TEST 3: Gradient Flow ──────────────────────────────")
    model.train()
    model.zero_grad()
    
    A_train = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    L_train = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    
    out = model(A_train, L_train)
    out.mean().backward()
    
    # Check if gradients propagated back to inputs and internal weights
    check("Gradients flow to Audio input", A_train.grad is not None and A_train.grad.abs().sum() > 0)
    check("Gradients flow to Lyrics input", L_train.grad is not None and L_train.grad.abs().sum() > 0)
    check("Gradients flow to Fusion Proj", model.fusion_proj.weight.grad is not None and model.fusion_proj.weight.grad.abs().sum() > 0)

    print(f"\n{'='*60}")
    print("  All tests passed! Ready for MultimodalEmotionModel integration.")
    print(f"{'='*60}\n")