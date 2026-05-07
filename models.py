# models.py — Audio+Lyrics ablation (no mel)
# Uses: AudioEncoder, LyricsProjection, CoAttentionFusion
# Removed: MelAdapter, mel arousal bypass
import torch
import torch.nn as nn
import math
from co_attention_fusion import CoAttentionFusion

# ============================================================
# Audio Encoder (2-layer MLP + residual)
# ============================================================

class AudioEncoder(nn.Module):
    def __init__(self, input_dim=260, d_model=256, dropout=0.1):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, d_model)
        self.layer2 = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual_proj = nn.Linear(input_dim, d_model)

    def forward(self, x):
        """x: (B, T, 260) → (B, T, 256)"""
        residual = self.residual_proj(x)
        x = self.activation(self.norm1(self.layer1(x)))
        x = self.dropout(x)
        x = self.norm2(self.layer2(x))
        x = self.dropout(x)
        return x + residual

# ============================================================
# Lyrics Projection
# ============================================================
class LyricsProjection(nn.Module):
    def __init__(self, input_dim=771, d_model=256):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.act  = nn.GELU()
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """x: (B, T, 771) → (B, T, 256)"""
        return self.norm(self.act(self.proj(x)))


# ============================================================
# ① Sinusoidal Positional Encoding
# ============================================================

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 256, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        pe = self._build_pe(max_len, d_model)
        self.register_buffer("pe", pe.unsqueeze(0))

    @staticmethod
    def _build_pe(length: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        # Dynamically extend the PE buffer if the input is longer than max_len.
        # Sinusoidal PE is deterministic, so extending at runtime is safe.
        if T > self.pe.size(1):
            self.pe = self._build_pe(T, self.d_model).unsqueeze(0).to(x.device)
        x = x + self.pe[:, :T, :]
        return self.dropout(x)


# ============================================================
# ② Relative Position Augmenter
# ============================================================

class RelativePositionAugmenter(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.proj = nn.Linear(d_model + 1, d_model)

    def forward(self, x):
        """x: (B, T, d_model) → (B, T, d_model)"""
        B, T, D = x.shape
        position_ratio = torch.linspace(0, 1, T, device=x.device)
        position_ratio = position_ratio.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1)
        x_augmented = torch.cat([x, position_ratio], dim=-1)
        return self.proj(x_augmented)


# ============================================================
# ③ Bidirectional Mamba  — with emotion-conditioned state init
# ============================================================

class BidirectionalMamba(nn.Module):
    def __init__(self, d_model: int = 256, d_state: int = 16,
                 d_conv: int = 4, expand: int = 4,
                 dropout: float = 0.15):
        super().__init__()
        from mamba_ssm import Mamba

        self.d_model = d_model

        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state,
                               d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state,
                               d_conv=d_conv, expand=expand)

        for mamba in [self.mamba_fwd, self.mamba_bwd]:
            with torch.no_grad():
                mamba.A_log.clamp_(-4.0, -0.1)

        d_inner = int(expand * d_model)
        self.ctx_proj_fwd = nn.Sequential(
            nn.Linear(d_model, d_inner, bias=False),
            nn.Tanh(),
        )
        self.ctx_proj_bwd = nn.Sequential(
            nn.Linear(d_model, d_inner, bias=False),
            nn.Tanh(),
        )
        for proj in [self.ctx_proj_fwd, self.ctx_proj_bwd]:
            nn.init.normal_(proj[0].weight, std=0.01)

        self.merge = nn.Linear(d_model * 2, d_model, bias=False)
        self.norm  = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(p=dropout)

    def _inject_context(self, x, ctx_proj):
        ctx  = x.mean(dim=1)
        bias = ctx_proj(ctx)
        bias = bias[:, :self.d_model]
        bias = bias.unsqueeze(1)
        return x + bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fwd = self._inject_context(x,         self.ctx_proj_fwd)
        x_bwd = self._inject_context(x.flip(1), self.ctx_proj_bwd)

        h_fwd = self.drop(self.mamba_fwd(x_fwd))
        h_bwd = self.drop(self.mamba_bwd(x_bwd)).flip(dims=[1])

        return self.norm(self.merge(torch.cat([h_fwd, h_bwd], dim=-1)) + x)


# ============================================================
# ④ Multi-Scale Temporal Mamba
# ============================================================

class MultiScaleTemporalMamba(nn.Module):
    def __init__(
        self,
        d_model:  int  = 256,
        scales:   list = [1, 2, 4],
        d_states: list = [16, 8, 4],
        d_conv:   int  = 4,
        expand:   int  = 4,
    ):
        super().__init__()
        assert len(scales) == len(d_states)

        self.scales  = scales
        self.d_model = d_model

        self.mambas = nn.ModuleList([
            BidirectionalMamba(d_model=d_model, d_state=ds,
                               d_conv=d_conv, expand=expand)
            for ds in d_states
        ])

        self.merge = nn.Linear(d_model * len(scales), d_model, bias=False)
        self.norm  = nn.LayerNorm(d_model)

    def _downsample(self, x, factor):
        if factor == 1:
            return x
        T = x.shape[1]
        if T < factor:
            return x
        x_t = x.transpose(1, 2)
        x_t = torch.nn.functional.avg_pool1d(x_t, kernel_size=factor, stride=factor)
        return x_t.transpose(1, 2)

    def _upsample(self, x, target_T):
        x_t = x.transpose(1, 2)
        x_t = torch.nn.functional.interpolate(x_t, size=target_T, mode="nearest")
        return x_t.transpose(1, 2)

    def forward(self, x):
        T = x.size(1)
        scale_outputs = []
        for factor, mamba in zip(self.scales, self.mambas):
            x_down    = self._downsample(x, factor)
            x_refined = mamba(x_down)
            del x_down
            x_up      = self._upsample(x_refined, T)
            del x_refined
            scale_outputs.append(x_up)
        merged = torch.cat(scale_outputs, dim=-1)
        del scale_outputs
        out    = self.merge(merged)
        del merged
        return self.norm(out + x)


# ============================================================
# ⑩ Multi-Task Emotion Head — no mel bypass
# ============================================================

class MultiTaskEmotionHead(nn.Module):
    """Audio+Lyrics head — no mel energy bypass."""
    def __init__(self, d_model=256, d_gru=64, d_hidden=64):
        super().__init__()

        self.gru_v = nn.GRU(
            input_size=d_model, hidden_size=d_gru, 
            num_layers=1, batch_first=True, bidirectional=False
        )
        
        self.gru_a = nn.GRU(
            input_size=d_model, hidden_size=d_gru, 
            num_layers=1, batch_first=True, bidirectional=False
        )
        
        self.gru_norm_v = nn.LayerNorm(d_gru)
        self.gru_norm_a = nn.LayerNorm(d_gru)

        self.v_branch = nn.Sequential(
            nn.Linear(d_gru, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
        )
        self.a_branch = nn.Sequential(
            nn.Linear(d_gru, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
        )

        self.smoother = nn.Conv1d(2, 2, kernel_size=5, padding=2, groups=2)
        with torch.no_grad():
            self.smoother.weight.zero_()
            self.smoother.weight[:, :, 2] = 1.0
            if self.smoother.bias is not None:
                self.smoother.bias.zero_()

    def forward(self, x):
        """
        x: (B, T, 256) - Fused audio+lyrics features
        No mel_energy bypass in this ablation.
        """
        gru_out_v, _ = self.gru_v(x)
        gru_out_a, _ = self.gru_a(x)
        
        gru_out_v = self.gru_norm_v(gru_out_v)
        gru_out_a = self.gru_norm_a(gru_out_a)

        V = self.v_branch(gru_out_v)            
        A = self.a_branch(gru_out_a)            

        va_raw = torch.cat([V, A], dim=-1)      
        va_pred = self.smoother(va_raw.transpose(1, 2)).transpose(1, 2)

        return {"va_pred": va_pred}

# ============================================================
# Top-level Model — Audio + Lyrics (no mel)
# ============================================================
class MultimodalEmotionModel(nn.Module):
    """Audio+Lyrics ablation — no mel modality."""
    def __init__(self):
        super().__init__()
        self.audio_encoder = AudioEncoder()
        self.lyrics_proj   = LyricsProjection()    
        self.fusion        = CoAttentionFusion(d_model=256, n_heads=4, dropout=0.1)
        self.pos_enc       = SinusoidalPositionalEncoding()
        self.rel_pos       = RelativePositionAugmenter()
        self.refiner1 = MultiScaleTemporalMamba()
        self.refiner2 = MultiScaleTemporalMamba()
        self.head = MultiTaskEmotionHead()

    def forward(self, audio, lyrics, mask=None): 
        # 1. Audio features
        A = self.audio_encoder(audio)              # (B, T, 256)
        
        # 2. Lyrical semantics
        L = self.lyrics_proj(lyrics)               # (B, T, 256)
        
        # 3. Co-Attention Fusion (audio ↔ lyrics)
        F = self.fusion(A, L, mask=mask)
        F = self.pos_enc(F)
        F = self.rel_pos(F)
        R = self.refiner1(F)
        R = self.refiner2(R)
        
        # 4. Prediction — no mel bypass
        return self.head(R)

# ============================================================
# SANITY TESTS
# ============================================================

if __name__ == "__main__":
    import sys

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Audio+Lyrics (no mel) Ablation — Sanity tests  [{device}]")
    print(f"{'='*60}\n")

    PASS = "✅ PASS"
    FAIL = "❌ FAIL"

    def check(name, cond, detail=""):
        status = PASS if cond else FAIL
        print(f"  {status}  {name}" + (f"  [{detail}]" if detail else ""))
        if not cond:
            sys.exit(1)

    B, D = 2, 256

    # ── TEST 1: Full model (audio + lyrics, no mel) ────────────
    print("── TEST 1: Audio+Lyrics model (T=33) ─────────────────")
    model  = MultimodalEmotionModel().to(device)
    audio  = torch.randn(4, 33, 260).to(device)
    lyrics = torch.randn(4, 33, 771).to(device)
    mask   = torch.ones(4, 33).to(device)
    with torch.no_grad():
        result = model(audio, lyrics, mask=mask)
    check("va_pred shape (4,33,2)", result["va_pred"].shape == (4, 33, 2))
    check("no NaN in output", not torch.isnan(result["va_pred"]).any())

    # ── TEST 2: Without mask ───────────────────────────────────
    print("\n── TEST 2: Without mask ──────────────────────────────")
    with torch.no_grad():
        result2 = model(audio, lyrics)
    check("va_pred shape (4,33,2)", result2["va_pred"].shape == (4, 33, 2))
    check("no NaN in output", not torch.isnan(result2["va_pred"]).any())

    # ── TEST 3: GRU head ───────────────────────────────────────
    print("\n── TEST 3: GRU head V & A branches ───────────────────")
    head = model.head
    x = torch.randn(B, 33, D).to(device)
    with torch.no_grad():
        out = head(x)
    check("Output shape (B,33,2)", out["va_pred"].shape == (B, 33, 2))

    print(f"\n{'='*60}")
    print("  All tests passed!")
    print(f"{'='*60}\n")