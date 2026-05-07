# MambaMER — Mamba-based Multimodal Emotion Recognition for Music

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Dynamic music emotion recognition** using a novel Bidirectional Mamba architecture with audio–lyrics co-attention fusion, predicting frame-level **Valence** and **Arousal** curves over time.

---

## ✨ Key Features

- **Bidirectional Mamba SSM** — Linear-time sequence modeling with emotion-conditioned state initialization, replacing costly Transformer self-attention.
- **Multi-Scale Temporal Refinement** — Captures both local note-level and global section-level emotion dynamics at multiple temporal resolutions (1×, 2×, 4×).
- **Co-Attention Fusion** — ViLBERT-style cross-modal attention fuses audio acoustics with lyrical semantics bidirectionally.
- **DeBERTa-v3 Lyrics Encoder** — Contextual token embeddings from DeBERTa-v3-base, augmented with NRC-VAD lexicon scores, with attention-weighted pooling and temporal interpolation.
- **Flask Web Interface** — Upload audio/video files or select PMEmo songs and visualize predicted emotion curves in real time.

---

## 🏗️ Architecture

```
Audio (openSMILE LLDs)          Lyrics (.lrc)
       │                              │
 ┌─────▼─────┐               ┌────────▼────────┐
 │  Audio     │               │  DeBERTa-v3     │
 │  Encoder   │               │  + NRC-VAD      │
 │  (MLP+Res) │               │  + AttnPool     │
 └─────┬──────┘               └────────┬────────┘
       │ (B,T,256)                     │ (B,T,256)
       └──────────┬────────────────────┘
                  ▼
        ┌─────────────────┐
        │  Co-Attention    │
        │  Fusion (ViLBERT)│
        └────────┬────────┘
                 ▼
        ┌─────────────────┐
        │  Positional Enc  │
        │  + Relative Aug  │
        └────────┬────────┘
                 ▼
        ┌─────────────────┐
        │  Multi-Scale     │
        │  Temporal Mamba  │  × 2 stacked refiners
        │  (Bidirectional) │
        └────────┬────────┘
                 ▼
        ┌─────────────────┐
        │  Multi-Task      │
        │  Emotion Head    │  Separate GRU branches
        │  (Valence+Arousal│  + learned smoother
        └────────┬────────┘
                 ▼
          Valence, Arousal
            (B, T, 2)
```

---

## 📁 Project Structure

```
MambaMER/
├── models.py                 # Full model: AudioEncoder, Mamba, Fusion, Head
├── co_attention_fusion.py    # ViLBERT-style co-attentional transformer
├── lyrics.py                 # DeBERTa lyrics encoder + NRC-VAD + LRC parser
├── dataset.py                # PMEmo dataset with disk caching
├── multimodal_datase2.py     # Alternative dataset loader
├── train1.py                 # Multi-seed training with composite loss
├── lyrics_pretrain.py        # DeBERTa lyrics pre-training script
├── infer.py                  # CLI inference pipeline
├── compute_global_stats.py   # Corpus-level VA statistics for de-normalization
├── app.py                    # Flask web interface
├── audio.py                  # Audio utilities
├── templates/
│   └── index.html            # Web UI template
├── requirements.txt          # Python dependencies
├── global_stats.npz          # Precomputed corpus VA mean/std
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- [FFmpeg](https://ffmpeg.org/) (for video file support)

### Installation

```bash
# Clone the repository
git clone https://github.com/YashKale02/MambaMER.git
cd MambaMER

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install mamba-ssm
pip install transformers flask opensmile numpy pandas matplotlib

# Install NRC-VAD Lexicon
# Place NRC-VAD-Lexicon-v2.1.txt in the project root
```

### Dataset

This project uses the [PMEmo2019](https://github.com/HuiZhangDB/PMEmo) dataset.
Download it separately and place it under `PMEmo2019/` in the project root:

```
PMEmo2019/
├── features/
│   └── dynamic_features.csv     # openSMILE LLD features
├── annotations/
│   └── dynamic_annotations.csv  # Valence/Arousal annotations
└── lyrics/
    └── *.lrc                    # Timestamped lyrics
```

### Model Weights

Pre-trained model checkpoints (`.pt` files) are not included in this repository due to size.  
Download them from the releases page or contact the author.

| File | Description |
|------|-------------|
| `best_model.pt` | Best overall checkpoint |
| `best_model_seed_42.pt` | Seed 42 checkpoint |
| `best_model_seed_123.pt` | Seed 123 checkpoint |
| `best_model_seed_2024.pt` | Seed 2024 checkpoint |
| `best_encoder.pt` | Pre-trained DeBERTa lyrics encoder |

---

## 🎯 Usage

### Training

```bash
# Train with multi-seed evaluation (seeds: 42, 123, 2024)
python train1.py
```

Training uses a composite loss function combining:
- MSE loss (warm-up phase)
- Per-song CCC loss (Valence & Arousal)
- Sliding-window local CCC
- Pearson correlation loss
- Mean bias & variance floor regularization
- Smoothness & direction penalties

### Inference (CLI)

```bash
# Predict emotion for a PMEmo song by ID
python infer.py --mode csv --music-id 1 --compare-gt

# Process all PMEmo songs
python infer.py --mode csv --all

# Predict from a raw audio/video file
python infer.py --mode audio --audio-path song.wav --lrc-path song.lrc

# Custom checkpoint
python infer.py --mode csv --music-id 1 --model best_model_seed_42.pt
```

### Web Interface

```bash
python app.py
# Open http://localhost:5000
```

The web interface supports:
- **PMEmo Mode** — Select a song ID from the dataset and visualize predictions
- **Upload Mode** — Upload any audio/video file (WAV, MP3, MP4, etc.) with optional `.lrc` lyrics
- **Interactive Charts** — View predicted Valence & Arousal curves with optional ground-truth overlay
- **CSV Export** — Download prediction results

---

## 🔧 Pre-processing

### Compute Global Statistics

Before running inference, compute corpus-level VA statistics (one-time step):

```bash
python compute_global_stats.py
```

This generates `global_stats.npz` containing the global mean and standard deviation of Valence/Arousal across all songs, used to de-normalize z-scored model predictions back to the [0, 1] scale.

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **CCC** (Concordance Correlation Coefficient) | Primary metric — measures agreement between predicted and ground-truth VA curves |
| **PCC** (Pearson Correlation Coefficient) | Linear correlation between predictions and targets |
| **RMSE** (Root Mean Squared Error) | Absolute prediction error |

---

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@misc{kale2026mambamer,
  title={MambaMER: Mamba-based Multimodal Emotion Recognition for Music},
  author={Kale, Yash},
  year={2026},
  url={https://github.com/YashKale02/MambaMER}
}
```

---

## 📝 License

This project is released under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- [PMEmo2019 Dataset](https://github.com/HuiZhangDB/PMEmo) — Zhang et al.
- [Mamba SSM](https://github.com/state-spaces/mamba) — Gu & Dao
- [DeBERTa-v3](https://huggingface.co/microsoft/deberta-v3-base) — Microsoft
- [NRC-VAD Lexicon](https://saifmohammad.com/WebPages/nrc-vad.html) — Mohammad
