# CBCT-to-CT Synthesis: 2.5D Variational Diffusion Pipeline

**Two-stage deep learning framework for medical image synthesis** using Vector Quantized Variational Autoencoder (VQ-VAE) and Variational Diffusion Models (VDM).

## 🎯 What This Does

This project implements an end-to-end pipeline to synthesize CT images from CBCT (Cone Beam CT) images:

1. **Stage 1 (VQ-VAE)**: Learns a compressed latent representation of CT images
   - Input: Multi-slice CBCT (configurable 1-9 slices)
   - Output: Compact latent features with reconstruction loss + adversarial loss + perceptual loss
   - Compression ratio: 2x, 4x, or 8x

2. **Stage 2 (VDM)**: Generates high-quality CT images from CBCT latents via diffusion
   - Conditions on CBCT latent representation
   - Generates CT latent codes using denoising diffusion process
   - Supports multiple backbones: UViT or MONAI DiffusionModelUNet
   - Evaluates with SSIM, PSNR, MSE, and FID metrics

## 📦 Dataset

- **SynthRAD2025**: 955 subjects across 3 anatomical regions
- Data root: `/data/prof2/mai/s2025/dataset`
- Train/Val split: 80/20

## 🏗️ Project Structure

```
cbct2ct2_5D/
├── stage1_vqvae.py          # VQ-VAE training script
├── stage2_vdm.py            # VDM training script  
├── stage1_full.py           # Full VQ-VAE experiment
├── stage2_full.py           # Full VDM experiment
├── eval.py                  # Evaluation utilities
├── run_stage1_ablation.sh   # Ablation study for Stage 1
├── run_stage2_ablation.sh   # Ablation study for Stage 2
├── configs/                 # YAML configuration files
├── data/                    # Dataset loaders (SynthRad2025)
├── models/                  # Model definitions (VDM, UViT, etc.)
├── optim/                   # Optimization utilities
└── utils/                   # Logging and helpers
```

## 🚀 Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
# or
uv sync
```

### Stage 1: Train VQ-VAE
```bash
python stage1_vqvae.py \
  --exp_name vqvae_ct \
  --device 0 \
  --in_channels 1 \
  --compress_ratio 4 \
  --batch_size 128 \
  --num_epochs 100
```

### Stage 2: Train VDM
```bash
python stage2_vdm.py \
  --exp_name vdm_ct_from_cbct \
  --device 0 \
  --stage1_checkpoint /path/to/vqvae.pt \
  --backbone uvit \
  --batch_size 64 \
  --num_steps 1_000_000
```

## 📊 Training Configuration

### Stage 1 (VQ-VAE)
- **Architecture**: Encoder → Codebook → Decoder
- **Losses**: 
  - Reconstruction (L1)
  - VQ codebook loss
  - Adversarial loss (patch discriminator)
  - Perceptual loss (LPIPS)
- **Metrics**: Reconstruction loss, Perplexity, validation samples
- **Logging**: WandB integration

### Stage 2 (VDM)
- **Architecture**: UViT or DiffusionModelUNet
- **Conditioning**: CBCT latent codes
- **Sampling**: Configurable diffusion steps (default 20 during evaluation)
- **Metrics**: SSIM, PSNR, MSE, FID
- **Optimization**: Warm-up, cosine annealing, EMA scheduler
- **Logging**: Training curves, qualitative samples, FID plots

## 🔍 Ablation Studies

The project includes comprehensive ablation scripts:

### Stage 1 Ablations
- **in_channels**: 1, 3, 5, 7, 9 (2.5D context depth)
- **compress_ratio**: 2, 4, 8 (latent compression)
- **architectures**: Standard VQVAE, with/without discriminator

### Stage 2 Ablations
- **backbones**: UViT vs DiffusionModelUNet
- **condition types**: Full latent, spatially-pooled latent
- **sampling strategies**: Different step counts and schedules

Run ablations with:
```bash
bash run_stage1_ablation.sh
bash run_stage2_ablation.sh
```

## 📈 Monitoring

All experiments logged to WandB:
- Real-time training curves (losses, metrics)
- Validation image samples (input, reconstruction, target)
- FID score tracking
- Hyperparameter sweeps

## 🛠️ Development Environment

- **GPU**: RTX A6000 (CUDA 12.6)
- **Server**: gpu.pknu.ac.kr (PKNU GPU Cluster)
- **Python**: ≥3.10
- **Key Dependencies**: PyTorch 2.6+, MONAI 1.5+, WandB

## 📝 Key Files

| File | Purpose |
|------|---------|
| `stage1_vqvae.py` | VQ-VAE single experiment training |
| `stage2_vdm.py` | VDM single experiment training |
| `stage1_full.py` | Full Stage 1 pipeline with checkpointing |
| `stage2_full.py` | Full Stage 2 pipeline with FID evaluation |
| `eval.py` | Standalone evaluation utilities |
| `data/synthrad2025.py` | Dataset loading and preprocessing |

## 🎓 Citation

If you use this code in your research:

```bibtex
@article{cbct2ct_2025,
  title={2.5D Context Encoding with Latent-Space Variational Diffusion for CBCT-to-CT Synthesis},
  author={Park, YeonSu},
  journal={MDPI},
  year={2025}
}
```

## 📋 Notes

- FID metric disabled during warmup phase (first 50K steps) to avoid NaN issues
- Uses 2.5D context encoding to balance computational cost and receptive field
- Both stages can be trained independently after Stage 1 checkpoint is saved
- Supports mixed precision training (AMP) for memory efficiency

---

**Last Updated**: April 2026  
**Status**: Active Development
