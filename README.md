# CBCT-to-CT Synthesis: 2.5D Variational Diffusion Pipeline

Two-stage deep learning framework for synthesizing CT images from CBCT using VQ-VAE and Variational Diffusion Models (VDM).

---

## Overview

| Stage | Model | Role |
|---|---|---|
| 1 | VQ-VAE | CBCT/CT → compressed latent representation |
| 2 | VDM (UViT / UNet) | CBCT latent → CT latent via denoising diffusion |

**Dataset**: SynthRAD2025 — 955 subjects, 3 anatomy regions (AB / HN / TH)  
**2.5D context**: configurable slice window n = 1, 3, 5, 7, 9  
**Compression ratio**: cpr = 2, 4, 8

---

## Project Structure

```
cbct2ct2_5D/
├── train/                    # 학습 스크립트
│   ├── stage1_vqvae.py       # Stage 1: VQ-VAE 학습
│   └── stage2_vdm.py         # Stage 2: VDM 학습
│
├── eval/                     # 평가·분석 스크립트
│   ├── eval_gen.py           # Step 1: inference → .mha 저장
│   ├── eval_full.py          # Step 2: PSNR/SSIM/MSE/FID 계산
│   ├── eval_vqvae.py         # VQ-VAE 단독 평가
│   ├── hu_analysis.py        # HU 분포 & tissue 오차 분석
│   ├── fn_fp_analysis.py     # tissue 분류 FP/FN rate
│   ├── best_case_analysis.py # n5 vs n1 최적 케이스 시각화
│   └── ttest_analysis.py     # paired t-test (Bonferroni)
│
├── data/                     # 데이터셋
│   ├── synthrad2025.py       # 메인 Dataset 클래스 (train/eval 공통)
│   ├── preprocessed_dataset.py  # 전처리 캐시 기반 Dataset
│   ├── preprocessing.py      # 3D volume → 2D slice 추출
│   ├── preprocess.py         # slice → .npy 저장 (일회성 실행)
│   └── verify_preprocessed.py   # 전처리 결과 검증
│
├── models/
│   └── lvdm/
│       ├── uvit.py           # UViT backbone
│       ├── vdm.py            # Diffusion model wrapper
│       └── utils.py          # LR scheduler 등
│
├── utils/
│   ├── hu.py                 # HU 변환 & tissue threshold (공유)
│   ├── mha.py                # SimpleITK .mha 읽기/쓰기 (공유)
│   ├── inception.py          # InceptionV3 FID helpers (공유)
│   └── wandb.py              # WandB 로깅 헬퍼
│
├── runs/                     # 실행 셸 스크립트
│   ├── run_stage1_ablation.sh
│   ├── run_stage2_ablation.sh
│   ├── run_stage1_64.sh
│   └── run_eval_full.sh
│
├── scripts/                  # 일회성 유틸 스크립트
│   ├── compare_vqvae_params.py
│   └── inspect_latent_dist.py
│
└── tests/
    └── test_eval_full.py
```

---

## Setup

```bash
uv sync
# 또는
pip install -r requirements.txt
```

---

## Training

### Stage 1 — VQ-VAE

```bash
PYTHONPATH=. python train/stage1_vqvae.py \
  --modality ct \
  --in_channels 5 \
  --compress_ratio 4 \
  --exp_name 1_vqvae_ct_n5_cpr4_img128 \
  --data_root /path/to/s2025 \
  --device 0
```

CT / CBCT 각각 학습 필요. `--modality cbct`로 반복 실행.

```bash
# ablation 전체 실행 (GPU 병렬)
bash runs/run_stage1_ablation.sh
```

### Stage 2 — VDM

```bash
PYTHONPATH=. python train/stage2_vdm.py \
  --backbone uvit \
  --in_channels 5 \
  --compress_ratio 4 \
  --ct_ckpt   checkpoints/stage1_vqvae/1_vqvae_ct_n5_cpr4_img128/best.pt \
  --cbct_ckpt checkpoints/stage1_vqvae/1_vqvae_cbct_n5_cpr4_img128/best.pt \
  --exp_name  1_vdm_uvit_n5_cpr4_img128 \
  --data_root /path/to/s2025 \
  --device 0
```

```bash
# ablation 전체 실행 (직렬, GPU 1개)
bash runs/run_stage2_ablation.sh
```

---

## Evaluation

평가는 **2단계**로 분리되어 있습니다. inference는 한 번만 실행하고, 분석은 저장된 `.mha` 파일에서 반복 가능합니다.

### Step 1 — Inference (`.mha` 저장)

```bash
PYTHONPATH=. python eval/eval_gen.py \
  --data_root /path/to/s2025 \
  --output_dir eval_results/gen
```

`eval_results/gen/{model_key}/{subj_id}.mha` 형식으로 저장됩니다.

### Step 2 — Metrics (PSNR / SSIM / MSE / FID)

```bash
PYTHONPATH=. python eval/eval_full.py \
  --gen_dir    eval_results/gen \
  --data_root  /path/to/s2025 \
  --output_dir eval_results
```

또는 한 번에:

```bash
bash runs/run_eval_full.sh
```

### 추가 분석

```bash
# HU 분포 & tissue 오차 (Air/Lung/Fat/Soft tissue/Bone)
PYTHONPATH=. python eval/hu_analysis.py --gen_dir eval_results/gen

# Tissue 분류 FP/FN rate
PYTHONPATH=. python eval/fn_fp_analysis.py --gen_dir eval_results/gen

# n5_cpr4 vs n1_cpr4 최적 케이스 시각화 (부위별 top-3)
PYTHONPATH=. python eval/best_case_analysis.py \
  --metrics_csv eval_results/raw_metrics.csv \
  --gen_dir eval_results/gen

# Paired t-test with Bonferroni correction
PYTHONPATH=. python eval/ttest_analysis.py \
  --metrics_csv eval_results/raw_metrics.csv
```

---

## Ablation Design

**8개 모델** 비교:

| Key | Backbone | n (슬라이스) | cpr (압축률) |
|---|---|---|---|
| uvit_n1_cpr4 | UViT | 1 | 4 |
| uvit_n3_cpr4 | UViT | 3 | 4 |
| **uvit_n5_cpr4** | UViT | **5** | 4 |
| uvit_n7_cpr4 | UViT | 7 | 4 |
| uvit_n9_cpr4 | UViT | 9 | 4 |
| uvit_n5_cpr2 | UViT | 5 | 2 |
| uvit_n5_cpr8 | UViT | 5 | 8 |
| unet_n5_cpr4 | UNet | 5 | 4 |

---

## Evaluation Outputs

| 파일 | 설명 |
|---|---|
| `eval_results/raw_metrics.csv` | 샘플별 PSNR / SSIM / MSE |
| `eval_results/summary_stats.csv` | 모델 × anatomy 집계 + FID |
| `eval_results/boxplot_*.png` | Boxplot 시각화 |
| `eval_results/hu_analysis/` | HU 히스토그램, anatomy/tissue 오차 CSV |
| `eval_results/fp_fn/` | FP/FN rate 표 & 히트맵 |
| `eval_results/best_case/` | 케이스별 4-panel 이미지 (CBCT·gen_n1·gen_n5·GT) |

---

## Key Design Decisions

- **2.5D context**: 슬라이스 n장을 채널로 쌓아 3D 문맥을 2D 모델에 전달
- **Evaluation split**: `stage2_vdm.py`와 동일한 seed=42, val_ratio=0.2 재현
- **FID warmup**: 초기 50,000 step은 FID 계산 생략 (학습 안정성)
- **scale_factor**: 훈련 첫 배치 기준으로 latent 정규화 (inference에서 동일하게 재현)

---

## Requirements

- Python ≥ 3.10
- PyTorch 2.6+, MONAI 1.5+, SimpleITK 2.5+
- WandB (학습 로깅)

---

**Last Updated**: 2026-04-30
