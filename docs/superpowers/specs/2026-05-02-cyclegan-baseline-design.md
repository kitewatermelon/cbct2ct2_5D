# CycleGAN Baseline Design

**Date:** 2026-05-02  
**Goal:** Ablation study용 CycleGAN baseline — 기존 VDM 모델들과 동일한 eval 파이프라인(SSIM, PSNR, MSE, FID)으로 비교

---

## 1. 파일 구조

```
train/
  cyclegan.py              # CycleGAN 학습 스크립트
models/
  cyclegan/
    __init__.py
    generator.py           # ResNet Generator (MONAI ResidualUnit 활용)
    discriminator.py       # MONAI PatchDiscriminator 래퍼
eval/
  eval_full.py             # cyclegan 항목 추가 (기존 파일 수정)
runs/
  run_cyclegan.sh          # 학습 실행 스크립트
```

VQ-VAE, VDM에 의존하지 않는 독립 스크립트. 기존 `train/stage1_vqvae.py`, `train/stage2_vdm.py`와 같은 코드 스타일 유지.

---

## 2. 아키텍처

### Generator (`models/cyclegan/generator.py`)
- 입력: `(B, 1, H, W)` — 단일 채널 2D 슬라이스
- 구조: Conv↓ (stride=2, ×2) → `ResidualUnit`×9 → `ConvTranspose2d`↑ (×2) → tanh
- MONAI `ResidualUnit` 활용하여 구현 최소화
- 2개 인스턴스: `G_CBCT2CT`, `G_CT2CBCT`

### Discriminator (`models/cyclegan/discriminator.py`)
- MONAI `PatchDiscriminator` 직접 사용 (70×70 PatchGAN)
- 입력: `(B, 1, H, W)`
- 2개 인스턴스: `D_CT`, `D_CBCT`

---

## 3. 학습 (`train/cyclegan.py`)

### 데이터
- `SynthRad2025` 데이터셋에서 슬라이스 단위로 샘플링 (2D flatten)
- anatomy: AB, HN, TH (기존과 동일)
- spatial_size: 128×128 (기존과 동일)
- val_ratio: 0.2

### Loss
| 항목 | 수식 | 가중치 |
|------|------|--------|
| Adversarial | MSE(D(G(x)), 1) | 1.0 |
| Cycle consistency | L1(G_BA(G_AB(x)), x) | λ=10 |
| Identity | L1(G_AB(y), y) | λ=5 |

### 옵티마이저
- Adam, lr=2e-4, betas=(0.5, 0.999)
- LR decay: epoch 100 이후 선형 감소 (원논문 방식)
- Generator, Discriminator 별도 optimizer

### 로깅
- WandB: train loss, val SSIM/PSNR/MSE, 이미지 grid (기존 패턴 동일)
- 체크포인트: `checkpoints/cyclegan_best.pt`

---

## 4. Eval 통합 (`eval/eval_full.py`)

`MODEL_CONFIGS`에 항목 추가:
```python
{"key": "cyclegan", "type": "cyclegan", "ckpt": "<path>"}
```

`--gen_dir` 방식(미리 생성된 `.mha` 파일에서 메트릭 계산)으로 통합. 모델 로딩 없이 평가 가능.

평가 메트릭: SSIM, PSNR, MSE, FID (기존 eval_full.py와 동일한 메트릭 세트)

---

## 5. 실행 스크립트 (`runs/run_cyclegan.sh`)

```bash
python train/cyclegan.py \
  --data_root /home/dministrator/s2025 \
  --anatomy AB HN TH \
  --spatial_size 128 \
  --n_epochs 200 \
  --device 0
```

---

## 6. 제약사항 및 결정사항

- **2D only**: 기존 VDM의 2.5D(N-slice)와 달리 단일 슬라이스 처리. Fair comparison 관점에서 한계가 있으나 CycleGAN 원논문 방식에 충실.
- **MONAI 의존**: `PatchDiscriminator`는 이미 venv에 설치된 MONAI 활용 (추가 의존성 없음).
- **독립 스크립트**: VQ-VAE 체크포인트 불필요. 픽셀 공간에서 직접 학습.
