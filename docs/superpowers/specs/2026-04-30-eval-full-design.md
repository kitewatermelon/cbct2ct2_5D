# Eval Full Framework Design
Date: 2026-04-30

## Goal

`eval_full.py` — 8개 Stage2 VDM 모델을 동일한 val split에서 평가하여 모델별·anatomy별 per-sample 메트릭 CSV와 boxplot을 생성한다.

---

## 평가 대상 모델 (8개)

| 모델 키 | backbone | n | cpr |
|---|---|---|---|
| uvit_n1_cpr4 | uvit | 1 | 4 |
| uvit_n3_cpr4 | uvit | 3 | 4 |
| uvit_n5_cpr4 | uvit | 5 | 4 |
| uvit_n7_cpr4 | uvit | 7 | 4 |
| uvit_n9_cpr4 | uvit | 9 | 4 |
| uvit_n5_cpr2 | uvit | 5 | 2 |
| uvit_n5_cpr8 | uvit | 5 | 8 |
| unet_n5_cpr4 | unet | 5 | 4 |

체크포인트: `checkpoints/stage2_vdm/{exp_name}/latest.pt`  
VQ-VAE 체크포인트: `checkpoints/stage1_vqvae/{vqvae_exp_name}/best.pt`

---

## 데이터 Split

stage2_vdm.py의 split 로직을 완전히 복제해 데이터 누수를 방지한다.

```python
full_ds = SynthRad2025(
    root=f"{data_root}/dataset/train/n{n}",
    modality=["cbct", "ct"],
    anatomy=["AB", "HN", "TH"],
    transform=build_transforms(["cbct", "ct"], spatial_size=(128, 128), augment=False),
)
n_val   = int(len(full_ds) * 0.2)
n_train = len(full_ds) - n_val
train_ds, val_ds = random_split(
    full_ds, [n_train, n_val],
    generator=torch.Generator().manual_seed(42),
)
```

- 각 n 값마다 독립적으로 split (`dataset/train/n{n}` 경로 사용)
- 슬라이스 = 피험자 (1 subj_id = 1 슬라이스 윈도우)

---

## Scale Factor 복원

체크포인트에 저장되지 않으므로 훈련과 동일한 방식으로 재계산:

```python
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                          generator=torch.Generator().manual_seed(42))
first_batch = next(iter(train_loader))
scale_factor = 1.0 / ct_ae.encode_stage_2_inputs(first_batch["ct"].to(device)).flatten().std().item()
```

`shuffle=True` + seed 42 고정으로 훈련 시 첫 배치와 동일하게 재현.

---

## Inference Pipeline

각 val 샘플에 대해:
1. `cbct_img` → `cbct_ae.encode_stage_2_inputs` × scale_factor → `cond`
2. VDM sampling (`n_sample_steps=200`) → `sampled_z`
3. `ct_ae.decode_stage_2_outputs(sampled_z / scale_factor)` → `ct_gen`
4. GT: `ct_img[:, mid:mid+1]` (중앙 슬라이스)
5. 메트릭 계산: PSNR, SSIM, MSE

---

## Anatomy 태깅

`subj_id`에서 경로 파싱으로 anatomy 추출:
```python
anatomy = subj_id.split("/")[0]  # "AB", "HN", "TH"
```

---

## 출력

### raw_metrics.csv
```
model, anatomy, subj_id, psnr, ssim, mse
uvit_n1_cpr4, AB, AB/1PA001_s042, 32.1, 0.901, 0.00062
...
```
- 8 모델 × val 샘플 수 행

### summary_stats.csv
```
model, anatomy, psnr_mean, psnr_std, ssim_mean, ssim_std, mse_mean, mse_std
...
```

### 시각화 (PNG)
- `boxplot_by_model_{metric}.png` × 3 (PSNR/SSIM/MSE): x축=모델, anatomy별 색상
- `boxplot_by_anatomy_{metric}.png` × 3: x축=anatomy, 모델별 서브플롯

---

## CLI 인터페이스

```bash
python eval_full.py \
    --data_root /home/dministrator/s2025 \
    --ckpt_base checkpoints/stage2_vdm \
    --vqvae_base checkpoints/stage1_vqvae \
    --output_dir eval_results \
    --device 0 \
    --n_sample_steps 200 \
    --batch_size 8
```

---

## 구현 파일

| 파일 | 역할 |
|---|---|
| `eval_full.py` | 메인 스크립트 (단일 파일) |
| `eval_results/raw_metrics.csv` | per-sample raw 메트릭 |
| `eval_results/summary_stats.csv` | 모델×anatomy 요약 통계 |
| `eval_results/boxplot_*.png` | 시각화 |
