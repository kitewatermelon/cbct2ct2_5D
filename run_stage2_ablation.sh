#!/bin/bash
# run_stage2_ablation.sh — Stage2 VDM ablation (GPU 4,5,6)
# Ablation 1: backbone(uvit) × in_channels(1,3,5,7,9) @ cpr4_img128
# Ablation 2: backbone(uvit) × compress_ratio(2,8)     @ n5

set -e
mkdir -p logs

DATA=/data/prof2/mai/s2025
CKPT=checkpoints/stage1_vqvae

# ---------------------------------------------------------------------------
# 공통 하이퍼파라미터 (stage2_vdm.py 인자에 맞춤)
# ---------------------------------------------------------------------------
NUM_STEPS=100000          # 총 학습 스텝
EVAL_EVERY=5000           # eval 주기 (PSNR/SSIM/FID, --eval_every)
LOG_EVERY=100             # train 로그 주기
WARMUP_STEPS=5000         # LR warmup 스텝
SAMPLE_STEPS_EVAL=20      # eval용 샘플링 스텝 (--sample_steps_eval)
FID_WARMUP_STEPS=50000    # FID 계산 시작 스텝 (--fid_warmup_steps)
QUAL_SAVE_EVERY=5         # eval 몇 회마다 qualitative grid 저장 (--qual_save_every)
NUM_EVAL_SAMPLES=4        # eval 샘플링 수 (--num_eval_samples)

COMMON="--num_steps ${NUM_STEPS} \
        --eval_every ${EVAL_EVERY} \
        --log_every ${LOG_EVERY} \
        --warmup_steps ${WARMUP_STEPS} \
        --sample_steps_eval ${SAMPLE_STEPS_EVAL} \
        --fid_warmup_steps ${FID_WARMUP_STEPS} \
        --qual_save_every ${QUAL_SAVE_EVERY} \
        --num_eval_samples ${NUM_EVAL_SAMPLES} \
        --data_root ${DATA}" \
        --embedding_dim 1 \

# ===========================================================================
# Ablation 1-A: n1, n3, n5 @ cpr4  (GPU 4,5,6 동시)
# ===========================================================================
echo "=== Ablation 1-A: n1, n3, n5 @ cpr4 ==="

PYTHONPATH=. python stage2_vdm.py --batch_size 8 \
  --backbone uvit --in_channels 1 --compress_ratio 4 \
  --ct_ckpt   ${CKPT}/vqvae_ct_n1_cpr4_img128/best.pt \
  --cbct_ckpt ${CKPT}/vqvae_cbct_n1_cpr4_img128/best.pt \
  --exp_name 1_vdm_uvit_n1_cpr4_img128 --device 4 \
  ${COMMON} > logs/stage2_uvit_n1_cpr4.log 2>&1 &

PYTHONPATH=. python stage2_vdm.py --batch_size 8 \
  --backbone uvit --in_channels 3 --compress_ratio 4 \
  --ct_ckpt   ${CKPT}/vqvae_ct_n3_cpr4_img128/best.pt \
  --cbct_ckpt ${CKPT}/vqvae_cbct_n3_cpr4_img128/best.pt \
  --exp_name 1_vdm_uvit_n3_cpr4_img128 --device 5 \
  ${COMMON} > logs/stage2_uvit_n3_cpr4.log 2>&1 &

PYTHONPATH=. python stage2_vdm.py --batch_size 8 \
  --backbone uvit --in_channels 5 --compress_ratio 4 \
  --ct_ckpt   ${CKPT}/vqvae_ct_n5_cpr4_img128/best.pt \
  --cbct_ckpt ${CKPT}/vqvae_cbct_n5_cpr4_img128/best.pt \
  --exp_name 1_vdm_uvit_n5_cpr4_img128 --device 6 \
  ${COMMON} > logs/stage2_uvit_n5_cpr4.log 2>&1 &

wait

# ===========================================================================
# Ablation 1-B: n7, n9 @ cpr4  (GPU 4,5,6 동시)
# ===========================================================================
echo "=== Ablation 1-B: n7, n9 @ cpr4 ==="

PYTHONPATH=. python stage2_vdm.py --batch_size 8 \
  --backbone uvit --in_channels 7 --compress_ratio 4 \
  --ct_ckpt   ${CKPT}/vqvae_ct_n7_cpr4_img128/best.pt \
  --cbct_ckpt ${CKPT}/vqvae_cbct_n7_cpr4_img128/best.pt \
  --exp_name 1_vdm_uvit_n7_cpr4_img128 --device 4 \
  ${COMMON} > logs/stage2_uvit_n7_cpr4.log 2>&1 &

PYTHONPATH=. python stage2_vdm.py --batch_size 8 \
  --backbone uvit --in_channels 9 --compress_ratio 4 \
  --ct_ckpt   ${CKPT}/vqvae_ct_n9_cpr4_img128/best.pt \
  --cbct_ckpt ${CKPT}/vqvae_cbct_n9_cpr4_img128/best.pt \
  --exp_name 1_vdm_uvit_n9_cpr4_img128 --device 5 \
  ${COMMON} > logs/stage2_uvit_n9_cpr4.log 2>&1 &

wait

echo "=== Ablation 1 done ==="

# ===========================================================================
# Ablation 2: compress_ratio(2,8) @ n5  (GPU 4,5,6 동시)
# ===========================================================================
echo "=== Ablation 2: cpr2, cpr8 @ n5 ==="

PYTHONPATH=. python stage2_vdm.py --batch_size 8 \
  --backbone uvit --in_channels 5 --compress_ratio 8 \
  --ct_ckpt   ${CKPT}/vqvae_ct_n5_cpr8_img128/best.pt \
  --cbct_ckpt ${CKPT}/vqvae_cbct_n5_cpr8_img128/best.pt \
  --exp_name 1_vdm_uvit_n5_cpr8_img128 --device 4 \
  ${COMMON} > logs/stage2_uvit_n5_cpr8.log 2>&1 &

PYTHONPATH=. python stage2_vdm.py --batch_size 8 \
  --backbone uvit --in_channels 5 --compress_ratio 2 \
  --ct_ckpt   ${CKPT}/vqvae_ct_n5_cpr2_img128/best.pt \
  --cbct_ckpt ${CKPT}/vqvae_cbct_n5_cpr2_img128/best.pt \
  --exp_name 1_vdm_uvit_n5_cpr2_img128 --device 5 \
  ${COMMON} > logs/stage2_uvit_n5_cpr2.log 2>&1 &

wait

echo "=== Ablation 2 done ==="

echo "=== All stage2 ablations done ==="