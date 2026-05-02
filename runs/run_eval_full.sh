#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/home/dministrator/s2025}"
CKPT_BASE="${CKPT_BASE:-checkpoints/stage2_vdm}"
VQVAE_BASE="${VQVAE_BASE:-checkpoints/stage1_vqvae}"
CYCLEGAN_CKPT="${CYCLEGAN_CKPT:-checkpoints/cyclegan/last.pt}"
GEN_DIR="${GEN_DIR:-eval_results/gen}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_results}"
DEVICE="${DEVICE:-0}"
N_SAMPLE_STEPS="${N_SAMPLE_STEPS:-200}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"

echo "=============================="
echo " Step 1: eval_gen — inference & .mha 저장"
echo "=============================="
echo "  data_root     : $DATA_ROOT"
echo "  ckpt_base     : $CKPT_BASE"
echo "  vqvae_base    : $VQVAE_BASE"
echo "  cyclegan_ckpt : $CYCLEGAN_CKPT"
echo "  gen_dir       : $GEN_DIR"
echo "  device        : cuda:$DEVICE"
echo "  n_sample_steps: $N_SAMPLE_STEPS"
echo "  batch_size    : $BATCH_SIZE"
echo "=============================="

PYTHONPATH=. python eval/eval_gen.py \
    --data_root      "$DATA_ROOT" \
    --ckpt_base      "$CKPT_BASE" \
    --vqvae_base     "$VQVAE_BASE" \
    --cyclegan_ckpt  "$CYCLEGAN_CKPT" \
    --output_dir     "$GEN_DIR" \
    --device         "$DEVICE" \
    --n_sample_steps "$N_SAMPLE_STEPS" \
    --batch_size     "$BATCH_SIZE" \
    --num_workers    "$NUM_WORKERS"

echo ""
echo "=============================="
echo " Step 2: eval_full — .mha에서 metric 계산"
echo "=============================="
echo "  gen_dir    : $GEN_DIR"
echo "  output_dir : $OUTPUT_DIR"
echo "=============================="

PYTHONPATH=. python eval/eval_full.py \
    --data_root   "$DATA_ROOT" \
    --gen_dir     "$GEN_DIR" \
    --output_dir  "$OUTPUT_DIR" \
    --device      "$DEVICE" \
    --batch_size  "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS"

echo ""
echo "결과 파일:"
ls -lh "$OUTPUT_DIR"/
