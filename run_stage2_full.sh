#!/bin/bash
# run_stage2_full.sh — Stage2 VDM full dataset ablation
# 조합: n 비교(cpr4 고정) × cpr 비교(n5 고정) = 7개
#   n:   n1,n3,n5,n7,n9 @ cpr4
#   cpr: cpr2,cpr8      @ n5  (n5_cpr4는 위에서 이미 커버)
# GPU 4,5,6 병렬
set -e
mkdir -p logs

DATA=/data/prof2/mai/s2025/dataset
CKPT_S1=checkpoints/stage1_vqvae
CKPT_S2=checkpoints/stage2_vdm
BATCH=64
WORKERS=8

COMMON="--embedding_dim     1 \
        --num_embeddings    2048 \
        --backbone          uvit \
        --data_root         $DATA \
        --num_steps         100000 \
        --batch_size        $BATCH \
        --num_workers       $WORKERS \
        --eval_every        5000 \
        --log_every         100 \
        --warmup_steps      5000 \
        --sample_steps_eval 20 \
        --fid_warmup_steps  10000 \
        --qual_save_every   5 \
        --num_eval_samples  4 \
        --wandb_project     cbct2ct-stage2-full"

run() {
    local N=$1 CPR=$2 GPU=$3
    local NAME="full_vdm_uvit_n${N}_cpr${CPR}_img128"
    local CT_CKPT="${CKPT_S1}/full_vqvae_ct_n${N}_cpr${CPR}_img128/best.pt"
    local CBCT_CKPT="${CKPT_S1}/full_vqvae_cbct_n${N}_cpr${CPR}_img128/best.pt"
    local RESUME_ARG=""

    if [ ! -f "$CT_CKPT" ];   then echo "[ERROR] CT ckpt 없음: $CT_CKPT";   exit 1; fi
    if [ ! -f "$CBCT_CKPT" ]; then echo "[ERROR] CBCT ckpt 없음: $CBCT_CKPT"; exit 1; fi
    if [ -f "${CKPT_S2}/${NAME}/best.pt" ]; then
        echo "[resume] $NAME"
        RESUME_ARG="--resume ${CKPT_S2}/${NAME}/best.pt"
    fi

    PYTHONPATH=. python stage2_full.py \
        --in_channels    $N \
        --compress_ratio $CPR \
        --ct_ckpt        $CT_CKPT \
        --cbct_ckpt      $CBCT_CKPT \
        --exp_name       $NAME \
        --device         $GPU \
        $COMMON \
        $RESUME_ARG \
        > logs/${NAME}.log 2>&1 &

    echo "[launched] $NAME → GPU $GPU (batch=$BATCH, workers=$WORKERS) | log: logs/${NAME}.log"
}

echo "=== Stage2 Full Ablation ==="

# n ablation: cpr4 고정, n1,n3,n5,n7,n9
echo "--- n ablation @ cpr4: n1,n3,n5 ---"
run 1 4 4
run 3 4 5
run 5 4 6
wait

echo "--- n ablation @ cpr4: n7,n9 ---"
run 7 4 4
run 9 4 5
run 5 2 6
wait

echo "=== All stage2 full ablations done ==="