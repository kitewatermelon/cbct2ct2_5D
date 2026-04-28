#!/bin/bash
# run_ablation.sh — Stage1 VQVAE ablation (GPU 4,5,6)

set -e
mkdir -p logs

DATA=/data/prof2/mai/s2025
CKPT_DIR=checkpoints/stage1_vqvae

run() {
    local MOD=$1 N=$2 CPR=$3 GPU=$4
    local NAME="1_vqvae_${MOD}_n${N}_cpr${CPR}_img128"
    local LATEST="${CKPT_DIR}/${NAME}/best.pt"
    local RESUME_ARG=""

    if [ -f "$LATEST" ]; then
        echo "[resume] $NAME"
        RESUME_ARG="--resume $LATEST"
    fi

    PYTHONPATH=. python stage1_vqvae.py \
        --modality $MOD --in_channels $N --compress_ratio $CPR \
        --exp_name $NAME --data_root $DATA --device $GPU --embedding_dim 1 \
        $RESUME_ARG \
        > logs/${NAME}.log 2>&1 &
}

for MOD in ct cbct; do
    echo "=== [$MOD] Ablation 1-A: n1, n3, n5 @ cpr4 ==="
    run $MOD 1 4 4
    run $MOD 3 4 5
    run $MOD 5 4 6
    wait

    echo "=== [$MOD] Ablation 1-B: n7, n9 @ cpr4 ==="
    run $MOD 7 4 4
    run $MOD 9 4 5
    wait

    echo "=== [$MOD] Ablation 2: cpr2, cpr8 @ n5 ==="
    run $MOD 5 8 4
    run $MOD 5 2 5
    wait

    echo "=== [$MOD] done ==="
done

echo "=== All stage1 ablations done ==="