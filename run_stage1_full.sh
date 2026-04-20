#!/bin/bash
# run_stage1_full.sh — Stage1 VQVAE full dataset ablation
# 조합: n 비교(cpr4 고정) × cpr 비교(n5 고정) = 7개
#   n:   n1,n3,n5,n7,n9 @ cpr4
#   cpr: cpr2,cpr8      @ n5  (n5_cpr4는 위에서 이미 커버)
# ct + cbct = 14개, GPU 4,5,6 병렬
set -e
mkdir -p logs

DATA=/data/prof2/mai/s2025/dataset
CKPT_DIR=checkpoints/stage1_vqvae
BATCH=128
WORKERS=8

run() {
    local MOD=$1 N=$2 CPR=$3 GPU=$4
    local NAME="full_vqvae_${MOD}_n${N}_cpr${CPR}_img128"
    local RESUME_ARG=""
    if [ -f "${CKPT_DIR}/${NAME}/best.pt" ]; then
        echo "[resume] $NAME"
        RESUME_ARG="--resume ${CKPT_DIR}/${NAME}/best.pt"
    fi

    PYTHONPATH=. python stage1_full.py \
        --modality       $MOD \
        --in_channels    $N \
        --compress_ratio $CPR \
        --embedding_dim  1 \
        --num_embeddings 2048 \
        --data_root      $DATA \
        --exp_name       $NAME \
        --device         $GPU \
        --batch_size     $BATCH \
        --num_workers    $WORKERS \
        --num_epochs     100 \
        --patience       20 \
        --lr_g           1e-4 \
        --lr_d           5e-5 \
        --wandb_project  cbct2ct-stage1-full \
        $RESUME_ARG \
        > logs/${NAME}.log 2>&1 &

    echo "[launched] $NAME → GPU $GPU (batch=$BATCH, workers=$WORKERS) | log: logs/${NAME}.log"
}

for MOD in ct cbct; do
    echo "=== [$MOD] Ablation ==="

    # n ablation: cpr4 고정, n1,n3,n5,n7,n9
    echo "--- [$MOD] n ablation @ cpr4: n1,n3,n5 ---"
    run $MOD 1 4 4
    run $MOD 3 4 5
    run $MOD 5 4 6
    wait

    echo "--- [$MOD] n ablation @ cpr4: n7,n9 ---"
    run $MOD 7 4 4
    run $MOD 9 4 5
    wait

    # cpr ablation: n5 고정, cpr2,cpr8 (cpr4는 위에서 완료)
    echo "--- [$MOD] cpr ablation @ n5: cpr2,cpr8 ---"
    run $MOD 5 2 4
    run $MOD 5 8 5
    wait

    echo "=== [$MOD] done ==="
done

echo "=== All stage1 full ablations done ==="