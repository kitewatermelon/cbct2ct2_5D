#!/bin/bash
# run_stage2_ablation.sh — Stage2 VDM ablation (직렬 실행, GPU 단일)
# Group 1: uvit × (n1,n3,n5,n7,n9) × cpr4
# Group 2: uvit × n5 × (cpr2,cpr8)
# Group 3: unet × n5 × cpr4

mkdir -p logs

DATA=../s2025
CKPT=checkpoints/stage1_vqvae
DEVICE=0

NUM_STEPS=100000
EVAL_EVERY=5000
LOG_EVERY=100
WARMUP_STEPS=5000
SAMPLE_STEPS_EVAL=20
FID_WARMUP_STEPS=50000
QUAL_SAVE_EVERY=5
NUM_EVAL_SAMPLES=4

COMMON="--num_steps ${NUM_STEPS} \
        --eval_every ${EVAL_EVERY} \
        --log_every ${LOG_EVERY} \
        --warmup_steps ${WARMUP_STEPS} \
        --sample_steps_eval ${SAMPLE_STEPS_EVAL} \
        --fid_warmup_steps ${FID_WARMUP_STEPS} \
        --qual_save_every ${QUAL_SAVE_EVERY} \
        --num_eval_samples ${NUM_EVAL_SAMPLES} \
        --data_root ${DATA} \
        --embedding_dim 1"

run_exp() {
    local backbone=$1 n=$2 cpr=$3
    local in_ch=$n
    local exp_name="1_vdm_${backbone}_n${n}_cpr${cpr}_img128"
    local log="logs/stage2_${backbone}_n${n}_cpr${cpr}.log"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START: ${exp_name}"
    PYTHONPATH=. python train/stage2_vdm.py --batch_size 8 \
        --backbone ${backbone} --in_channels ${in_ch} --compress_ratio ${cpr} \
        --ct_ckpt   ${CKPT}/1_vqvae_ct_n${n}_cpr${cpr}_img128/best.pt \
        --cbct_ckpt ${CKPT}/1_vqvae_cbct_n${n}_cpr${cpr}_img128/best.pt \
        --exp_name ${exp_name} --device ${DEVICE} \
        ${COMMON} > "${log}" 2>&1
    local status=$?
    if [ ${status} -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE:  ${exp_name}"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAIL:  ${exp_name} (exit ${status})"
    fi
    return ${status}
}

# ===========================================================================
# Group 1: uvit × (n1,n3,n5,n7,n9) × cpr4
# ===========================================================================

run_exp uvit 9 4
run_exp uvit 7 4
run_exp uvit 5 4
run_exp uvit 3 4
run_exp uvit 1 4

# ===========================================================================
# Group 2: uvit × n5 × (cpr2, cpr8)
# ===========================================================================
run_exp uvit 5 2
run_exp uvit 5 8

# ===========================================================================
# Group 3: unet × n5 × cpr4
# ===========================================================================
run_exp unet 5 4

echo "All experiments finished."
