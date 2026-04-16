#!/bin/bash
# run_ablation.sh — Stage1 VQVAE ablation (GPU 4,5,6)
# Ablation 1: in_channels (1,3,5,7,9) @ latent_size=32
# Ablation 2: latent_size (16,64)      @ in_channels=5  (32는 Ablation1에서 커버)

set -e
mkdir -p logs

DATA=/data/prof2/mai/s2025
DEVICES=(4 5 6)

for MOD in ct cbct; do
    echo "=== [$MOD] Ablation 1-A: n1, n3, n5 ==="
    PYTHONPATH=. python stage1_vqvae.py --modality $MOD --in_channels 1 --latent_size 32 --exp_name vqvae_${MOD}_n1_lat32 --data_root $DATA --device 4 > logs/${MOD}_n1_lat32.log 2>&1 &
    PYTHONPATH=. python stage1_vqvae.py --modality $MOD --in_channels 3 --latent_size 32 --exp_name vqvae_${MOD}_n3_lat32 --data_root $DATA --device 5 > logs/${MOD}_n3_lat32.log 2>&1 &
    PYTHONPATH=. python stage1_vqvae.py --modality $MOD --in_channels 5 --latent_size 32 --exp_name vqvae_${MOD}_n5_lat32 --data_root $DATA --device 6 > logs/${MOD}_n5_lat32.log 2>&1 &
    wait

    echo "=== [$MOD] Ablation 1-B: n7, n9 ==="
    PYTHONPATH=. python stage1_vqvae.py --modality $MOD --in_channels 7 --latent_size 32 --exp_name vqvae_${MOD}_n7_lat32 --data_root $DATA --device 4 > logs/${MOD}_n7_lat32.log 2>&1 &
    PYTHONPATH=. python stage1_vqvae.py --modality $MOD --in_channels 9 --latent_size 32 --exp_name vqvae_${MOD}_n9_lat32 --data_root $DATA --device 5 > logs/${MOD}_n9_lat32.log 2>&1 &
    wait

    echo "=== [$MOD] Ablation 2: lat16, lat64 @ n5 ==="
    PYTHONPATH=. python stage1_vqvae.py --modality $MOD --in_channels 5 --latent_size 16 --exp_name vqvae_${MOD}_n5_lat16 --data_root $DATA --device 4 > logs/${MOD}_n5_lat16.log 2>&1 &
    PYTHONPATH=. python stage1_vqvae.py --modality $MOD --in_channels 5 --latent_size 64 --exp_name vqvae_${MOD}_n5_lat64 --data_root $DATA --device 5 > logs/${MOD}_n5_lat64.log 2>&1 &
    wait

    echo "=== [$MOD] done ==="
done

echo "=== All ablations done ==="