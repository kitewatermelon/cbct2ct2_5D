#!/usr/bin/env bash
set -e

python -m train.cyclegan \
  --data_root /home/dministrator/s2025 \
  --anatomy AB HN TH \
  --spatial_size 128 \
  --batch_size 4 \
  --n_epochs 200 \
  --lr 2e-4 \
  --lambda_cycle 10.0 \
  --lambda_idt 5.0 \
  --ngf 64 \
  --n_blocks 9 \
  --num_workers 4 \
  --eval_every 5 \
  --checkpoint_dir checkpoints/cyclegan \
  --wandb_project cbct2ct-cyclegan \
  --exp_name cyclegan_2d \
  --device 0
