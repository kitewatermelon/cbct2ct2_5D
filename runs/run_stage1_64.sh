set -e
mkdir -p logs

DATA=/data/prof2/mai/s2025
DEVICES=(4 5 6)

PYTHONPATH=. python train/stage1_vqvae.py --modality cbct --in_channels 5 --latent_size 64 --exp_name vqvae_cbct_n5_lat64 --data_root $DATA --device 5 --batch_size 32 > logs/cbct_n5_lat64.log 2>&1 &
PYTHONPATH=. python train/stage1_vqvae.py --modality ct --in_channels 5 --latent_size 64 --exp_name vqvae_ct_n5_lat64 --data_root $DATA --device 4 --batch_size 32 > logs/ct_n5_lat64.log 2>&1 &
