"""inspect_latent_dist.py — VQ-VAE latent distribution check (mean/std per batch).

Usage:
  python inspect_latent_dist.py \
      --ct_ckpt   checkpoints/stage1_ct/best.pt \
      --cbct_ckpt checkpoints/stage1_cbct/best.pt \
      --data_root /home/dministrator/s2025 \
      --max_batches 100
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from monai.networks.nets import VQVAE
from monai.utils import set_determinism

from data.synthrad2025 import SynthRad2025, build_transforms


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="VQ-VAE latent distribution inspector")
    p.add_argument("--data_root",      type=str,   default="/home/dministrator/s2025")
    p.add_argument("--anatomy",        nargs="+",  default=["AB", "HN", "TH"])
    p.add_argument("--spatial_size",   type=int,   default=128)
    p.add_argument("--in_channels",    type=int,   default=5)
    p.add_argument("--ct_ckpt",        type=str,   required=True)
    p.add_argument("--cbct_ckpt",      type=str,   required=True)
    p.add_argument("--compress_ratio", type=int,   default=4, choices=[2, 4, 8])
    p.add_argument("--embedding_dim",  type=int,   default=4)
    p.add_argument("--num_embeddings", type=int,   default=2048)
    p.add_argument("--batch_size",     type=int,   default=16)
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--val_ratio",      type=float, default=0.2)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--device",         type=int,   default=0)
    p.add_argument("--max_batches",    type=int,   default=None,
                   help="max number of batches to inspect (None = all)")
    p.add_argument("--split",          type=str,   default="val",
                   choices=["train", "val", "all"],
                   help="dataset split to use")
    p.add_argument("--output_dir",     type=str,   default="latent_dist_results")
    return p.parse_args()


# ---------------------------------------------------------------------------
# VQVAE builder (same as stage2_vdm.py)
# ---------------------------------------------------------------------------

def build_vqvae(in_channels, out_channels, compress_ratio, embedding_dim, num_embeddings):
    cfg = {
        2: (((2,4,1,1),(1,3,1,1),(1,3,1,1),(1,3,1,1)),
            ((1,3,1,1,0),(1,3,1,1,0),(1,3,1,1,0),(2,4,1,1,0))),
        4: (((2,4,1,1),(2,4,1,1),(1,3,1,1),(1,3,1,1)),
            ((1,3,1,1,0),(1,3,1,1,0),(2,4,1,1,0),(2,4,1,1,0))),
        8: (((2,4,1,1),(2,4,1,1),(2,4,1,1),(1,3,1,1)),
            ((1,3,1,1,0),(2,4,1,1,0),(2,4,1,1,0),(2,4,1,1,0))),
    }
    down, up = cfg[compress_ratio]
    return VQVAE(
        spatial_dims=2, in_channels=in_channels, out_channels=out_channels,
        channels=(128, 256, 512, 512), num_res_channels=256, num_res_layers=2,
        downsample_parameters=down, upsample_parameters=up,
        num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=0.4,
    )


def load_frozen_vqvae(ckpt_path, device, **kwargs):
    ae = build_vqvae(**kwargs).to(device)
    ae.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae


# ---------------------------------------------------------------------------
# Running statistics (online algorithm)
# ---------------------------------------------------------------------------

class RunningStats:
    """Per-channel global mean/std computed online."""

    def __init__(self, n_channels: int):
        self.n   = 0
        self.sum = np.zeros(n_channels, dtype=np.float64)
        self.sq  = np.zeros(n_channels, dtype=np.float64)

    def update(self, z: torch.Tensor):
        z_np = z.float().cpu().numpy()
        n_new = z_np.shape[0] * z_np.shape[2] * z_np.shape[3]
        self.n   += n_new
        self.sum += z_np.sum(axis=(0, 2, 3))
        self.sq  += (z_np ** 2).sum(axis=(0, 2, 3))

    @property
    def mean(self):
        return self.sum / self.n

    @property
    def std(self):
        var = self.sq / self.n - self.mean ** 2
        return np.sqrt(np.maximum(var, 0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    set_determinism(seed=args.seed)

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.device)
        print(f"GPU: {torch.cuda.get_device_name(args.device)}")

    # ── Data ────────────────────────────────────────────────────────────────
    ss = (args.spatial_size, args.spatial_size)
    full_ds = SynthRad2025(
        root=f"{args.data_root}/dataset/train/n{args.in_channels}",
        modality=["cbct", "ct"], anatomy=args.anatomy,
        transform=build_transforms(["cbct", "ct"], spatial_size=ss, augment=False),
    )
    n_val   = int(len(full_ds) * args.val_ratio)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    ds_map = {"train": train_ds, "val": val_ds, "all": full_ds}
    dataset = ds_map[args.split]
    print(f"[{args.split}] {len(dataset)} samples")

    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True, drop_last=False,
    )

    # ── VQVAE ───────────────────────────────────────────────────────────────
    vqvae_kwargs = dict(out_channels=1, compress_ratio=args.compress_ratio,
                        embedding_dim=args.embedding_dim, num_embeddings=args.num_embeddings)
    ct_ae   = load_frozen_vqvae(args.ct_ckpt,   device, in_channels=args.in_channels, **vqvae_kwargs)
    cbct_ae = load_frozen_vqvae(args.cbct_ckpt, device, in_channels=args.in_channels, **vqvae_kwargs)

    with torch.no_grad():
        dummy        = torch.zeros(1, args.in_channels, *ss, device=device)
        latent_shape = tuple(ct_ae.encode_stage_2_inputs(dummy).shape[1:])
    n_channels = latent_shape[0]
    print(f"latent shape (C, H, W): {latent_shape}")

    # ── Per-batch stats ──────────────────────────────────────────────────────
    ct_stats   = RunningStats(n_channels)
    cbct_stats = RunningStats(n_channels)

    batch_records = []

    print("\n=== Per-batch latent mean / std ===")
    header = (f"{'batch':>5}  "
              f"{'CT mean':>10}  {'CT std':>10}  {'CT min':>10}  {'CT max':>10}  "
              f"{'CBCT mean':>10}  {'CBCT std':>10}  {'CBCT min':>10}  {'CBCT max':>10}")
    print(header)
    print("-" * len(header))

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Encoding")):
            if args.max_batches is not None and i >= args.max_batches:
                break

            ct_img   = batch["ct"].to(device)
            cbct_img = batch["cbct"].to(device)

            z_ct   = ct_ae.encode_stage_2_inputs(ct_img)     # (B, C, H, W)
            z_cbct = cbct_ae.encode_stage_2_inputs(cbct_img)

            ct_stats.update(z_ct)
            cbct_stats.update(z_cbct)

            ct_mean = z_ct.mean().item()
            ct_std  = z_ct.std().item()
            ct_min  = z_ct.min().item()
            ct_max  = z_ct.max().item()

            cbct_mean = z_cbct.mean().item()
            cbct_std  = z_cbct.std().item()
            cbct_min  = z_cbct.min().item()
            cbct_max  = z_cbct.max().item()

            batch_records.append({
                "batch": i,
                "ct_mean": ct_mean, "ct_std": ct_std,
                "ct_min": ct_min,   "ct_max": ct_max,
                "cbct_mean": cbct_mean, "cbct_std": cbct_std,
                "cbct_min": cbct_min,   "cbct_max": cbct_max,
            })

            print(f"{i:>5}  "
                  f"{ct_mean:>10.4f}  {ct_std:>10.4f}  {ct_min:>10.4f}  {ct_max:>10.4f}  "
                  f"{cbct_mean:>10.4f}  {cbct_std:>10.4f}  {cbct_min:>10.4f}  {cbct_max:>10.4f}")

    # ── Global summary ───────────────────────────────────────────────────────
    print("\n=== Per-channel global stats ===")
    print(f"{'ch':>6}  {'CT mean':>10}  {'CT std':>10}  {'CBCT mean':>10}  {'CBCT std':>10}")
    for c in range(n_channels):
        print(f"{c:>6}  "
              f"{ct_stats.mean[c]:>10.4f}  {ct_stats.std[c]:>10.4f}  "
              f"{cbct_stats.mean[c]:>10.4f}  {cbct_stats.std[c]:>10.4f}")

    print(f"\n[CT   global] mean={ct_stats.mean.mean():.4f}  std={ct_stats.std.mean():.4f}")
    print(f"[CBCT global] mean={cbct_stats.mean.mean():.4f}  std={cbct_stats.std.mean():.4f}")

    # ── Plots ────────────────────────────────────────────────────────────────
    batches = [r["batch"] for r in batch_records]

    # 1. Per-batch mean / std trend
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"VQ-VAE Latent Distribution ({args.split} split)", fontsize=13)

    for ax, key, title, color in [
        (axes[0, 0], "ct_mean",   "CT   — mean per batch",   "steelblue"),
        (axes[0, 1], "ct_std",    "CT   — std per batch",    "steelblue"),
        (axes[1, 0], "cbct_mean", "CBCT — mean per batch",   "darkorange"),
        (axes[1, 1], "cbct_std",  "CBCT — std per batch",    "darkorange"),
    ]:
        vals = [r[key] for r in batch_records]
        ax.plot(batches, vals, lw=1.2, color=color)
        ax.axhline(np.mean(vals), ls="--", color="gray", lw=1, label=f"avg={np.mean(vals):.4f}")
        ax.set_title(title)
        ax.set_xlabel("batch")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "batch_mean_std.png", dpi=150)
    plt.close(fig)

    # 2. Per-channel global mean / std bar chart
    ch = list(range(n_channels))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Per-channel global mean / std", fontsize=13)

    w = 0.35
    axes[0].bar([c - w/2 for c in ch], ct_stats.mean,   width=w, label="CT",   color="steelblue")
    axes[0].bar([c + w/2 for c in ch], cbct_stats.mean, width=w, label="CBCT", color="darkorange")
    axes[0].set_title("Mean per channel")
    axes[0].set_xlabel("channel")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar([c - w/2 for c in ch], ct_stats.std,   width=w, label="CT",   color="steelblue")
    axes[1].bar([c + w/2 for c in ch], cbct_stats.std, width=w, label="CBCT", color="darkorange")
    axes[1].set_title("Std per channel")
    axes[1].set_xlabel("channel")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "channel_global_stats.png", dpi=150)
    plt.close(fig)

    # 3. Per-batch value range (min / max)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("Latent value range per batch (min / max)", fontsize=13)
    for ax, mod, color in [(axes[0], "ct", "steelblue"), (axes[1], "cbct", "darkorange")]:
        mins = [r[f"{mod}_min"] for r in batch_records]
        maxs = [r[f"{mod}_max"] for r in batch_records]
        ax.fill_between(batches, mins, maxs, alpha=0.3, color=color)
        ax.plot(batches, mins, lw=1, color=color, label="min")
        ax.plot(batches, maxs, lw=1, color=color, ls="--", label="max")
        ax.set_title(mod.upper())
        ax.set_xlabel("batch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "batch_minmax.png", dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to {out_dir}/")


if __name__ == "__main__":
    main()
