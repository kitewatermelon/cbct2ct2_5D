"""stage1_vqvae.py — VQVAE 학습 (Stage 1).

데이터: PreprocessedDataset (전처리된 슬라이스 .npy)
  - in:  (B, N, H, W)  N = in_channels (슬라이스 수)
  - out: (B, 1, H, W)  중앙 슬라이스만
로깅: WandB

[데이터셋 변경사항]
- data/preprocess.py 로 미리 저장된 슬라이스 파일 사용 → DataLoader stall 제거
- train: 전체 슬라이스 / val: 케이스당 중간 슬라이스 1장만
- 케이스 단위 split (seed=42) — split_seed42.json 에 기록됨
"""
from __future__ import annotations

import argparse
import pathlib

import torch
from torch.nn import L1Loss
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.losses.perceptual import PerceptualLoss
from monai.networks.nets import VQVAE, PatchDiscriminator
from monai.utils import set_determinism

# 전처리된 슬라이스 파일 기반 데이터셋
from data.preprocessed_dataset import (
    PreprocessedDataset,
    build_preprocessed_transforms,
)

from models.lvdm.utils import get_lr, setup_scheduler
from utils.wandb import finish, init_wandb, log_images, log_train, log_val


# ---------------------------------------------------------------------------
# Mean
# ---------------------------------------------------------------------------

class Mean:
    def __init__(self, device=None):
        self.device = device
        self.reset()

    def reset(self):
        self._sum   = torch.tensor(0.0, device=self.device)
        self._count = torch.tensor(0,   device=self.device)

    def update(self, val):
        if isinstance(val, torch.Tensor):
            val = val.detach()
        self._sum   += val
        self._count += 1

    def compute(self):
        return self._sum / self._count.clamp(min=1)


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------

def to_3ch(x: torch.Tensor) -> torch.Tensor:
    """(B, 1, H, W) → (B, 3, H, W). LPIPS용."""
    return x.repeat(1, 3, 1, 1)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Stage1 VQVAE 학습")

    # ── 데이터 ───────────────────────────────────────────────────────────
    p.add_argument("--preprocessed_root", type=str,
                   default="/data/prof2/mai/s2025/dataset/preprocessed",
                   help="data/preprocess.py 의 --output_root 경로")
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--modality",       type=str,   default="ct", choices=["cbct", "ct"])

    # ── 모델 ────────────────────────────────────────────────────────────
    p.add_argument("--in_channels",    type=int,   default=9)
    p.add_argument("--embedding_dim",  type=int,   default=1)
    p.add_argument("--num_embeddings", type=int,   default=2048)
    p.add_argument("--compress_ratio", type=int,   default=4, choices=[2, 4, 8])

    # ── 학습 ────────────────────────────────────────────────────────────
    p.add_argument("--device",         type=int,   default=0)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--num_epochs",     type=int,   default=1000)
    p.add_argument("--lr_g",           type=float, default=1e-4)
    p.add_argument("--lr_d",           type=float, default=5e-5)
    p.add_argument("--adv_weight",     type=float, default=0.01)
    p.add_argument("--perc_weight",    type=float, default=0.001)
    p.add_argument("--amp",            action="store_true", default=True)
    p.add_argument("--patience",       type=int,   default=20)

    # ── 체크포인트 & WandB ───────────────────────────────────────────────
    p.add_argument("--checkpoint_dir", type=str,   default="checkpoints/stage1_vqvae")
    p.add_argument("--resume",         type=str,   default=None)
    p.add_argument("--wandb_project",  type=str,   default="cbct2ct-stage1-128-1")
    p.add_argument("--wandb_entity",   type=str,   default=None)
    p.add_argument("--exp_name",       type=str,   default="vqvae_ct_n9_cpr4")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    ckpt_dir = pathlib.Path(args.checkpoint_dir) / args.exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ckpt] {ckpt_dir}")

    # init_wandb(
    #     config=vars(args),
    #     project=args.wandb_project,
    #     experiment_name=args.exp_name,
    #     entity=args.wandb_entity,
    # )
    set_determinism(seed=args.seed)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.device)
        print(f"GPU: {torch.cuda.get_device_name(args.device)}")

    # ── 데이터 ───────────────────────────────────────────────────────────
    train_ds = PreprocessedDataset(
        preprocessed_root = args.preprocessed_root,
        split     = "train",
        n_slices  = args.in_channels,
        modality  = [args.modality],
        transform = build_preprocessed_transforms([args.modality], augment=True),
    )
    val_ds = PreprocessedDataset(
        preprocessed_root = args.preprocessed_root,
        split     = "val",
        n_slices  = args.in_channels,
        modality  = [args.modality],
        transform = build_preprocessed_transforms([args.modality], augment=False),
    )
    print(f"train: {len(train_ds):,}슬라이스 ({len(train_ds.subject_dirs)}케이스) | "
          f"val: {len(val_ds):,}슬라이스 ({len(val_ds.subject_dirs)}케이스)")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True,  num_workers=args.num_workers,
        drop_last=True, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # ── 모델 ─────────────────────────────────────────────────────────────
    cfg = {
        2: (((2,4,1,1),(1,3,1,1),(1,3,1,1),(1,3,1,1)),
            ((1,3,1,1,0),(1,3,1,1,0),(1,3,1,1,0),(2,4,1,1,0))),
        4: (((2,4,1,1),(2,4,1,1),(1,3,1,1),(1,3,1,1)),
            ((1,3,1,1,0),(1,3,1,1,0),(2,4,1,1,0),(2,4,1,1,0))),
        8: (((2,4,1,1),(2,4,1,1),(2,4,1,1),(1,3,1,1)),
            ((1,3,1,1,0),(2,4,1,1,0),(2,4,1,1,0),(2,4,1,1,0))),
    }
    down, up = cfg[args.compress_ratio]
    mid = args.in_channels // 2   # 중앙 슬라이스 index

    model = VQVAE(
        spatial_dims     = 2,
        in_channels      = args.in_channels,
        out_channels     = 1,
        channels         = (128, 256, 512, 512),
        num_res_channels = 256,
        num_res_layers   = 2,
        downsample_parameters = down,
        upsample_parameters   = up,
        num_embeddings   = args.num_embeddings,
        embedding_dim    = args.embedding_dim,
        commitment_cost  = 0.4,
    ).to(device)

    discriminator = PatchDiscriminator(
        spatial_dims=2, in_channels=1, num_layers_d=3, channels=64,
    ).to(device)

    perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="squeeze").to(device)

    optimizer_g = torch.optim.Adam(model.parameters(),         lr=args.lr_g)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d)
    scheduler_g = setup_scheduler(optimizer_g, args.num_epochs, warmup_epochs=0, min_lr=args.lr_g * 0.1)
    scheduler_d = setup_scheduler(optimizer_d, args.num_epochs, warmup_epochs=0, min_lr=args.lr_d * 0.1)

    l1_loss  = L1Loss()
    adv_loss = PatchAdversarialLoss(criterion="least_squares")

    # ── Resume ───────────────────────────────────────────────────────────
    best_val_loss    = float("inf")
    patience_counter = 0
    start_epoch      = 1

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        discriminator.load_state_dict(ckpt["discriminator_state_dict"])
        optimizer_g.load_state_dict(ckpt["optimizer_g_state_dict"])
        optimizer_d.load_state_dict(ckpt["optimizer_d_state_dict"])
        scheduler_g.load_state_dict(ckpt["scheduler_g_state_dict"])
        scheduler_d.load_state_dict(ckpt["scheduler_d_state_dict"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"[resume] epoch {start_epoch}부터 재개")

    # ── 학습 루프 ─────────────────────────────────────────────────────────
    epbar = tqdm(range(start_epoch, args.num_epochs + 1), desc="Training", position=0)

    for epoch in epbar:

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        discriminator.train()

        tm = {k: Mean(device=device) for k in
              ["recon_loss", "vq_loss", "gen_loss", "disc_loss", "perceptual_loss", "perplexity"]}

        bbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, position=1)
        for batch in bbar:
            data   = batch[args.modality].to(device)   # (B, N, H, W)
            target = data[:, mid:mid+1].float()         # (B, 1, H, W)

            # Generator
            optimizer_g.zero_grad()
            with torch.amp.autocast("cuda", enabled=args.amp, dtype=torch.bfloat16):
                recon, vq_loss  = model(images=data)
                logits_fake     = discriminator(recon.contiguous().float())[-1]
                recon_loss      = l1_loss(recon.float(), target)
                p_loss          = perceptual_loss(to_3ch(recon.float()), to_3ch(target))
                gen_loss        = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g          = recon_loss + vq_loss + args.perc_weight * p_loss + args.adv_weight * gen_loss

            loss_g.backward()
            optimizer_g.step()

            # Discriminator
            optimizer_d.zero_grad()
            with torch.amp.autocast("cuda", enabled=args.amp, dtype=torch.bfloat16):
                logits_fake = discriminator(recon.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(target.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True,  for_discriminator=True)
                loss_d      = args.adv_weight * (loss_d_fake + loss_d_real) * 0.5

            loss_d.backward()
            optimizer_d.step()

            tm["recon_loss"].update(recon_loss)
            tm["vq_loss"].update(vq_loss)
            tm["gen_loss"].update(gen_loss)
            tm["disc_loss"].update(loss_d)
            tm["perceptual_loss"].update(p_loss)
            tm["perplexity"].update(model.quantizer.perplexity)

            bbar.set_postfix(recon=f"{recon_loss.item():.4f}", vq=f"{vq_loss.item():.4f}")

        train_vals = {k: v.compute().item() for k, v in tm.items()}
        log_train(train_vals, epoch)

        # ── Val ───────────────────────────────────────────────────────────
        model.eval()
        vm = {k: Mean(device=device) for k in ["recon_loss", "perceptual_loss"]}

        with torch.no_grad():
            for step, batch in enumerate(tqdm(val_loader, desc="Val", leave=False)):
                data   = batch[args.modality].to(device)
                target = data[:, mid:mid+1].float()

                with torch.amp.autocast("cuda", enabled=args.amp, dtype=torch.bfloat16):
                    recon, _   = model(images=data)
                    recon_loss = l1_loss(recon.float(), target)
                    p_loss     = perceptual_loss(to_3ch(recon.float()), to_3ch(target))

                vm["recon_loss"].update(recon_loss)
                vm["perceptual_loss"].update(p_loss)

                if step == 0:
                    log_images("val/input_mid", data[:, mid:mid+1].detach().float(), epoch)
                    log_images("val/recon",     recon.detach().float(),              epoch)

        val_vals = {k: v.compute().item() for k, v in vm.items()}
        log_val(val_vals, epoch)

        scheduler_g.step()
        scheduler_d.step()

        epbar.set_postfix(
            recon     = f"{train_vals['recon_loss']:.4f}",
            val_recon = f"{val_vals['recon_loss']:.4f}",
            vq        = f"{train_vals['vq_loss']:.4f}",
            lr        = f"{get_lr(optimizer_g):.2e}",
        )

        # ── Best 저장 & Early Stopping ────────────────────────────────────
        if val_vals["recon_loss"] < best_val_loss:
            best_val_loss    = val_vals["recon_loss"]
            patience_counter = 0
            torch.save({
                "epoch":                    epoch,
                "best_val_loss":            best_val_loss,
                "model_state_dict":         model.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "optimizer_g_state_dict":   optimizer_g.state_dict(),
                "optimizer_d_state_dict":   optimizer_d.state_dict(),
                "scheduler_g_state_dict":   scheduler_g.state_dict(),
                "scheduler_d_state_dict":   scheduler_d.state_dict(),
            }, ckpt_dir / "best.pt")
            print(f"[best] epoch {epoch}  val_recon={best_val_loss:.4f} → saved")
        else:
            patience_counter += 1
            print(f"[patience] {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f"[early stop] epoch {epoch}")
                break

    finish()


if __name__ == "__main__":
    main()