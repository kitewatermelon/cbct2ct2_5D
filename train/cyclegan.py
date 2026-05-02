# train/cyclegan.py
"""train/cyclegan.py — CycleGAN 학습 (CBCT↔CT, 2D).

- SynthRad2025 n=1 데이터 (단일 슬라이스)
- Generator: ResNetGenerator (MONAI ResidualUnit×9)
- Discriminator: MONAI PatchDiscriminator (70×70)
- Loss: adversarial(MSE) + cycle(L1, λ=10) + identity(L1, λ=5)
- 로깅: WandB / 체크포인트: checkpoints/cyclegan/
"""
from __future__ import annotations

import argparse
import pathlib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from monai.metrics import PSNRMetric, SSIMMetric
from monai.utils import set_determinism
from tqdm.auto import tqdm

from data.synthrad2025 import SynthRad2025, build_transforms
from models.cyclegan.generator import ResNetGenerator
from models.cyclegan.discriminator import build_discriminator
from utils.wandb import init_wandb, log_train, log_val, finish


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CycleGAN 학습 (CBCT↔CT 2D)")
    p.add_argument("--data_root",    type=str,   default="/home/dministrator/s2025")
    p.add_argument("--anatomy",      nargs="+",  default=["AB", "HN", "TH"])
    p.add_argument("--spatial_size", type=int,   default=128)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--val_ratio",    type=float, default=0.2)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--n_epochs",     type=int,   default=200)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--lambda_cycle", type=float, default=10.0)
    p.add_argument("--lambda_idt",   type=float, default=5.0)
    p.add_argument("--ngf",          type=int,   default=64)
    p.add_argument("--n_blocks",     type=int,   default=9)
    p.add_argument("--device",       type=int,   default=0)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/cyclegan")
    p.add_argument("--wandb_project",  type=str, default="cbct2ct-cyclegan")
    p.add_argument("--wandb_entity",   type=str, default=None)
    p.add_argument("--exp_name",       type=str, default="cyclegan_2d")
    p.add_argument("--eval_every",   type=int,   default=5)
    return p.parse_args()


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------

def to_input(x: torch.Tensor) -> torch.Tensor:
    """[0,1] → [-1,1] (Generator 입력 정규화)."""
    return x * 2.0 - 1.0


def to_output(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] → [0,1] (메트릭 계산용)."""
    return (x + 1.0) / 2.0


@torch.no_grad()
def run_val(
    G_A2B: ResNetGenerator,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    G_A2B.eval()
    psnr_m = PSNRMetric(max_val=1.0)
    ssim_m = SSIMMetric(data_range=1.0, spatial_dims=2)

    for batch in val_loader:
        cbct = batch["cbct"].to(device)   # (B, 1, H, W)
        ct   = batch["ct"].to(device)     # (B, 1, H, W)

        ct_gen = to_output(G_A2B(to_input(cbct))).clamp(0, 1).float()
        ct_gt  = ct.float()

        psnr_m(ct_gen, ct_gt)
        ssim_m(ct_gen, ct_gt)

    G_A2B.train()
    return {
        "psnr": psnr_m.aggregate().item(),
        "ssim": ssim_m.aggregate().item(),
    }


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> None:
    args   = get_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    set_determinism(seed=args.seed)

    # ── 데이터 ──────────────────────────────────────────────────────────────
    data_root = f"{args.data_root}/dataset/train/n1"
    tf_train  = build_transforms(["cbct", "ct"], (args.spatial_size, args.spatial_size), augment=True)
    tf_val    = build_transforms(["cbct", "ct"], (args.spatial_size, args.spatial_size), augment=False)

    full_ds = SynthRad2025(
        root=data_root, modality=["cbct", "ct"],
        anatomy=args.anatomy, transform=tf_train,
    )
    n_val   = int(len(full_ds) * args.val_ratio)
    n_train = len(full_ds) - n_val
    train_ds, _ = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    val_ds_full = SynthRad2025(
        root=data_root, modality=["cbct", "ct"],
        anatomy=args.anatomy, transform=tf_val,
    )
    _, val_ds = random_split(
        val_ds_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── 모델 ────────────────────────────────────────────────────────────────
    G_A2B = ResNetGenerator(ngf=args.ngf, n_blocks=args.n_blocks).to(device)  # CBCT→CT
    G_B2A = ResNetGenerator(ngf=args.ngf, n_blocks=args.n_blocks).to(device)  # CT→CBCT
    D_A   = build_discriminator().to(device)  # CBCT 판별
    D_B   = build_discriminator().to(device)  # CT 판별

    opt_G = torch.optim.Adam(
        list(G_A2B.parameters()) + list(G_B2A.parameters()),
        lr=args.lr, betas=(0.9, 0.999),
    )
    opt_D = torch.optim.Adam(
        list(D_A.parameters()) + list(D_B.parameters()),
        lr=args.lr, betas=(0.9, 0.999),
    )

    decay_start = args.n_epochs // 2
    def lambda_lr(epoch: int) -> float:
        if epoch < decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_start) / decay_start)

    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=lambda_lr)
    sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda=lambda_lr)

    # ── WandB ────────────────────────────────────────────────────────────────
    init_wandb(vars(args), project=args.wandb_project,
               experiment_name=args.exp_name, entity=args.wandb_entity)

    ckpt_dir = pathlib.Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_psnr = -float("inf")

    # ── 학습 루프 ────────────────────────────────────────────────────────────
    for epoch in range(args.n_epochs):
        G_A2B.train(); G_B2A.train(); D_A.train(); D_B.train()

        loss_G_sum = loss_D_sum = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.n_epochs}", leave=False):
            cbct = batch["cbct"].to(device)  # (B, 1, H, W)  [0, 1]
            ct   = batch["ct"].to(device)    # (B, 1, H, W)  [0, 1]

            real_A = to_input(cbct)  # [-1, 1]
            real_B = to_input(ct)

            # ---- Generator ----
            opt_G.zero_grad()

            fake_B = G_A2B(real_A)
            fake_A = G_B2A(real_B)

            # Adversarial (LSGAN)
            pred_fake_B = D_B(fake_B)[-1]
            pred_fake_A = D_A(fake_A)[-1]
            loss_adv = (
                F.mse_loss(pred_fake_B, torch.ones_like(pred_fake_B)) +
                F.mse_loss(pred_fake_A, torch.ones_like(pred_fake_A))
            )

            # Cycle consistency
            rec_A = G_B2A(fake_B)
            rec_B = G_A2B(fake_A)
            loss_cycle = (
                F.l1_loss(rec_A, real_A) + F.l1_loss(rec_B, real_B)
            ) * args.lambda_cycle

            # Identity
            idt_B = G_A2B(real_B)
            idt_A = G_B2A(real_A)
            loss_idt = (
                F.l1_loss(idt_B, real_B) + F.l1_loss(idt_A, real_A)
            ) * args.lambda_idt

            loss_G = loss_adv + loss_cycle + loss_idt
            loss_G.backward()
            opt_G.step()

            # ---- Discriminator ----
            opt_D.zero_grad()

            pred_real_B = D_B(real_B.detach())[-1]
            pred_fake_B_stop = D_B(fake_B.detach())[-1]
            loss_D_B = 0.5 * (
                F.mse_loss(pred_real_B, torch.ones_like(pred_real_B)) +
                F.mse_loss(pred_fake_B_stop, torch.zeros_like(pred_fake_B_stop))
            )

            pred_real_A = D_A(real_A.detach())[-1]
            pred_fake_A_stop = D_A(fake_A.detach())[-1]
            loss_D_A = 0.5 * (
                F.mse_loss(pred_real_A, torch.ones_like(pred_real_A)) +
                F.mse_loss(pred_fake_A_stop, torch.zeros_like(pred_fake_A_stop))
            )

            loss_D = loss_D_A + loss_D_B
            loss_D.backward()
            opt_D.step()

            loss_G_sum += loss_G.item()
            loss_D_sum += loss_D.item()
            n_batches  += 1

        sched_G.step()
        sched_D.step()

        log_train({
            "loss_G": loss_G_sum / n_batches,
            "loss_D": loss_D_sum / n_batches,
            "lr":     sched_G.get_last_lr()[0],
        }, epoch=epoch)

        # ---- Validation ----
        if (epoch + 1) % args.eval_every == 0:
            val_metrics = run_val(G_A2B, val_loader, device)
            log_val(val_metrics, epoch=epoch)
            print(f"[Epoch {epoch+1}] PSNR={val_metrics['psnr']:.2f}  SSIM={val_metrics['ssim']:.4f}")

            if val_metrics["psnr"] > best_psnr:
                best_psnr = val_metrics["psnr"]
                torch.save({
                    "epoch":              epoch,
                    "G_A2B_state_dict":   G_A2B.state_dict(),
                    "G_B2A_state_dict":   G_B2A.state_dict(),
                    "D_A_state_dict":     D_A.state_dict(),
                    "D_B_state_dict":     D_B.state_dict(),
                    "opt_G_state_dict":   opt_G.state_dict(),
                    "opt_D_state_dict":   opt_D.state_dict(),
                    "best_psnr":          best_psnr,
                    "args":               vars(args),
                }, ckpt_dir / "best.pt")

    # 최종 체크포인트 (resume 가능하도록 옵티마이저 포함)
    torch.save({
        "epoch":              args.n_epochs - 1,
        "G_A2B_state_dict":   G_A2B.state_dict(),
        "G_B2A_state_dict":   G_B2A.state_dict(),
        "D_A_state_dict":     D_A.state_dict(),
        "D_B_state_dict":     D_B.state_dict(),
        "opt_G_state_dict":   opt_G.state_dict(),
        "opt_D_state_dict":   opt_D.state_dict(),
        "args":               vars(args),
    }, ckpt_dir / "latest.pt")

    finish()
    print(f"학습 완료. Best PSNR: {best_psnr:.2f}")


if __name__ == "__main__":
    main()
