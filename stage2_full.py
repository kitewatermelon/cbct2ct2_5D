"""stage2_vdm.py — VDM 학습 (Stage 2).

- Stage 1 VQVAE (CT, CBCT) 고정
- CT latent → VDM denoising (conditioned on CBCT latent)
- backbone: UViT or DiffusionModelUNet
- 로깅: WandB
- 평가: SSIM, PSNR, MSE, FID (step 기반)

[데이터셋 변경사항]
- 볼륨 단위 → 슬라이스 단위 학습 (split_dataset 사용)
- train: 전체 슬라이스 / val: 케이스당 중간 슬라이스 1장만 (val_middle_only=True)
- 케이스 단위 split으로 data leakage 방지
- 두 데이터 경로(Task2_Train, Task2_Train_D) 동시 로드

[최적화 변경사항]
- sample_steps_eval 기본값: 100 → 20
- eval_every 기본값: 2_000 → 5_000
- num_eval_samples 기본값: 8 → 4
- FID 계산: step > fid_warmup_steps 이후에만 실행 (기본 10_000)
- qualitative grid 저장: eval_every * qual_save_every 마다만 저장
- DataLoader persistent_workers=True, prefetch_factor=2 추가
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from torchvision.models import inception_v3
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from monai.metrics import PSNRMetric, SSIMMetric
from monai.metrics.fid import FIDMetric
from monai.networks.nets import VQVAE, DiffusionModelUNet
from monai.utils import set_determinism

# 새 슬라이스 단위 데이터셋
from data.dataset import SynthRad2025, build_transforms, split_dataset

from models.lvdm.uvit import UViT
from models.lvdm.vdm import VDM
from models.lvdm.utils import get_lr
from utils.wandb import finish, init_wandb, log_train, log_images


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Stage2 VDM 학습")

    # ── 데이터 ───────────────────────────────────────────────────────────
    p.add_argument("--data_root",        type=str,
                   default="/data/prof2/mai/s2025/dataset",
                   help="synthRAD2025_Task2_Train / _Train_D 의 상위 디렉토리")
    p.add_argument("--anatomy",          nargs="+",  default=["AB", "HN", "TH"])
    p.add_argument("--spatial_size",     type=int,   default=128)
    p.add_argument("--num_workers",      type=int,   default=4)
    p.add_argument("--val_ratio",        type=float, default=0.2)

    # ── 모델 ────────────────────────────────────────────────────────────
    p.add_argument("--in_channels",      type=int,   default=9)
    p.add_argument("--ct_ckpt",          type=str,   required=True)
    p.add_argument("--cbct_ckpt",        type=str,   required=True)
    p.add_argument("--compress_ratio",   type=int,   default=4,  choices=[2, 4, 8])
    p.add_argument("--embedding_dim",    type=int,   default=4)
    p.add_argument("--num_embeddings",   type=int,   default=2048)
    p.add_argument("--backbone",         type=str,   default="uvit", choices=["uvit", "unet"])
    p.add_argument("--uvit_embed_dim",   type=int,   default=512)
    p.add_argument("--uvit_depth",       type=int,   default=11)
    p.add_argument("--uvit_num_heads",   type=int,   default=4)
    p.add_argument("--uvit_patch_size",  type=int,   default=2)
    p.add_argument("--unet_channels",    nargs="+",  type=int, default=[128, 256, 512])

    # ── VDM ─────────────────────────────────────────────────────────────
    p.add_argument("--noise_schedule",   type=str,   default="fixed_linear")
    p.add_argument("--gamma_min",        type=float, default=-5.0)
    p.add_argument("--gamma_max",        type=float, default=5.0)
    p.add_argument("--antithetic",       action="store_true", default=True)
    p.add_argument("--n_sample_steps",   type=int,   default=200)

    # ── 학습 ────────────────────────────────────────────────────────────
    p.add_argument("--device",           type=int,   default=0)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--batch_size",       type=int,   default=16)
    p.add_argument("--num_steps",        type=int,   default=100_000)
    p.add_argument("--lr",               type=float, default=2e-4)
    p.add_argument("--weight_decay",     type=float, default=1e-4)
    p.add_argument("--amp",              action="store_true", default=True)
    p.add_argument("--warmup_steps",     type=int,   default=5_000)

    # ── Eval ─────────────────────────────────────────────────────────────
    p.add_argument("--eval_every",       type=int,   default=5_000)
    p.add_argument("--log_every",        type=int,   default=100)
    p.add_argument("--num_eval_samples", type=int,   default=4,
                   help="eval 시 샘플링할 최대 샘플 수")
    p.add_argument("--sample_steps_eval",type=int,   default=20,
                   help="eval 샘플링 DDPM 스텝 수")
    p.add_argument("--fid_warmup_steps", type=int,   default=10_000,
                   help="이 step 이후부터 FID 계산 시작")
    p.add_argument("--qual_save_every",  type=int,   default=5,
                   help="eval 몇 회마다 qualitative grid 저장할지")

    # ── 체크포인트 & WandB ───────────────────────────────────────────────
    p.add_argument("--checkpoint_dir",   type=str,   default="checkpoints/stage2_vdm")
    p.add_argument("--resume",           type=str,   default=None)
    p.add_argument("--wandb_project",    type=str,   default="cbct2ct-stage2-128-1")
    p.add_argument("--wandb_entity",     type=str,   default=None)
    p.add_argument("--exp_name",         type=str,   default="vdm_uvit_n9_cpr4")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------

class Mean:
    def __init__(self, device=None):
        self.device = device
        self.reset()

    def reset(self):
        self._sum   = torch.tensor(0.0, device=self.device)
        self._count = torch.tensor(0,   device=self.device)

    def update(self, val):
        self._sum   += val.detach() if isinstance(val, torch.Tensor) else val
        self._count += 1

    def compute(self):
        return (self._sum / self._count.clamp(min=1)).item()


def infinite_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def _prepare_cond(z_cond: torch.Tensor, backbone: str) -> torch.Tensor:
    """backbone 종류에 따라 conditioning tensor 형태 조정."""
    if backbone == "unet":
        B = z_cond.shape[0]
        return z_cond.view(B, 1, -1)
    return z_cond


def _lr_lambda(warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
    def fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return fn


# ---------------------------------------------------------------------------
# VQVAE 빌더
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
# 샘플링
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_conditional(vdm, cond, n_steps, device):
    z = torch.randn((cond.shape[0], *vdm.image_shape), device=device)
    steps = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
    for i in range(n_steps):
        z = vdm.sample_p_s_t(z, steps[i], steps[i + 1], clip_samples=True, context=cond)
    return z


# ---------------------------------------------------------------------------
# Inception (FID)
# ---------------------------------------------------------------------------

def build_inception(device):
    m = inception_v3(pretrained=True, transform_input=False).to(device)
    m.fc = torch.nn.Identity()
    m.eval()
    return m


@torch.no_grad()
def inception_feats(x: torch.Tensor, model) -> torch.Tensor:
    x = ((x.clamp(-1, 1) + 1) / 2).repeat(1, 3, 1, 1)
    return model(x).flatten(1)


# ---------------------------------------------------------------------------
# 체크포인트 저장
# ---------------------------------------------------------------------------

def save_ckpt(path, step, vdm, optimizer, scheduler, best_val_loss):
    torch.save({
        "step":                 step,
        "best_val_loss":        best_val_loss,
        "model_state_dict":     vdm.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, path)


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(vdm, ct_ae, cbct_ae, val_loader, args, device, step,
             ckpt_dir, inception, eval_count: int):
    vdm.eval()

    compute_fid = (step >= args.fid_warmup_steps)
    save_qual   = (eval_count % args.qual_save_every == 0)

    psnr_m = PSNRMetric(max_val=1.0)
    ssim_m = SSIMMetric(data_range=1.0, spatial_dims=2)
    fid_m  = FIDMetric()
    loss_m = Mean(device=device)

    mse_sum, n_seen = 0.0, 0
    feats_real, feats_fake = [], []
    qual_imgs, samples_done = [], 0

    mid = args.in_channels // 2   # 중간 슬라이스 인덱스

    for batch in tqdm(val_loader, desc=f"[Eval {step}]", leave=False):
        ct_img   = batch["ct"].to(device)    # (B, n_slices, H, W)
        cbct_img = batch["cbct"].to(device)  # (B, n_slices, H, W)

        # VQVAE encode
        z      = ct_ae.encode_stage_2_inputs(ct_img)
        z_cond = cbct_ae.encode_stage_2_inputs(cbct_img)
        cond   = _prepare_cond(z_cond, args.backbone)

        # val loss (샘플링 없이 — 빠름)
        with torch.amp.autocast("cuda", enabled=args.amp, dtype=torch.bfloat16):
            _, metrics = vdm(z, cond, ct_img)
        loss_m.update(metrics["bpd"])

        # 샘플링은 num_eval_samples만큼만
        if samples_done < args.num_eval_samples:
            with torch.amp.autocast("cuda", enabled=args.amp, dtype=torch.bfloat16):
                sampled_z = sample_conditional(vdm, cond, args.sample_steps_eval, device)
            ct_gen = ct_ae.decode_stage_2_outputs(sampled_z)   # (B, 1, H, W)

            # GT: 중간 슬라이스 1장 (diffusion output과 채널 맞춤)
            ct_gt = ct_img[:, mid:mid+1]   # (B, 1, H, W)

            psnr_m(ct_gen, ct_gt)
            ssim_m(ct_gen, ct_gt)
            mse_sum += F.mse_loss(ct_gen, ct_gt, reduction="sum").item()
            n_seen  += ct_gt.numel()

            if compute_fid:
                feats_fake.append(inception_feats(ct_gen, inception))
                feats_real.append(inception_feats(ct_gt,  inception))

            if save_qual:
                for i in range(ct_gen.shape[0]):
                    if len(qual_imgs) >= 4 * args.num_eval_samples:
                        break
                    diff = (ct_gen[i] - ct_gt[i]).abs()
                    for t in (cbct_img[i, mid:mid+1], ct_gen[i], ct_gt[i], diff):
                        qual_imgs.append(t.detach().cpu())

            # 첫 번째 배치만 WandB 이미지 로그
            if samples_done == 0:
                n = min(4, ct_gen.shape[0])
                log_images("eval/cbct_input", cbct_img[:n, mid:mid+1].float(), step)
                log_images("eval/ct_gen",     ct_gen[:n].float(),              step)
                log_images("eval/ct_gt",      ct_gt[:n].float(),               step)

            samples_done += ct_gen.shape[0]

    # qualitative grid 저장
    if save_qual and qual_imgs:
        grid = make_grid(torch.stack(qual_imgs), nrow=4, normalize=True, value_range=(-1, 1))
        plt.figure(figsize=(10, 3 * args.num_eval_samples))
        plt.axis("off")
        plt.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
        plt.savefig(ckpt_dir / f"qualitative_step{step}.png", bbox_inches="tight", dpi=150)
        plt.close()

    # 집계
    val_loss = loss_m.compute()
    psnr_val = psnr_m.aggregate().item()
    ssim_val = ssim_m.aggregate().item()
    mse_val  = mse_sum / max(n_seen, 1)

    if compute_fid and feats_fake:
        fid_val = fid_m(torch.vstack(feats_fake), torch.vstack(feats_real)).item()
    else:
        fid_val = float("nan")

    results = dict(step=step, val_loss=val_loss,
                   psnr=psnr_val, ssim=ssim_val, mse=mse_val, fid=fid_val)
    (ckpt_dir / f"eval_step{step}.json").write_text(json.dumps(results, indent=2))

    log_dict = {
        "eval/loss": val_loss,
        "eval/psnr": psnr_val,
        "eval/ssim": ssim_val,
        "eval/mse":  mse_val,
    }
    if compute_fid:
        log_dict["eval/fid"] = fid_val
    wandb.log(log_dict, step=step)

    fid_str = f"  FID={fid_val:.3f}" if compute_fid else "  FID=skip"
    print(f"[Eval {step}] loss={val_loss:.4f}  PSNR={psnr_val:.3f}  "
          f"SSIM={ssim_val:.4f}  MSE={mse_val:.6f}{fid_str}")

    vdm.train()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    ckpt_dir = pathlib.Path(args.checkpoint_dir) / args.exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    init_wandb(config=vars(args), project=args.wandb_project,
               experiment_name=args.exp_name, entity=args.wandb_entity)
    set_determinism(seed=args.seed)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.device)
        print(f"GPU: {torch.cuda.get_device_name(args.device)}")

    # ── 데이터 ───────────────────────────────────────────────────────────
    ss = (args.spatial_size, args.spatial_size)

    # 두 경로 모두 Task2/ 하위에 AB/HN/TH 구조
    data_roots = [
        f"{args.data_root}/synthRAD2025_Task2_Train/Task2",
        f"{args.data_root}/synthRAD2025_Task2_Train_D/Task2",
    ]

    train_ds, val_ds = split_dataset(
        root             = data_roots,
        modality         = ["cbct", "ct"],
        anatomy          = args.anatomy,
        n_slices         = args.in_channels,
        val_ratio        = args.val_ratio,
        seed             = args.seed,
        val_middle_only  = True,      # val은 케이스당 중간 슬라이스 1장
        train_transform  = build_transforms(["cbct", "ct"], spatial_size=ss, augment=True),
        val_transform    = build_transforms(["cbct", "ct"], spatial_size=ss, augment=False),
    )
    print(f"train: {len(train_ds)}슬라이스 ({len(train_ds.subject_dirs)}케이스) | "
          f"val: {len(val_ds)}슬라이스 ({len(val_ds.subject_dirs)}케이스)")

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
    train_iter = infinite_loader(train_loader)

    # ── VQVAE ────────────────────────────────────────────────────────────
    vqvae_kwargs = dict(out_channels=1, compress_ratio=args.compress_ratio,
                        embedding_dim=args.embedding_dim, num_embeddings=args.num_embeddings)
    ct_ae   = load_frozen_vqvae(args.ct_ckpt,   device, in_channels=args.in_channels, **vqvae_kwargs)
    cbct_ae = load_frozen_vqvae(args.cbct_ckpt, device, in_channels=args.in_channels, **vqvae_kwargs)

    with torch.no_grad():
        dummy        = torch.zeros(1, args.in_channels, args.spatial_size, args.spatial_size, device=device)
        latent_shape = tuple(ct_ae.encode_stage_2_inputs(dummy).shape[1:])
    print(f"latent shape: {latent_shape}")

    # ── Backbone ─────────────────────────────────────────────────────────
    if args.backbone == "uvit":
        backbone = UViT(
            img_size    = args.spatial_size // args.compress_ratio,
            patch_size  = args.uvit_patch_size,
            in_chans    = latent_shape[0],
            embed_dim   = args.uvit_embed_dim,
            depth       = args.uvit_depth,
            num_heads   = args.uvit_num_heads,
            conv        = True,
        ).to(device)
        backbone = torch.compile(backbone)
    else:
        COND_DIM = latent_shape[0] * latent_shape[1] * latent_shape[2]
        backbone = DiffusionModelUNet(
            spatial_dims     = 2,
            in_channels      = latent_shape[0],
            out_channels     = latent_shape[0],
            num_res_blocks   = 2,
            channels         = tuple(args.unet_channels),
            attention_levels = (True,) * len(args.unet_channels),
            norm_num_groups  = 8,
            num_head_channels= tuple(16 * 2**i for i in range(len(args.unet_channels))),
            with_conditioning= True,
            cross_attention_dim    = COND_DIM,
            transformer_num_layers = 1,
            use_flash_attention    = True,
        ).to(device)

    n_params = sum(p.numel() for p in backbone.parameters())
    print(f"[{args.backbone}] params: {n_params:,}")

    # ── VDM ──────────────────────────────────────────────────────────────
    vdm = VDM(
        model      = backbone,
        cfg        = SimpleNamespace(
                         noise_schedule          = args.noise_schedule,
                         gamma_min               = args.gamma_min,
                         gamma_max               = args.gamma_max,
                         antithetic_time_sampling= args.antithetic,
                     ),
        ae         = ct_ae,
        image_shape= latent_shape,
    ).to(device)

    inception = build_inception(device)

    # ── 옵티마이저 & 스케줄러 ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        backbone.parameters(), lr=args.lr,
        betas=(0.9, 0.99), weight_decay=args.weight_decay, eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _lr_lambda(args.warmup_steps, args.num_steps)
    )

    # ── 상태 변수 ─────────────────────────────────────────────────────────
    best_val_loss, global_step, eval_count = float("inf"), 0, 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        vdm.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        global_step   = ckpt.get("step", 0)
        print(f"[resume] step {global_step} | best_val_loss={best_val_loss:.4f}")

    # ── 학습 루프 ─────────────────────────────────────────────────────────
    LOG_KEYS = ["bpd", "diff_loss", "latent_loss", "recon_loss"]
    log_m = {k: Mean(device=device) for k in LOG_KEYS}
    vdm.train()

    pbar = tqdm(range(global_step, args.num_steps), desc="Training",
                initial=global_step, total=args.num_steps)

    for _ in pbar:
        batch    = next(train_iter)
        ct_img   = batch["ct"].to(device)    # (B, n_slices, H, W)
        cbct_img = batch["cbct"].to(device)  # (B, n_slices, H, W)

        with torch.no_grad():
            z      = ct_ae.encode_stage_2_inputs(ct_img)
            z_cond = cbct_ae.encode_stage_2_inputs(cbct_img)
        cond = _prepare_cond(z_cond, args.backbone)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=args.amp, dtype=torch.bfloat16):
            loss, metrics = vdm(z, cond, ct_img)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        global_step += 1

        for k in LOG_KEYS:
            if k in metrics:
                log_m[k].update(metrics[k])
        pbar.set_postfix(loss=f"{metrics['bpd'].item():.4f}",
                         lr=f"{get_lr(optimizer):.2e}")

        # train 로그
        if global_step % args.log_every == 0:
            train_vals = {k: v.compute() for k, v in log_m.items()}
            train_vals["lr"] = get_lr(optimizer)
            log_train(train_vals, global_step)
            for v in log_m.values():
                v.reset()

        # eval
        if global_step % args.eval_every == 0:
            eval_count += 1
            result = evaluate(
                vdm, ct_ae, cbct_ae, val_loader, args,
                device, global_step, ckpt_dir, inception, eval_count,
            )
            val_loss = result["val_loss"]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_ckpt(ckpt_dir / "best.pt", global_step, vdm,
                          optimizer, scheduler, best_val_loss)
                print(f"[best] step {global_step}  val_loss={best_val_loss:.4f}")

            save_ckpt(ckpt_dir / "latest.pt", global_step, vdm,
                      optimizer, scheduler, best_val_loss)

    # 마지막 eval
    if global_step % args.eval_every != 0:
        eval_count += 1
        evaluate(vdm, ct_ae, cbct_ae, val_loader, args,
                 device, global_step, ckpt_dir, inception, eval_count)

    finish()


if __name__ == "__main__":
    main()