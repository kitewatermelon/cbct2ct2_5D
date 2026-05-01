# CycleGAN Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** CBCT→CT ablation 비교용 CycleGAN 2D baseline을 구현하고 기존 eval_full.py에 통합한다.

**Architecture:** MONAI ResidualUnit 기반 ResNet Generator (9 ResBlock) + MONAI PatchDiscriminator (70×70 PatchGAN) 2쌍. SynthRad2025 n=1 데이터(2D 단일 슬라이스)로 학습. eval_full.py의 gen_dir 방식으로 메트릭 통합.

**Tech Stack:** PyTorch, MONAI (ResidualUnit, PatchDiscriminator), SynthRad2025 n=1 데이터, WandB, SimpleITK

---

## 파일 구조

| 파일 | 역할 |
|------|------|
| `models/cyclegan/__init__.py` | 패키지 초기화 |
| `models/cyclegan/generator.py` | `ResNetGenerator` 클래스 |
| `models/cyclegan/discriminator.py` | `build_discriminator()` 팩토리 |
| `train/cyclegan.py` | 전체 학습 루프 |
| `eval/eval_gen_cyclegan.py` | 학습된 G_A2B로 val .mha 생성 |
| `eval/eval_full.py` | MODEL_CONFIGS에 cyclegan 추가, main() 분기 |
| `tests/test_cyclegan_models.py` | Generator/Discriminator shape 테스트 |
| `tests/test_eval_full.py` | MODEL_CONFIGS count 업데이트 |
| `runs/run_cyclegan.sh` | 학습 실행 스크립트 |

---

## Task 1: Generator 모델

**Files:**
- Create: `models/cyclegan/__init__.py`
- Create: `models/cyclegan/generator.py`
- Create: `tests/test_cyclegan_models.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_cyclegan_models.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import torch
from models.cyclegan.generator import ResNetGenerator

def test_generator_output_shape():
    G = ResNetGenerator(in_channels=1, out_channels=1, ngf=64, n_blocks=9)
    x = torch.randn(2, 1, 128, 128)
    out = G(x)
    assert out.shape == (2, 1, 128, 128), f"expected (2,1,128,128), got {out.shape}"

def test_generator_output_range():
    G = ResNetGenerator(in_channels=1, out_channels=1, ngf=64, n_blocks=9)
    x = torch.randn(2, 1, 128, 128)
    out = G(x)
    assert out.min() >= -1.0 - 1e-5 and out.max() <= 1.0 + 1e-5, \
        f"tanh output out of range: [{out.min():.4f}, {out.max():.4f}]"
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd /home/dministrator/cbct2ct2_5D
python -m pytest tests/test_cyclegan_models.py::test_generator_output_shape -v
```
Expected: `ImportError` 또는 `ModuleNotFoundError`

- [ ] **Step 3: `__init__.py` 생성**

```python
# models/cyclegan/__init__.py
```
(빈 파일)

- [ ] **Step 4: Generator 구현**

```python
# models/cyclegan/generator.py
"""ResNet Generator for CycleGAN (2D, single-channel)."""
from __future__ import annotations
import torch
import torch.nn as nn
from monai.networks.blocks import ResidualUnit


class ResNetGenerator(nn.Module):
    """CBCT↔CT 변환용 ResNet Generator.

    구조: Conv↓(×2) → ResBlock(×n_blocks) → ConvTranspose↑(×2) → tanh
    입력/출력: (B, in_channels, H, W), 값 범위 [-1, 1]
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        ngf: int = 64,
        n_blocks: int = 9,
    ) -> None:
        super().__init__()

        # Encoder
        enc: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, 7, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(True),
        ]

        # ResBlocks (MONAI ResidualUnit, in==out이므로 shortcut = identity)
        res: list[nn.Module] = [
            ResidualUnit(
                spatial_dims=2,
                in_channels=ngf * 4,
                out_channels=ngf * 4,
                kernel_size=3,
                act="RELU",
                norm="INSTANCE",
                bias=False,
            )
            for _ in range(n_blocks)
        ]

        # Decoder
        dec: list[nn.Module] = [
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, 7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*enc, *res, *dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
```

- [ ] **Step 5: 테스트 통과 확인**

```bash
python -m pytest tests/test_cyclegan_models.py::test_generator_output_shape tests/test_cyclegan_models.py::test_generator_output_range -v
```
Expected: 2 passed

- [ ] **Step 6: 커밋**

```bash
git add models/cyclegan/__init__.py models/cyclegan/generator.py tests/test_cyclegan_models.py
git commit -m "feat: add CycleGAN ResNet generator with MONAI ResidualUnit"
```

---

## Task 2: Discriminator 모델

**Files:**
- Create: `models/cyclegan/discriminator.py`
- Modify: `tests/test_cyclegan_models.py`

- [ ] **Step 1: 테스트 추가**

`tests/test_cyclegan_models.py` 맨 아래에 추가:

```python
from models.cyclegan.discriminator import build_discriminator

def test_discriminator_output_shape():
    D = build_discriminator(in_channels=1)
    x = torch.randn(2, 1, 128, 128)
    out = D(x)
    # PatchDiscriminator는 리스트 반환 — 마지막 요소가 patch map
    patch = out[-1]
    assert patch.shape[0] == 2
    assert patch.shape[1] == 1
    assert patch.ndim == 4, f"expected 4D patch output, got {patch.shape}"

def test_discriminator_patch_is_smaller_than_input():
    D = build_discriminator(in_channels=1)
    x = torch.randn(2, 1, 128, 128)
    patch = D(x)[-1]
    assert patch.shape[2] < 128 and patch.shape[3] < 128
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
python -m pytest tests/test_cyclegan_models.py::test_discriminator_output_shape -v
```
Expected: `ImportError`

- [ ] **Step 3: Discriminator 구현**

```python
# models/cyclegan/discriminator.py
"""PatchDiscriminator wrapper for CycleGAN."""
from __future__ import annotations
import torch.nn as nn
from monai.networks.nets import PatchDiscriminator


def build_discriminator(in_channels: int = 1) -> PatchDiscriminator:
    """70×70 PatchGAN discriminator (MONAI 구현).

    반환값: PatchDiscriminator 인스턴스.
    forward() 반환: list[Tensor] — [-1] 요소가 패치 맵 (B, 1, H', W')
    """
    return PatchDiscriminator(
        in_channels=in_channels,
        num_layers_d=3,
        spatial_dims=2,
        channels=64,
        out_channels=1,
    )
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
python -m pytest tests/test_cyclegan_models.py -v
```
Expected: 4 passed

- [ ] **Step 5: 커밋**

```bash
git add models/cyclegan/discriminator.py tests/test_cyclegan_models.py
git commit -m "feat: add MONAI PatchDiscriminator wrapper for CycleGAN"
```

---

## Task 3: 학습 스크립트 (`train/cyclegan.py`)

**Files:**
- Create: `train/cyclegan.py`

- [ ] **Step 1: smoke test 작성**

`tests/test_cyclegan_models.py` 맨 아래에 추가:

```python
def test_one_training_step():
    """학습 1 step이 에러 없이 실행되고 loss가 scalar인지 확인."""
    import torch.nn.functional as F
    from models.cyclegan.generator import ResNetGenerator
    from models.cyclegan.discriminator import build_discriminator

    device = torch.device("cpu")
    G_A2B = ResNetGenerator().to(device)
    G_B2A = ResNetGenerator().to(device)
    D_A   = build_discriminator().to(device)
    D_B   = build_discriminator().to(device)

    opt_G = torch.optim.Adam(
        list(G_A2B.parameters()) + list(G_B2A.parameters()),
        lr=2e-4, betas=(0.5, 0.999)
    )
    opt_D = torch.optim.Adam(
        list(D_A.parameters()) + list(D_B.parameters()),
        lr=2e-4, betas=(0.5, 0.999)
    )

    real_A = torch.randn(2, 1, 128, 128, device=device)  # CBCT
    real_B = torch.randn(2, 1, 128, 128, device=device)  # CT

    # Generator step
    opt_G.zero_grad()
    fake_B = G_A2B(real_A)
    fake_A = G_B2A(real_B)
    loss_adv = (
        F.mse_loss(D_B(fake_B)[-1], torch.ones_like(D_B(fake_B)[-1])) +
        F.mse_loss(D_A(fake_A)[-1], torch.ones_like(D_A(fake_A)[-1]))
    )
    rec_A = G_B2A(fake_B)
    rec_B = G_A2B(fake_A)
    loss_cycle = F.l1_loss(rec_A, real_A) + F.l1_loss(rec_B, real_B)
    loss_G = loss_adv + 10.0 * loss_cycle
    loss_G.backward()
    opt_G.step()

    # Discriminator step
    opt_D.zero_grad()
    loss_D = (
        0.5 * (F.mse_loss(D_B(real_B.detach())[-1], torch.ones(2,1,14,14)) +
               F.mse_loss(D_B(fake_B.detach())[-1], torch.zeros(2,1,14,14))) +
        0.5 * (F.mse_loss(D_A(real_A.detach())[-1], torch.ones(2,1,14,14)) +
               F.mse_loss(D_A(fake_A.detach())[-1], torch.zeros(2,1,14,14)))
    )
    loss_D.backward()
    opt_D.step()

    assert loss_G.item() > 0
    assert loss_D.item() > 0
```

- [ ] **Step 2: 테스트 실패 확인 (smoke test는 모델만 필요하므로 이미 통과 가능)**

```bash
python -m pytest tests/test_cyclegan_models.py::test_one_training_step -v
```
Expected: PASSED (모델은 이미 구현됨)

- [ ] **Step 3: 학습 스크립트 구현**

```python
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
from utils.wandb import init_wandb, log_train, log_val, log_images, finish


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
        lr=args.lr, betas=(0.5, 0.999),
    )
    opt_D = torch.optim.Adam(
        list(D_A.parameters()) + list(D_B.parameters()),
        lr=args.lr, betas=(0.5, 0.999),
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

    # 최종 체크포인트
    torch.save({
        "epoch":            args.n_epochs - 1,
        "G_A2B_state_dict": G_A2B.state_dict(),
        "G_B2A_state_dict": G_B2A.state_dict(),
        "args":             vars(args),
    }, ckpt_dir / "latest.pt")

    finish()
    print(f"학습 완료. Best PSNR: {best_psnr:.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: smoke test 통과 확인**

```bash
python -m pytest tests/test_cyclegan_models.py -v
```
Expected: 5 passed

- [ ] **Step 5: 커밋**

```bash
git add train/cyclegan.py tests/test_cyclegan_models.py
git commit -m "feat: add CycleGAN training script (2D, CBCT↔CT)"
```

---

## Task 4: Eval gen 스크립트 (`eval/eval_gen_cyclegan.py`)

**Files:**
- Create: `eval/eval_gen_cyclegan.py`

학습 완료 후 val 집합의 모든 subject에 대해 G_A2B 추론 결과를 `.mha`로 저장. eval_full.py의 `evaluate_from_gen`이 읽는 포맷과 동일하게 저장.

- [ ] **Step 1: 스크립트 구현**

```python
# eval/eval_gen_cyclegan.py
"""eval_gen_cyclegan.py — 학습된 CycleGAN으로 val .mha 생성.

출력 구조:
  {gen_dir}/cyclegan/{subj_id}.mha   # (H, W) float32
  {gen_dir}/cyclegan/meta.json       # {subj_id: anatomy}
"""
from __future__ import annotations

import argparse
import json
import pathlib

import torch
from torch.utils.data import DataLoader, random_split
from monai.utils import set_determinism
from tqdm.auto import tqdm

from data.synthrad2025 import SynthRad2025, build_transforms
from models.cyclegan.generator import ResNetGenerator
from utils.mha import save_mha


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CycleGAN val 집합 .mha 생성")
    p.add_argument("--ckpt",       type=str, required=True,
                   help="학습 체크포인트 경로 (best.pt)")
    p.add_argument("--gen_dir",    type=str, default="gen_outputs",
                   help="생성 결과 저장 디렉토리")
    p.add_argument("--data_root",  type=str, default="/home/dministrator/s2025")
    p.add_argument("--anatomy",    nargs="+", default=["AB", "HN", "TH"])
    p.add_argument("--spatial_size", type=int, default=128)
    p.add_argument("--val_ratio",  type=float, default=0.2)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--num_workers", type=int,  default=4)
    p.add_argument("--device",     type=int,   default=0)
    return p.parse_args()


def main() -> None:
    args   = get_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    set_determinism(seed=args.seed)

    # ── 체크포인트 로드 ──────────────────────────────────────────────────────
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    ngf      = saved_args.get("ngf", 64)
    n_blocks = saved_args.get("n_blocks", 9)

    G_A2B = ResNetGenerator(ngf=ngf, n_blocks=n_blocks).to(device)
    G_A2B.load_state_dict(ckpt["G_A2B_state_dict"])
    G_A2B.eval()

    # ── Val split (학습과 동일한 seed/ratio) ─────────────────────────────────
    data_root = f"{args.data_root}/dataset/train/n1"
    tf = build_transforms(["cbct", "ct"], (args.spatial_size, args.spatial_size), augment=False)
    full_ds = SynthRad2025(
        root=data_root, modality=["cbct", "ct"],
        anatomy=args.anatomy, transform=tf,
    )
    n_val   = int(len(full_ds) * args.val_ratio)
    n_train = len(full_ds) - n_val
    _, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── subj_id → anatomy 매핑 ───────────────────────────────────────────────
    subj_anatomy: dict[str, str] = {
        d.name: d.parent.name for d in full_ds.subject_dirs
    }

    # ── 생성 및 저장 ─────────────────────────────────────────────────────────
    out_dir = pathlib.Path(args.gen_dir) / "cyclegan"
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="CycleGAN gen"):
            cbct = batch["cbct"].to(device)    # (B, 1, H, W)  [0, 1]
            real_A = cbct * 2.0 - 1.0          # [-1, 1]
            fake_B = G_A2B(real_A)
            ct_gen = ((fake_B + 1.0) / 2.0).clamp(0, 1)  # [0, 1]

            for i, sid in enumerate(batch["subj_id"]):
                arr = ct_gen[i, 0].cpu().numpy()  # (H, W)
                save_mha(arr, out_dir / f"{sid}.mha")

    # meta.json 저장
    val_subj_ids = [full_ds.subject_dirs[i].name for i in val_ds.indices]
    meta = {sid: subj_anatomy.get(sid, "UNKNOWN") for sid in val_subj_ids}
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"생성 완료: {len(val_subj_ids)}개 subject → {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: import 확인**

```bash
python -c "import eval.eval_gen_cyclegan" 2>&1
```
Expected: 에러 없음

- [ ] **Step 3: 커밋**

```bash
git add eval/eval_gen_cyclegan.py
git commit -m "feat: add CycleGAN val inference script (eval_gen_cyclegan.py)"
```

---

## Task 5: `eval_full.py` 통합

**Files:**
- Modify: `eval/eval_full.py`
- Modify: `tests/test_eval_full.py`

CycleGAN을 MODEL_CONFIGS에 추가하고, gen_dir 없이 실행할 경우 자동 스킵.

- [ ] **Step 1: 테스트 수정**

`tests/test_eval_full.py`에서 두 테스트를 수정:

```python
def test_model_configs_count():
    assert len(MODEL_CONFIGS) == 9   # 8 → 9

def test_model_configs_keys():
    keys = {c["key"] for c in MODEL_CONFIGS}
    assert keys == {
        "uvit_n1_cpr4", "uvit_n3_cpr4", "uvit_n5_cpr4",
        "uvit_n7_cpr4", "uvit_n9_cpr4",
        "uvit_n5_cpr2", "uvit_n5_cpr8",
        "unet_n5_cpr4",
        "cyclegan",
    }
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
python -m pytest tests/test_eval_full.py::test_model_configs_count tests/test_eval_full.py::test_model_configs_keys -v
```
Expected: 2 FAILED

- [ ] **Step 3: MODEL_CONFIGS에 cyclegan 추가**

`eval/eval_full.py`의 `MODEL_CONFIGS` 리스트 마지막에 추가:

```python
MODEL_CONFIGS: list[dict] = [
    {"key": "uvit_n1_cpr4", "backbone": "uvit", "n": 1, "cpr": 4},
    {"key": "uvit_n3_cpr4", "backbone": "uvit", "n": 3, "cpr": 4},
    {"key": "uvit_n5_cpr4", "backbone": "uvit", "n": 5, "cpr": 4},
    {"key": "uvit_n7_cpr4", "backbone": "uvit", "n": 7, "cpr": 4},
    {"key": "uvit_n9_cpr4", "backbone": "uvit", "n": 9, "cpr": 4},
    {"key": "uvit_n5_cpr2", "backbone": "uvit", "n": 5, "cpr": 2},
    {"key": "uvit_n5_cpr8", "backbone": "uvit", "n": 5, "cpr": 8},
    {"key": "unet_n5_cpr4", "backbone": "unet", "n": 5, "cpr": 4},
    {"key": "cyclegan",     "type": "cyclegan", "n": 1},       # <-- 추가
]
```

- [ ] **Step 4: `main()`에 cyclegan 스킵 분기 추가**

`eval/eval_full.py`의 `main()` 함수 내 `for cfg in MODEL_CONFIGS:` 루프 안, `try:` 블록 시작 직전에 추가:

```python
    for cfg in MODEL_CONFIGS:
        print(f"\n>>> {cfg['key']}")
        # CycleGAN은 gen_dir 방식만 지원 (VQ-VAE 불필요)
        if cfg.get("type") == "cyclegan" and not use_gen:
            print("    [스킵] CycleGAN은 --gen_dir 필요 (eval_gen_cyclegan.py 먼저 실행)")
            continue
        try:
            ...
```

- [ ] **Step 5: 테스트 통과 확인**

```bash
python -m pytest tests/test_eval_full.py::test_model_configs_count tests/test_eval_full.py::test_model_configs_keys -v
```
Expected: 2 passed

- [ ] **Step 6: 기존 테스트도 통과 확인**

```bash
python -m pytest tests/test_eval_full.py -v -k "not val_split_ratio and not val_split_reproducible and not load_model and not evaluate_model"
```
Expected: 4 passed (count, keys, compute_*, save_*)

- [ ] **Step 7: 커밋**

```bash
git add eval/eval_full.py tests/test_eval_full.py
git commit -m "feat: add cyclegan to MODEL_CONFIGS in eval_full.py"
```

---

## Task 6: 실행 스크립트 및 최종 커밋

**Files:**
- Create: `runs/run_cyclegan.sh`

- [ ] **Step 1: 실행 스크립트 작성**

```bash
# runs/run_cyclegan.sh
#!/usr/bin/env bash
set -e

python train/cyclegan.py \
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
```

- [ ] **Step 2: 실행 스크립트에 실행 권한 부여**

```bash
chmod +x runs/run_cyclegan.sh
```

- [ ] **Step 3: 전체 테스트 최종 확인**

```bash
python -m pytest tests/test_cyclegan_models.py tests/test_eval_full.py -v \
  -k "not val_split_ratio and not val_split_reproducible and not load_model and not evaluate_model"
```
Expected: 9 passed

- [ ] **Step 4: 최종 커밋**

```bash
git add runs/run_cyclegan.sh
git commit -m "feat: add run_cyclegan.sh execution script"
```

---

## 실행 순서 (학습 후 평가)

```bash
# 1. 학습
bash runs/run_cyclegan.sh

# 2. Val .mha 생성
python eval/eval_gen_cyclegan.py \
  --ckpt checkpoints/cyclegan/best.pt \
  --gen_dir gen_outputs \
  --data_root /home/dministrator/s2025 \
  --device 0

# 3. 전체 ablation 평가 (기존 VDM + CycleGAN)
python eval/eval_full.py \
  --gen_dir gen_outputs \
  --data_root /home/dministrator/s2025 \
  --device 0
```
