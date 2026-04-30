"""eval_vqvae.py — VQ-VAE 표현 품질 평가.

체크포인트 자동 탐색 후 재구성/코드북 지표를 측정해 CSV로 저장.

사용법:
    python eval_vqvae.py --preprocessed_root /data/prof2/mai/s2025/dataset/preprocessed
    python eval_vqvae.py --preprocessed_root /data/prof2/mai/s2025/dataset/preprocessed --device 1
"""
from __future__ import annotations

import argparse
import pathlib
import re

import pandas as pd
import torch
import torch.nn.functional as F
from monai.networks.nets import VQVAE
from monai.utils import set_determinism
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.preprocessed_dataset import PreprocessedDataset

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

CKPT_BASE   = pathlib.Path("checkpoints/stage1_vqvae")
CKPT_PATTERN = re.compile(
    r"full_vqvae_(?P<modality>cbct|ct)_n(?P<n>\d+)_cpr(?P<cpr>\d+)_img\d+"
)
ANATOMY       = ["AB", "HN", "TH"]
NUM_EMBEDDINGS = 2048
EMBEDDING_DIM  = 1
CHANNELS       = (128, 256, 512, 512)

CPR_CFG = {
    2: (
        ((2,4,1,1),(1,3,1,1),(1,3,1,1),(1,3,1,1)),
        ((1,3,1,1,0),(1,3,1,1,0),(1,3,1,1,0),(2,4,1,1,0)),
    ),
    4: (
        ((2,4,1,1),(2,4,1,1),(1,3,1,1),(1,3,1,1)),
        ((1,3,1,1,0),(1,3,1,1,0),(2,4,1,1,0),(2,4,1,1,0)),
    ),
    8: (
        ((2,4,1,1),(2,4,1,1),(2,4,1,1),(1,3,1,1)),
        ((1,3,1,1,0),(2,4,1,1,0),(2,4,1,1,0),(2,4,1,1,0)),
    ),
}


# ---------------------------------------------------------------------------
# 지표 계산 유틸
# ---------------------------------------------------------------------------

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred.float(), target.float()).item()
    if mse < 1e-12:
        return 100.0
    import math
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """배치 평균 SSIM. 입력: (B, 1, H, W), 값 범위 [0, 1]."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pred, target = pred.float(), target.float()

    mu_p = F.avg_pool2d(pred,   kernel_size=11, stride=1, padding=5)
    mu_t = F.avg_pool2d(target, kernel_size=11, stride=1, padding=5)

    mu_p2  = mu_p * mu_p
    mu_t2  = mu_t * mu_t
    mu_pt  = mu_p * mu_t

    sigma_p2  = F.avg_pool2d(pred * pred,     11, 1, 5) - mu_p2
    sigma_t2  = F.avg_pool2d(target * target, 11, 1, 5) - mu_t2
    sigma_pt  = F.avg_pool2d(pred * target,   11, 1, 5) - mu_pt

    ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / \
               ((mu_p2 + mu_t2 + C1) * (sigma_p2 + sigma_t2 + C2))
    return ssim_map.mean().item()


def get_encoding_indices(
    model: VQVAE,
    x: torch.Tensor,
) -> torch.Tensor:
    """인코더 출력에서 가장 가까운 코드북 인덱스를 반환."""
    with torch.no_grad():
        z_e = model.encoder(x)                       # (B, emb_dim, H', W')
        emb = model.quantizer.embedding.weight        # (K, emb_dim)
        flat = z_e.permute(0, 2, 3, 1).reshape(-1, emb.shape[1])  # (N, emb_dim)
        d = (
            flat.pow(2).sum(1, keepdim=True)
            + emb.pow(2).sum(1)
            - 2 * flat @ emb.T
        )
        return d.argmin(dim=1)                        # (N,)


# ---------------------------------------------------------------------------
# 모델 빌더
# ---------------------------------------------------------------------------

def build_model(in_channels: int, compress_ratio: int) -> VQVAE:
    down, up = CPR_CFG[compress_ratio]
    return VQVAE(
        spatial_dims          = 2,
        in_channels           = in_channels,
        out_channels          = 1,
        channels              = CHANNELS,
        num_res_channels      = 256,
        num_res_layers        = 2,
        downsample_parameters = down,
        upsample_parameters   = up,
        num_embeddings        = NUM_EMBEDDINGS,
        embedding_dim         = EMBEDDING_DIM,
        commitment_cost       = 0.4,
    )


# ---------------------------------------------------------------------------
# 체크포인트 탐색
# ---------------------------------------------------------------------------

def discover_checkpoints() -> list[dict]:
    configs = []
    for folder in sorted(CKPT_BASE.iterdir()):
        m = CKPT_PATTERN.match(folder.name)
        if not m:
            continue
        ckpt_path = folder / "best.pt"
        if not ckpt_path.exists():
            continue
        configs.append({
            "name"     : folder.name,
            "modality" : m.group("modality"),
            "n"        : int(m.group("n")),
            "cpr"      : int(m.group("cpr")),
            "ckpt"     : ckpt_path,
        })
    return configs


# ---------------------------------------------------------------------------
# 단일 모델 평가
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    cfg: dict,
    preprocessed_root: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict:
    modality = cfg["modality"]
    n        = cfg["n"]
    cpr      = cfg["cpr"]

    # ── 데이터 ──────────────────────────────────────────────────────────────
    val_ds = PreprocessedDataset(
        preprocessed_root = preprocessed_root,
        split    = "val",
        n_slices = n,
        modality = [modality],
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )

    # ── 모델 로드 ────────────────────────────────────────────────────────────
    model = build_model(n, cpr).to(device)
    state = torch.load(cfg["ckpt"], map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    mid = n // 2  # 중앙 슬라이스 인덱스

    # ── 평가 루프 ────────────────────────────────────────────────────────────
    l1_sum = ssim_sum = psnr_sum = perp_sum = 0.0
    used_codes: set[int] = set()
    n_batches = 0

    for batch in tqdm(val_loader, desc=cfg["name"], leave=False):
        data   = batch[modality].to(device)           # (B, N, H, W)
        target = data[:, mid:mid+1].float()            # (B, 1, H, W)

        recon, _ = model(images=data)
        recon    = recon.float().clamp(0, 1)
        target   = target.clamp(0, 1)

        l1_sum   += F.l1_loss(recon, target).item()
        psnr_sum += compute_psnr(recon, target)
        ssim_sum += compute_ssim(recon, target)
        perp_sum += model.quantizer.perplexity.item()

        indices = get_encoding_indices(model, data)
        used_codes.update(indices.cpu().tolist())

        n_batches += 1

    usage_rate = len(used_codes) / NUM_EMBEDDINGS * 100

    return {
        "name"            : cfg["name"],
        "modality"        : modality,
        "n"               : n,
        "cpr"             : cpr,
        "epoch"           : state.get("epoch", -1),
        "L1"              : l1_sum   / n_batches,
        "PSNR"            : psnr_sum / n_batches,
        "SSIM"            : ssim_sum / n_batches,
        "Perplexity"      : perp_sum / n_batches,
        "Codebook_Usage%" : usage_rate,
        "Used_Codes"      : len(used_codes),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="VQ-VAE 품질 평가")
    p.add_argument("--preprocessed_root", type=str,
                   default="/data/prof2/mai/s2025/dataset/preprocessed")
    p.add_argument("--device",      type=int, default=0)
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--output",      type=str, default="eval_results.csv")
    return p.parse_args()


def main():
    args   = get_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    set_determinism(seed=args.seed)

    configs = discover_checkpoints()
    if not configs:
        print(f"[오류] {CKPT_BASE} 에서 체크포인트를 찾을 수 없습니다.")
        return
    print(f"[평가 대상] {len(configs)}개 체크포인트")

    rows = []
    for cfg in configs:
        print(f"\n>>> {cfg['name']}")
        try:
            row = evaluate_model(
                cfg, args.preprocessed_root, device,
                args.batch_size, args.num_workers,
            )
            rows.append(row)
            print(
                f"    L1={row['L1']:.4f}  PSNR={row['PSNR']:.2f}dB"
                f"  SSIM={row['SSIM']:.4f}  Perp={row['Perplexity']:.1f}"
                f"  Usage={row['Codebook_Usage%']:.1f}%"
            )
        except Exception as e:
            print(f"    [건너뜀] {e}")

    if not rows:
        print("평가 결과가 없습니다.")
        return

    df = pd.DataFrame(rows).sort_values(["modality", "n", "cpr"])
    df.to_csv(args.output, index=False)
    print(f"\n[저장] {args.output}")

    # ── 비교표 출력 ──────────────────────────────────────────────────────────
    for mod in ["ct", "cbct"]:
        sub = df[df["modality"] == mod]
        if sub.empty:
            continue
        print(f"\n{'='*70}")
        print(f"  {mod.upper()} — compress_ratio 비교 (n=5 고정)")
        print(f"{'='*70}")
        view = sub[sub["n"] == 5][["cpr","L1","PSNR","SSIM","Perplexity","Codebook_Usage%"]]
        if not view.empty:
            print(view.to_string(index=False))

        print(f"\n  {mod.upper()} — in_channels(n) 비교 (cpr=4 고정)")
        print(f"{'-'*70}")
        view = sub[sub["cpr"] == 4][["n","L1","PSNR","SSIM","Perplexity","Codebook_Usage%"]]
        if not view.empty:
            print(view.to_string(index=False))


if __name__ == "__main__":
    main()
