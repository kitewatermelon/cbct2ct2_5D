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
