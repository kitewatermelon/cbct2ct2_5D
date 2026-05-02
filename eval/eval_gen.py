"""eval_gen.py — Stage2 VDM + CycleGAN inference-only: 생성 결과를 .mha로 저장.

Usage:
    python eval_gen.py [--keys uvit_n5_cpr4 unet_n5_cpr4 cyclegan] [--overwrite]

출력:
    {output_dir}/{model_key}/{subj_id}.mha  — generated CT (float32, [0,1] normalized)
    {output_dir}/{model_key}/meta.json      — {"subj_id": "anatomy", ...}
"""
from __future__ import annotations
import argparse, json, pathlib
import torch
from torch.utils.data import DataLoader, random_split
from monai.utils import set_determinism
from tqdm.auto import tqdm

from eval.eval_full import MODEL_CONFIGS, load_model_for_eval
from train.stage2_vdm import _prepare_cond, sample_conditional
from data.synthrad2025 import SynthRad2025, build_transforms
from models.cyclegan.generator import ResNetGenerator
from utils.mha import save_mha


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_val_subject_ids(
    data_root: str, n: int, val_ratio: float, seed: int,
) -> list[str]:
    """val split의 subject id 목록 반환."""
    full_ds = SynthRad2025(
        root=f"{data_root}/dataset/train/n{n}",
        modality=["cbct", "ct"],
        anatomy=["AB", "HN", "TH"],
        transform=build_transforms(["cbct", "ct"], spatial_size=(128, 128), augment=False),
    )
    n_val = int(len(full_ds) * val_ratio)
    n_train = len(full_ds) - n_val
    _, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    return [full_ds.subject_dirs[i].name for i in val_ds.indices]


def all_files_exist(model_dir: pathlib.Path, subject_ids: list[str]) -> bool:
    return all((model_dir / f"{sid}.mha").exists() for sid in subject_ids)


def missing_subjects(model_dir: pathlib.Path, subject_ids: list[str]) -> set[str]:
    return {sid for sid in subject_ids if not (model_dir / f"{sid}.mha").exists()}


# ---------------------------------------------------------------------------
# Per-model generation — VDM
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_model(
    cfg: dict,
    loaded: dict,
    device: torch.device,
    out_dir: pathlib.Path,
    n_sample_steps: int = 200,
    overwrite: bool = False,
) -> None:
    ct_ae        = loaded["ct_ae"]
    cbct_ae      = loaded["cbct_ae"]
    vdm          = loaded["vdm"]
    scale_factor = loaded["scale_factor"]
    val_loader   = loaded["val_loader"]
    subj_anatomy = loaded["subj_anatomy"]
    backbone     = cfg["backbone"]
    key          = cfg["key"]

    model_dir = out_dir / key
    model_dir.mkdir(parents=True, exist_ok=True)

    meta: dict[str, str] = {}
    first_batch = True

    for batch in tqdm(val_loader, desc=key, leave=False):
        cbct_img = batch["cbct"].to(device)
        subj_ids = batch["subj_id"]

        # 배치 내 모든 subject가 이미 존재하면 스킵
        if not overwrite and all((model_dir / f"{sid}.mha").exists() for sid in subj_ids):
            for sid in subj_ids:
                meta[sid] = subj_anatomy.get(sid, "UNKNOWN")
            continue

        z_cond    = cbct_ae.encode_stage_2_inputs(cbct_img) * scale_factor
        cond      = _prepare_cond(z_cond, backbone)
        sampled_z = sample_conditional(vdm, cond, n_sample_steps, device)
        ct_gen    = ct_ae.decode_stage_2_outputs(sampled_z / scale_factor).float()

        if first_batch:
            print(f"  [range] ct_gen: [{ct_gen.min():.4f}, {ct_gen.max():.4f}]")
            first_batch = False

        for i in range(ct_gen.shape[0]):
            sid  = subj_ids[i]
            path = model_dir / f"{sid}.mha"
            if not overwrite and path.exists():
                meta[sid] = subj_anatomy.get(sid, "UNKNOWN")
                continue
            arr = ct_gen[i, 0].cpu().numpy()
            save_mha(arr, path)
            meta[sid] = subj_anatomy.get(sid, "UNKNOWN")

    meta_path = model_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  [저장] {model_dir}/ ({len(meta)}개 샘플)")


# ---------------------------------------------------------------------------
# Per-model generation — CycleGAN
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_cyclegan(
    cyclegan_ckpt: str,
    data_root: str,
    device: torch.device,
    out_dir: pathlib.Path,
    val_ratio: float = 0.2,
    seed: int = 42,
    batch_size: int = 8,
    num_workers: int = 4,
    overwrite: bool = False,
) -> None:
    key = "cyclegan"
    model_dir = out_dir / key
    model_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(cyclegan_ckpt, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    ngf      = saved_args.get("ngf", 64)
    n_blocks = saved_args.get("n_blocks", 9)

    G_A2B = ResNetGenerator(ngf=ngf, n_blocks=n_blocks).to(device)
    G_A2B.load_state_dict(ckpt["G_A2B_state_dict"])
    G_A2B.eval()

    data_path = f"{data_root}/dataset/train/n1"
    tf = build_transforms(["cbct", "ct"], (128, 128), augment=False)
    full_ds = SynthRad2025(
        root=data_path, modality=["cbct", "ct"],
        anatomy=["AB", "HN", "TH"], transform=tf,
    )
    n_val   = int(len(full_ds) * val_ratio)
    n_train = len(full_ds) - n_val
    _, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    subj_anatomy: dict[str, str] = {
        d.name: d.parent.name for d in full_ds.subject_dirs
    }

    meta: dict[str, str] = {}
    first_batch = True

    for batch in tqdm(val_loader, desc=key, leave=False):
        subj_ids = list(batch["subj_id"])

        if not overwrite and all((model_dir / f"{sid}.mha").exists() for sid in subj_ids):
            for sid in subj_ids:
                meta[sid] = subj_anatomy.get(sid, "UNKNOWN")
            continue

        cbct   = batch["cbct"].to(device)
        real_A = cbct * 2.0 - 1.0
        fake_B = G_A2B(real_A)
        ct_gen = ((fake_B + 1.0) / 2.0).clamp(0, 1).float()

        if first_batch:
            print(f"  [range] ct_gen: [{ct_gen.min():.4f}, {ct_gen.max():.4f}]")
            first_batch = False

        for i, sid in enumerate(subj_ids):
            path = model_dir / f"{sid}.mha"
            if not overwrite and path.exists():
                meta[sid] = subj_anatomy.get(sid, "UNKNOWN")
                continue
            arr = ct_gen[i, 0].cpu().numpy()
            save_mha(arr, path)
            meta[sid] = subj_anatomy.get(sid, "UNKNOWN")

    with open(model_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  [저장] {model_dir}/ ({len(meta)}개 샘플)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="VDM + CycleGAN inference-only: .mha 저장")
    p.add_argument("--data_root",       type=str,   default="/home/dministrator/s2025")
    p.add_argument("--ckpt_base",       type=str,   default="checkpoints/stage2_vdm")
    p.add_argument("--vqvae_base",      type=str,   default="checkpoints/stage1_vqvae")
    p.add_argument("--cyclegan_ckpt",   type=str,   default="checkpoints/cyclegan/best.pt",
                   help="CycleGAN 체크포인트 경로")
    p.add_argument("--output_dir",      type=str,   default="eval_results/gen")
    p.add_argument("--device",          type=int,   default=0)
    p.add_argument("--n_sample_steps",  type=int,   default=200)
    p.add_argument("--batch_size",      type=int,   default=8)
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--val_ratio",       type=float, default=0.2)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--keys",  nargs="+", default=None,
                   help="평가할 모델 key 목록 (기본: 전체)")
    p.add_argument("--overwrite", action="store_true",
                   help="이미 존재하는 .mha 파일도 덮어쓰기")
    return p.parse_args()


def main():
    args   = get_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    set_determinism(seed=args.seed)
    out = pathlib.Path(args.output_dir)

    configs = MODEL_CONFIGS
    if args.keys:
        configs = [c for c in MODEL_CONFIGS if c["key"] in args.keys]

    # 이미 완료된 모델과 미완료 모델 분리 (미완료 먼저 처리)
    complete, incomplete = [], []
    for cfg in configs:
        key = cfg["key"]
        n   = cfg.get("n", 1)
        model_dir = out / key

        try:
            sids = get_val_subject_ids(args.data_root, n, args.val_ratio, args.seed)
        except Exception:
            incomplete.append(cfg)
            continue

        if not args.overwrite and all_files_exist(model_dir, sids):
            complete.append(cfg)
        else:
            incomplete.append(cfg)

    if complete:
        print(f"[스킵] 이미 완료된 모델: {[c['key'] for c in complete]}")
    print(f"[생성 대상] {[c['key'] for c in incomplete]}")

    for cfg in incomplete:
        key = cfg["key"]
        print(f"\n>>> {key}")

        if cfg.get("type") == "cyclegan":
            try:
                generate_cyclegan(
                    cyclegan_ckpt=args.cyclegan_ckpt,
                    data_root=args.data_root,
                    device=device,
                    out_dir=out,
                    val_ratio=args.val_ratio,
                    seed=args.seed,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    overwrite=args.overwrite,
                )
            except Exception as e:
                print(f"  [건너뜀] {e}")
        else:
            try:
                loaded = load_model_for_eval(
                    cfg=cfg,
                    data_root=args.data_root,
                    ckpt_base=args.ckpt_base,
                    vqvae_base=args.vqvae_base,
                    device=device,
                    val_ratio=args.val_ratio,
                    seed=args.seed,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )
                generate_model(cfg, loaded, device, out,
                               n_sample_steps=args.n_sample_steps,
                               overwrite=args.overwrite)
            except Exception as e:
                print(f"  [건너뜀] {e}")

    print(f"\n완료: {out}/")


if __name__ == "__main__":
    main()
