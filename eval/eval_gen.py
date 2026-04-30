"""eval_gen.py — Stage2 VDM inference-only: 생성 결과를 .mha로 저장.

Usage:
    python eval_gen.py [--keys uvit_n5_cpr4 unet_n5_cpr4] [--overwrite]

출력:
    {output_dir}/{model_key}/{subj_id}.mha  — generated CT (float32, [0,1] normalized)
    {output_dir}/{model_key}/meta.json      — {"subj_id": "anatomy", ...}
"""
from __future__ import annotations
import argparse, json, pathlib
import torch
from monai.utils import set_determinism
from tqdm.auto import tqdm

from eval.eval_full import MODEL_CONFIGS, load_model_for_eval
from train.stage2_vdm import _prepare_cond, sample_conditional
from utils.mha import save_mha, load_mha  # re-exported for external importers


# ---------------------------------------------------------------------------
# Per-model generation
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
    n            = cfg["n"]
    mid          = n // 2
    key          = cfg["key"]

    model_dir = out_dir / key
    model_dir.mkdir(parents=True, exist_ok=True)

    meta: dict[str, str] = {}
    first_batch = True

    for batch in tqdm(val_loader, desc=key, leave=False):
        ct_img   = batch["ct"].to(device)
        cbct_img = batch["cbct"].to(device)
        subj_ids = batch["subj_id"]

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
            arr = ct_gen[i, 0].cpu().numpy()   # (H, W)
            save_mha(arr, path)
            meta[sid] = subj_anatomy.get(sid, "UNKNOWN")

    meta_path = model_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  [저장] {model_dir}/ ({len(meta)}개 샘플)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Stage2 VDM inference-only: .mha 저장")
    p.add_argument("--data_root",       type=str,   default="/home/dministrator/s2025")
    p.add_argument("--ckpt_base",       type=str,   default="checkpoints/stage2_vdm")
    p.add_argument("--vqvae_base",      type=str,   default="checkpoints/stage1_vqvae")
    p.add_argument("--output_dir",      type=str,   default="eval_results/gen")
    p.add_argument("--device",          type=int,   default=0)
    p.add_argument("--n_sample_steps",  type=int,   default=200)
    p.add_argument("--batch_size",      type=int,   default=8)
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--val_ratio",       type=float, default=0.2)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--keys",  nargs="+", default=None,
                   help="평가할 모델 key 목록 (기본: 전체 8개)")
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

    print(f"생성 대상 모델: {[c['key'] for c in configs]}")

    for cfg in configs:
        print(f"\n>>> {cfg['key']}")
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
