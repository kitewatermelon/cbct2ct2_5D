"""n1~n9 VQVAE 파라미터 수 비교.

stage1_full.py / stage2_full.py 의 build_vqvae 설정을 그대로 사용.
  - Stage 1: embedding_dim=1, compress_ratio={2,4,8}
  - Stage 2: embedding_dim=4, compress_ratio={2,4,8}
"""

from monai.networks.nets import VQVAE


def build_vqvae(in_channels, out_channels, compress_ratio, embedding_dim, num_embeddings=2048):
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


def count_params(model):
    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def run_comparison(stage_label, embedding_dim, compress_ratios=(4,), ns=range(1, 10)):
    print(f"\n{'='*70}")
    print(f"  {stage_label}  (embedding_dim={embedding_dim})")
    print(f"{'='*70}")

    for cpr in compress_ratios:
        print(f"\n  compress_ratio = {cpr}")
        print(f"  {'n':>4}  {'total params':>15}  {'trainable':>15}  {'diff vs n1':>12}")
        print(f"  {'-'*50}")

        baseline = None
        for n in ns:
            model = build_vqvae(
                in_channels=n, out_channels=1,
                compress_ratio=cpr, embedding_dim=embedding_dim,
            )
            total, trainable = count_params(model)
            if baseline is None:
                baseline = total
            diff = total - baseline
            sign = "+" if diff >= 0 else ""
            print(f"  n={n}  {total:>15,}  {trainable:>15,}  {sign}{diff:>11,}")


if __name__ == "__main__":
    compress_ratios = [2, 4, 8]

    # Stage 1: embedding_dim=1
    run_comparison("Stage 1  (stage1_full.py)", embedding_dim=1, compress_ratios=compress_ratios)

    # Stage 2: embedding_dim=4
    run_comparison("Stage 2  (stage2_full.py)", embedding_dim=4, compress_ratios=compress_ratios)

    print(f"\n{'='*70}")
    print("  참고: n1~n9 차이는 첫 번째 Conv 입력 채널 수에서만 발생합니다.")
    print(f"{'='*70}\n")
