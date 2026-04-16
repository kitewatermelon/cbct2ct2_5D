"""utils/wandb.py — WandB 학습 로깅 유틸리티."""
from __future__ import annotations

import wandb


def init_wandb(config: dict, project: str, experiment_name: str, entity: str | None = None):
    """WandB run 초기화.

    Args:
        config:          하이퍼파라미터 dict (yaml config 내용).
        project:         WandB 프로젝트 이름.
        experiment_name: run 이름.
        entity:          WandB 팀/유저명. None이면 기본값 사용.
    """
    run = wandb.init(
        project=project,
        name=experiment_name,
        entity=entity,
        config=config,
        resume="allow",
    )
    return run


def log_train(metrics: dict, epoch: int):
    """학습 메트릭 로깅."""
    wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=epoch)


def log_val(metrics: dict, epoch: int):
    """검증 메트릭 로깅."""
    wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=epoch)


def log_images(tag: str, images: "torch.Tensor", epoch: int, max_images: int = 4):
    """이미지 텐서 로깅. images: (B, C, H, W) or (B, 1, H, W).

    슬라이스가 여러 채널이면 중앙 채널만 시각화.
    """
    import torch
    imgs = images[:max_images].detach().float().cpu()
    # 채널이 여러 개면 중앙 슬라이스만 사용
    if imgs.shape[1] > 1:
        mid = imgs.shape[1] // 2
        imgs = imgs[:, mid : mid + 1]

    # [0, 1] 클리핑
    imgs = imgs.clamp(0, 1)

    wandb.log(
        {tag: [wandb.Image(imgs[i]) for i in range(imgs.shape[0])]},
        step=epoch,
    )


def finish():
    wandb.finish()