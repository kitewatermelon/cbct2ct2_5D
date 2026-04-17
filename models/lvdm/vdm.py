import numpy as np
import torch
from torch import allclose, argmax, autograd, exp, linspace, nn, sigmoid, sqrt
from torch.special import expm1
from tqdm import trange

from models.lvdm.utils import maybe_unpack_batch, unsqueeze_right
import pdb


class VDM(nn.Module):
    def __init__(self, model, cfg, ae, image_shape):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.image_shape = image_shape
        self.vocab_size = 1
        self.ae = ae
        if cfg.noise_schedule == "fixed_linear":
            self.gamma = FixedLinearSchedule(cfg.gamma_min, cfg.gamma_max)
        elif cfg.noise_schedule == "learned_linear":
            self.gamma = LearnedLinearSchedule(cfg.gamma_min, cfg.gamma_max)
        else:
            raise ValueError(f"Unknown noise schedule {cfg.noise_schedule}")

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def sample_p_s_t(self, z, t, s, clip_samples, context=None, eta: float=0.0):
        """Samples from p(z_s | z_t, x). Used for standard ancestral sampling."""
        gamma_t = self.gamma(t)
        gamma_s = self.gamma(s)
        c = -expm1(gamma_s - gamma_t)
        alpha_t = sqrt(sigmoid(-gamma_t))
        alpha_s = sqrt(sigmoid(-gamma_s))
        sigma_t = sqrt(sigmoid(gamma_t))
        sigma_s = sqrt(sigmoid(gamma_s))

        #pred_noise = self.model(z, gamma_t)
        z_hat = self.model(x=z, timesteps=gamma_t, context=context)

        # 수정
        from monai.networks.nets import DiffusionModelUNet
        gamma_t_in = gamma_t.expand(z.shape[0]) if (
            isinstance(self.model, DiffusionModelUNet) and gamma_t.dim() == 0
        ) else gamma_t
        z_hat = self.model(x=z, timesteps=gamma_t_in, context=context)

        if clip_samples:
            z_hat =  z_hat.clamp_(0.0, 1.0)
            mean = alpha_s * (z * (1 - c) / alpha_t + c * z_hat)
        
        else:
            mean = alpha_s * (z * (1 - c) / alpha_t + c * z_hat)
        
        scale = eta * sigma_s * sqrt(c)
        return mean + scale * torch.randn_like(z)

    @torch.no_grad()
    def sample(self, batch_size, n_sample_steps, clip_samples, y=None):
        z = torch.randn((batch_size, *self.image_shape), device=self.device)
        steps = linspace(1.0, 0.0, n_sample_steps + 1, device=self.device)
        for i in trange(n_sample_steps, desc="sampling"):
            z = self.sample_p_s_t(z, steps[i], steps[i + 1], clip_samples, context=y)
        return z

    def sample_q_t_0(self, x, times, noise=None):
        """Samples from the distributions q(x_t | x_0) at the given time steps."""
        with torch.enable_grad():  # Need gradient to compute loss even when evaluating
            gamma_t = self.gamma(times)
        gamma_t_padded = unsqueeze_right(gamma_t, x.ndim - gamma_t.ndim)
        mean = x * sqrt(sigmoid(-gamma_t_padded))  # x * alpha
        scale = sqrt(sigmoid(gamma_t_padded))
        if noise is None:
            noise = torch.randn_like(x)
        return mean + noise * scale, gamma_t

    def sample_times(self, batch_size):
        if self.cfg.antithetic_time_sampling:
            t0 = np.random.uniform(0, 1 / batch_size)
            times = torch.arange(t0, 1.0, 1.0 / batch_size, device=self.device)
        else:
            times = torch.rand(batch_size, device=self.device)
        return times

    
    def _alpha_sigma(self, gamma):
        sigma = torch.sqrt(sigmoid(gamma))
        alpha = torch.sqrt(1.0 - sigma**2)
        return alpha, sigma

    
    def recon_loss(self, img, f):
        g0 = self.gamma(torch.tensor(0.0, device=f.device))
        alpha0, sigma0 = self._alpha_sigma(g0)

        eps  = torch.randn_like(f)
        z0   = alpha0 * f + sigma0 * eps
        z0_r = z0 / alpha0
        with torch.no_grad():
            x_dist = self.ae.decode_stage_2_outputs(z0_r)          # returns distribution
        mse = 0.5 * ((img - x_dist)**2).sum((1,2,3)) 
        return mse


    def forward(self, x, cond, img, *, noise=None):
        assert x.shape[1:] == self.image_shape

        bpd_factor = 1 / (np.prod(x.shape[1:]) * np.log(2))

        # Sample from q(x_t | x_0) with random t.
        times = self.sample_times(x.shape[0]).requires_grad_(True)
        if noise is None:
            noise = torch.randn_like(x)
        x_t, gamma_t = self.sample_q_t_0(x=x, times=times, noise=noise)

        # Forward through model
        x_hat = self.model(x=x_t, timesteps=gamma_t, context=cond)
        gamma_grad = autograd.grad(  # gamma_grad shape: (B, )
            gamma_t,  # (B, )
            times,  # (B, )
            grad_outputs=torch.ones_like(gamma_t),
            create_graph=True,
            retain_graph=True,
        )[0]
        snr_t = torch.exp(-gamma_t)
        pred_loss = snr_t * ((x_hat - x) ** 2).sum((1, 2, 3))  # (B, )
        diffusion_loss = 0.5 * pred_loss * gamma_grad * bpd_factor
        # *** Latent loss (bpd): KL divergence from N(0, 1) to q(z_1 | x)
        gamma_1 = self.gamma(torch.tensor([1.0], device=self.device))
        sigma_1_sq = sigmoid(gamma_1)
        mean_sq = (1 - sigma_1_sq) * x**2  # (alpha_1 * x)**2
        latent_loss = kl_std_normal(mean_sq, sigma_1_sq).sum((1, 2, 3)) * bpd_factor

        with torch.no_grad():
            f = self.ae.encode_stage_2_inputs(img)
        recon_loss = self.recon_loss(img, f) * bpd_factor

        loss = diffusion_loss + latent_loss + recon_loss

        with torch.no_grad():
            gamma_0 = self.gamma(torch.tensor([0.0], device=self.device))
        metrics = {
            "bpd": loss.mean(),
            "diff_loss": diffusion_loss.mean(),
            "latent_loss": latent_loss.mean(),
            "recon_loss": recon_loss.mean(),
            "gamma_0": gamma_0.item(),
            "gamma_1": gamma_1.item(),
        }
        return loss.mean(), metrics


def kl_std_normal(mean_squared, var):
    return 0.5 * (var + mean_squared - torch.log(var.clamp(min=1e-15)) - 1.0)


class FixedLinearSchedule(nn.Module):
    def __init__(self, gamma_min, gamma_max):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def forward(self, t):
        return self.gamma_min + (self.gamma_max - self.gamma_min) * t


class LearnedLinearSchedule(nn.Module):
    def __init__(self, gamma_min, gamma_max):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(gamma_min))
        self.w = nn.Parameter(torch.tensor(gamma_max - gamma_min))

    def forward(self, t):
        return self.b + self.w.abs() * t


if __name__ == "__main__":
    from models.lvdm.uvit import UViT
    from monai.networks.nets import VQVAE, DiffusionModelUNet
    from monai.networks.nets.diffusion_model_unet import DiffusionModelUNet
    from types import SimpleNamespace
    import torch.nn.functional as F

    cfg = SimpleNamespace(
        noise_schedule="fixed_linear",
        gamma_min=-13.3,
        gamma_max=5.0,
        antithetic_time_sampling=True,
    )

    ae = VQVAE(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(128, 256, 512, 512),
        num_res_channels=256,
        num_res_layers=256,
        downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1), (1, 3, 1, 1), (1, 3, 1, 1)),
        upsample_parameters=((1, 3, 1, 1, 0), (1, 3, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
        num_embeddings=2048,
        embedding_dim=1,
        commitment_cost=0.4,
    )

    # cond latent: (B, 1, 32, 32) → (B, 1, 1024)
    COND_DIM = 1 * 32 * 32  # 1024

    uvit = UViT(
        img_size=32,
        patch_size=2,
        in_chans=1,
        embed_dim=512,
        depth=11,
        num_heads=4,
        conv=True,
    )

    unet = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_res_blocks=2,
        channels=(128, 256, 512),
        attention_levels=(True, True, True),
        norm_num_groups=8,
        num_head_channels=(16, 32, 64),
        with_conditioning=True,
        cross_attention_dim=COND_DIM,  # 1024
        transformer_num_layers=1,
    )

    B, C, H, W = 2, 1, 32, 32
    image_shape = (C, H, W)

    x_img    = torch.randn(B, 1, 128, 128)   # CT
    cond_img = torch.randn(B, 1, 128, 128)   # CBCT

    with torch.no_grad():
        x    = ae.encode_stage_2_inputs(x_img)     # (B, 1, 32, 32)
        cond_latent = ae.encode_stage_2_inputs(cond_img)  # (B, 1, 32, 32)

    # UViT 테스트
    print("=== UViT ===")
    vdm_uvit = VDM(model=uvit, cfg=cfg, ae=ae, image_shape=image_shape)
    loss, metrics = vdm_uvit(x, cond_latent, x_img)
    print(f"loss: {loss.item():.4f}", metrics)

    # UNet 테스트
    print("=== UNet ===")
    vdm_unet = VDM(model=unet, cfg=cfg, ae=ae, image_shape=image_shape)
    loss, metrics = vdm_unet(x, cond_latent.view(B, 1, -1), x_img)
    print(f"loss: {loss.item():.4f}", metrics)