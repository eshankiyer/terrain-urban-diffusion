"""Terrain-conditioned denoising diffusion model.

A compact U-Net (~13M params at base=64) predicts the noise added to the
2-channel expansion target, given 4 conditioning channels concatenated to the
noisy input. Cosine noise schedule, epsilon prediction, DDIM sampling.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

T_STEPS = 1000


def cosine_schedule(t_steps=T_STEPS, s=0.008):
    steps = torch.arange(t_steps + 1, dtype=torch.float64)
    f = torch.cos((steps / t_steps + s) / (1 + s) * math.pi / 2) ** 2
    alphas_bar = (f / f[0]).clamp(1e-8, 1.0)
    betas = (1 - alphas_bar[1:] / alphas_bar[:-1]).clamp(1e-8, 0.999)
    return betas.float()


class SinusoidalEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) *
                          torch.arange(half, device=t.device) / (half - 1))
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, c_out)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.GroupNorm(8, c)
        self.qkv = nn.Conv2d(c, c * 3, 1)
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.flatten(2).transpose(1, 2)   # b, hw, c
        k = k.flatten(2).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        return x + self.proj(out.transpose(1, 2).reshape(b, c, h, w))


class UNet(nn.Module):
    def __init__(self, cond_ch=4, out_ch=2, base=64, t_dim=256):
        super().__init__()
        self.t_emb = nn.Sequential(SinusoidalEmb(t_dim),
                                   nn.Linear(t_dim, t_dim), nn.SiLU(),
                                   nn.Linear(t_dim, t_dim))
        cs = [base, base * 2, base * 4]
        self.stem = nn.Conv2d(out_ch + cond_ch, base, 3, padding=1)
        # down
        self.d1 = ResBlock(cs[0], cs[0], t_dim)
        self.d2 = ResBlock(cs[0], cs[1], t_dim)
        self.d3 = ResBlock(cs[1], cs[2], t_dim)
        self.att3 = SelfAttention(cs[2])
        self.pool = nn.AvgPool2d(2)
        # mid
        self.m1 = ResBlock(cs[2], cs[2], t_dim)
        self.matt = SelfAttention(cs[2])
        self.m2 = ResBlock(cs[2], cs[2], t_dim)
        # up
        self.u3 = ResBlock(cs[2] + cs[2], cs[1], t_dim)
        self.uatt3 = SelfAttention(cs[1])
        self.u2 = ResBlock(cs[1] + cs[1], cs[0], t_dim)
        self.u1 = ResBlock(cs[0] + cs[0], cs[0], t_dim)
        self.head = nn.Sequential(nn.GroupNorm(8, cs[0]), nn.SiLU(),
                                  nn.Conv2d(cs[0], out_ch, 3, padding=1))

    def forward(self, x_noisy, cond, t):
        temb = self.t_emb(t)
        x = self.stem(torch.cat([x_noisy, cond], dim=1))
        h1 = self.d1(x, temb)                      # 128
        h2 = self.d2(self.pool(h1), temb)          # 64
        h3 = self.att3(self.d3(self.pool(h2), temb))  # 32
        m = self.m2(self.matt(self.m1(self.pool(h3), temb)), temb)  # 16
        u = F.interpolate(m, scale_factor=2, mode="nearest")
        u = self.uatt3(self.u3(torch.cat([u, h3], 1), temb))
        u = F.interpolate(u, scale_factor=2, mode="nearest")
        u = self.u2(torch.cat([u, h2], 1), temb)
        u = F.interpolate(u, scale_factor=2, mode="nearest")
        u = self.u1(torch.cat([u, h1], 1), temb)
        return self.head(u)


class Diffusion:
    def __init__(self, model, device="cpu", t_steps=T_STEPS):
        self.model = model
        self.device = device
        self.t_steps = t_steps
        betas = cosine_schedule(t_steps).to(device)
        alphas = 1.0 - betas
        self.alphas_bar = torch.cumprod(alphas, dim=0)

    def loss(self, x0, cond):
        b = x0.shape[0]
        t = torch.randint(0, self.t_steps, (b,), device=self.device)
        noise = torch.randn_like(x0)
        ab = self.alphas_bar[t][:, None, None, None]
        x_noisy = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        pred = self.model(x_noisy, cond, t)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample_ddim(self, cond, steps=50, eta=0.0, generator=None):
        b = cond.shape[0]
        x = torch.randn(b, 2, cond.shape[2], cond.shape[3],
                        device=self.device, generator=generator)
        ts = torch.linspace(self.t_steps - 1, 0, steps, device=self.device).long()
        for i, t in enumerate(ts):
            ab_t = self.alphas_bar[t]
            ab_prev = self.alphas_bar[ts[i + 1]] if i + 1 < steps else torch.tensor(1.0, device=self.device)
            eps = self.model(x, cond, t.repeat(b))
            x0 = ((x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()).clamp(-1, 1)
            sigma = eta * ((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev)).sqrt()
            dir_xt = (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps
            x = ab_prev.sqrt() * x0 + dir_xt
            if eta > 0 and i + 1 < steps:
                x = x + sigma * torch.randn_like(x)
        return x


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow)
