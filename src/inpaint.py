"""RePaint-style region-locked regeneration.

Lets a planner LOCK parts of the canvas -- existing fabric, floodplains,
protected land -- and have the diffusion model regenerate only the rest.
At every reverse step the locked region is replaced by the KNOWN content
forward-diffused to the current noise level, so the model denoises the free
region while always seeing consistent locked context. Resampling jumps
(go back jump_len steps and redo them, jump_n times) let the free region
re-harmonize with the boundary instead of leaving a visible seam.

The loop mirrors model.Diffusion.sample_ddim exactly (same linspace
timesteps, same eta=0 update, same x0 clamp); RePaint noise enters only
through the known-composite and the back-jump re-noising. Honest caveats:
the model was never trained with masked context, so a hard lock is
distribution shift -- narrow free regions squeezed between locked fabric can
come out blander than free-run samples, and jump_n=2/jump_len=10 is a cost
knob, not a tuned optimum. Locked pixels are hard-composited at the end, so
they are exact by construction, not evidence the model respected them.
"""

import numpy as np
import torch


def encode_known(roads_new0, dens_next0, amen0=None, out_ch=3):
    """Stack known rasters (each [G,G] in 0..1) into target space [-1,1].
    amen0=None fills the amenity channel with -1 (no known amenities)."""
    chans = [np.asarray(roads_new0, np.float32),
             np.asarray(dens_next0, np.float32)]
    if out_ch >= 3:
        chans.append(np.zeros_like(chans[0]) if amen0 is None
                     else np.asarray(amen0, np.float32))
    return (2.0 * np.stack(chans[:out_ch]) - 1.0).astype(np.float32)


def protect_mask(*masks):
    """Union of boolean-ish [G,G] masks -> float32 [G,G], 1=locked.
    Convenience for floodplain | steep | existing fabric."""
    out = np.zeros(np.asarray(masks[0]).shape, dtype=bool)
    for m in masks:
        out |= np.asarray(m) > 0.5
    return out.astype(np.float32)


def repaint_schedule(steps, jump_len=10, jump_n=2):
    """Indices into the DDIM timestep array, with back-jumps: every jump_len
    outer steps, rewind jump_len steps and redo them, jump_n times."""
    seq, i = [], 0
    while i < steps:
        seq.append(i)
        i += 1
        if jump_len > 0 and jump_n > 0 and i < steps and i % jump_len == 0:
            for _ in range(jump_n):
                seq.extend(range(i - jump_len, i))
    return seq


def _as_tensor(a, device):
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(np.ascontiguousarray(a))
    return a.to(device=device, dtype=torch.float32)


@torch.no_grad()
def sample_inpaint(diff, cond, keep_mask, x_known, steps=50,
                   jump_len=10, jump_n=2, generator=None):
    """DDIM sampling with locked regions. diff: model.Diffusion; cond
    [b,4,G,G] on diff.device; keep_mask [G,G]/[1,G,G]/[out_ch,G,G] (1=locked,
    numpy or torch); x_known [out_ch,G,G] target-space content for locked
    pixels. Returns [b,out_ch,G,G]; locked pixels are exactly x_known."""
    dev, b = diff.device, cond.shape[0]
    out_ch = getattr(diff.model, "out_ch", 2)
    m = _as_tensor(keep_mask, dev).clamp(0, 1)
    if m.dim() == 2:
        m = m[None]
    if m.shape[0] == 1:
        m = m.expand(out_ch, -1, -1)
    m = m[None]                                   # [1,out_ch,G,G]
    xk = _as_tensor(x_known, dev)[None]           # [1,out_ch,G,G]

    ts = torch.linspace(diff.t_steps - 1, 0, steps, device=dev).long()
    one = torch.tensor(1.0, device=dev)
    ab = lambda lvl: diff.alphas_bar[ts[lvl]] if lvl < steps else one  # noqa: E731

    def randn():
        return torch.randn(b, out_ch, cond.shape[2], cond.shape[3],
                           device=dev, generator=generator)

    x, lvl = randn(), 0
    for idx in repaint_schedule(steps, jump_len, jump_n):
        if idx < lvl:                             # back-jump: re-noise x
            r = (ab(idx) / ab(lvl)).clamp(max=1.0)
            x = r.sqrt() * x + (1 - r).sqrt() * randn()
        ab_t, ab_prev = ab(idx), ab(idx + 1)
        xk_t = ab_t.sqrt() * xk + (1 - ab_t).sqrt() * randn()
        x = m * xk_t + (1 - m) * x
        eps = diff.model(x, cond, ts[idx].repeat(b))
        x0 = ((x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()).clamp(-1, 1)
        x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        lvl = idx + 1
    return m * xk + (1 - m) * x


if __name__ == "__main__":
    import torch.nn as nn

    import model

    class _Toy(nn.Module):
        out_ch = 3

        def forward(self, x, cond, t):
            return 0.1 * x

    G = 128
    diff = model.Diffusion(_Toy(), device="cpu")
    cond = torch.zeros(1, 4, G, G)

    keep = np.zeros((G, G), np.float32)
    keep[:, :G // 2] = 1.0                        # left half locked
    stripes = np.zeros((G, G), np.float32)
    stripes[::4] = 1.0
    xk = encode_known(stripes, stripes, stripes)  # +1 stripes, target space
    assert xk.shape == (3, G, G) and xk.min() == -1 and xk.max() == 1
    assert (encode_known(stripes, stripes)[2] == -1).all()  # amen0=None
    print("encode_known ok")

    pm = protect_mask(keep, stripes > 0.5)
    assert pm.dtype == np.float32 and pm.shape == (G, G) and pm.max() == 1
    print("protect_mask ok")

    steps, jl, jn = 20, 5, 1
    seq = repaint_schedule(steps, jl, jn)
    assert len(seq) > steps and max(seq) == steps - 1
    assert len(repaint_schedule(steps, jl, 0)) == steps
    print("jump schedule ok")

    gen = torch.Generator().manual_seed(0)
    out = sample_inpaint(diff, cond, keep, xk, steps=steps,
                         jump_len=jl, jump_n=jn, generator=gen)
    xk_t = torch.from_numpy(xk)
    assert out.shape == (1, 3, G, G)
    assert (out[0, :, :, :G // 2] - xk_t[:, :, :G // 2]).abs().max() < 1e-5
    assert (out[0, :, :, G // 2:] - xk_t[:, :, G // 2:]).std() > 0
    print("sample_inpaint ok")
