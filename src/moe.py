"""Two-expert mixture with a hard conditioning-stats router.

The prescribed MoE for this project is deliberately not a learned gate:
with two experts and interpretable conditioning, a hard rule on window
statistics is auditable and cannot silently misroute. The router reads
the same 4-channel conditioning the UNet sees and picks the expert whose
training distribution the window resembles:

  expert "town"  - hilly small towns (TOWNS): slope-rich, low built share
  expert "urban" - flat metro fringes (URBAN_TOWNS + US_TOWNS): low slope
                   variance, higher built share, denser road grids

Routing features (from cond [4,G,G]): mean of the slope channel, built
fraction (density > 0.03), road density. Thresholds were chosen from the
training-window distributions and are exposed as arguments so a control
experiment can sweep them. route() returns the name plus the features so
callers can log WHY a window was routed.

Honest caveats: a hard gate has no blending, so windows near the
boundary get whichever side of the threshold they fall on; and the gate
encodes our split of the data, not a learned notion of urbanness.
"""
import numpy as np

SLOPE_T = 0.18      # mean of slope/30 channel; hilly towns sit well above
BUILT_T = 0.055     # built fraction; metro fringes sit above
ROAD_T = 0.030      # road-pixel fraction; dense grids sit above


def route_features(cond):
    """cond: numpy [4,G,G] (elev_z, slope/30, density, core roads)."""
    c = np.asarray(cond)
    if c.ndim == 4:
        c = c[0]
    return {"slope_mean": float(c[1].mean()),
            "built_frac": float((c[2] > 0.03).mean()),
            "road_frac": float((c[3] > 0.5).mean())}


def route(cond, slope_t=SLOPE_T, built_t=BUILT_T, road_t=ROAD_T):
    """Hard gate. Returns (expert_name, features)."""
    f = route_features(cond)
    urban_votes = int(f["slope_mean"] < slope_t) \
        + int(f["built_frac"] > built_t) + int(f["road_frac"] > road_t)
    return ("urban" if urban_votes >= 2 else "town"), f


class MoE:
    """Holds two Diffusion wrappers; delegates sampling to the routed one.

    experts: dict name -> model.Diffusion. Both must share cond format.
    Mixed batches are split by window and sampled per expert, preserving
    input order in the output.
    """

    def __init__(self, experts, slope_t=SLOPE_T, built_t=BUILT_T,
                 road_t=ROAD_T):
        if set(experts) != {"town", "urban"}:
            raise ValueError("experts must be {'town','urban'}")
        self.experts = experts
        self.t = (slope_t, built_t, road_t)
        self.last_log = []

    def sample_ddim(self, cond, steps=50, eta=0.0, generator=None):
        import torch
        self.last_log = []
        names = []
        for i in range(cond.shape[0]):
            name, f = route(cond[i].detach().cpu().numpy(), *self.t)
            names.append(name)
            self.last_log.append((name, f))
        out = [None] * len(names)
        for name in ("town", "urban"):
            idx = [i for i, n in enumerate(names) if n == name]
            if not idx:
                continue
            xs = self.experts[name].sample_ddim(
                cond[idx], steps=steps, eta=eta, generator=generator)
            for j, i in enumerate(idx):
                out[i] = xs[j]
        return torch.stack(out)


def load_moe(town_ckpt, urban_ckpt, device="cuda", cond_ch=4, out_ch=3):
    """Build a MoE from two checkpoint paths (EMA weights)."""
    import torch
    from model import UNet, Diffusion
    experts = {}
    for name, path in (("town", town_ckpt), ("urban", urban_ckpt)):
        net = UNet(cond_ch=cond_ch, out_ch=out_ch).to(device)
        net.load_state_dict(torch.load(path, map_location=device)["ema"])
        experts[name] = Diffusion(net, device=device)
    return MoE(experts)


if __name__ == "__main__":
    G = 128
    rng = np.random.default_rng(0)

    def fake_cond(slope, built, road):
        c = np.zeros((4, G, G), dtype=np.float32)
        c[1] = slope
        n_b = int(built * G * G)
        n_r = int(road * G * G)
        c[2].ravel()[rng.choice(G * G, n_b, replace=False)] = 0.5
        c[3].ravel()[rng.choice(G * G, n_r, replace=False)] = 1.0
        return c

    hilly = fake_cond(slope=0.4, built=0.02, road=0.01)
    flat_metro = fake_cond(slope=0.05, built=0.12, road=0.05)
    name, f = route(hilly)
    assert name == "town", (name, f)
    print("route hilly -> town ok")
    name, f = route(flat_metro)
    assert name == "urban", (name, f)
    print("route flat metro -> urban ok")

    class _FakeDiff:
        def __init__(self, tag):
            self.tag = tag

        def sample_ddim(self, cond, steps=50, eta=0.0, generator=None):
            import torch
            b = cond.shape[0]
            return torch.full((b, 3, G, G), float(self.tag))

    try:
        import torch
        moe = MoE({"town": _FakeDiff(1), "urban": _FakeDiff(2)})
        cond = torch.from_numpy(np.stack([hilly, flat_metro, hilly]))
        x = moe.sample_ddim(cond, steps=4)
        assert x.shape == (3, 3, G, G)
        assert float(x[0, 0, 0, 0]) == 1 and float(x[1, 0, 0, 0]) == 2 \
            and float(x[2, 0, 0, 0]) == 1
        assert [n for n, _ in moe.last_log] == ["town", "urban", "town"]
        print("moe order-preserving dispatch ok")
    except ImportError:
        print("torch unavailable: dispatch test skipped (router tested)")
    print("all moe self-tests passed")
