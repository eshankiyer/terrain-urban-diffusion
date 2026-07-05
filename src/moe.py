"""Five-expert mixture with a hard conditioning-stats router.

v4 had two growth styles: hilly European towns and planned flat fringes.
That pair covers a narrow slice of how settlements actually grow, so v5
widens the bench to five experts sharing one architecture:

expert "village"   near-empty windows where the settlement is a speck
                   and the question is which way it creeps (VILLAGES)
expert "town"      hilly small towns, the original regime (TOWNS)
expert "urban"     planned flat fringes: EU/US/AU/NZ/CA subdivisions
                   (URBAN_TOWNS + US_TOWNS + AUS_NZ_CA_TOWNS)
expert "informal"  unplanned periurban frontiers, where building outruns
                   the mapped road network (INFORMAL_TOWNS)
expert "megacity"  saturated growth edges of the largest cities, Delhi
                   included (MEGACITY_TOWNS)

The router stays a hard rule on window statistics, not a learned gate,
for the same reason as v4: with interpretable conditioning a rule is
auditable and cannot silently misroute, and every decision comes back
with the features that produced it. Rules read cond channels 1-3
(slope, density, roads -- unchanged since v3; the v4 water channel is
ignored). First match wins:

1. built_frac < VILLAGE_T                    -> village
2. built_frac > MEGA_T                       -> megacity
3. built_frac > INFORMAL_BUILT_T and
   road_per_built < INFORMAL_RATIO_T         -> informal
4. slope_mean >= SLOPE_T                     -> town
5. built_frac > BUILT_T or road_frac > ROAD_T -> urban
6. otherwise                                 -> town

road_per_built is the one new feature. Informal frontiers put up
buildings faster than mapped roads; planned fringes do the opposite.
The ratio separates them where built fraction alone cannot. Note this
also means a window whose roads are simply unmapped in OSM reads as
informal -- for routing purposes that is arguably the right call, since
the model has no roads to hang planned growth on either way.

Deployments rarely carry all five checkpoints, so routing degrades
explicitly rather than crashing: FALLBACK maps each expert to
substitutes in preference order and the MoE dispatches only to experts
it actually holds. A two-expert deployment (town + urban) behaves like
v4. route() reports the requested expert and the log records which one
was used, so a fallback is never silent.

Scale is the router's job, not a sixth expert's. The window is fixed at
1.9 km, so "village vs Delhi" is a question of how full and how road-
served the window is -- which is exactly what built_frac and road_frac
measure. Central Delhi windows are already saturated and route to
megacity, whose training data is saturated edges; there is no growth to
predict in a fully built window anyway.

Honest caveats, unchanged from v4: a hard gate has no blending, so a
window near a threshold gets whichever side it falls on; and the
thresholds encode our split of the training data, not a learned notion
of urbanness. They were chosen from training-window distributions and
are exposed as arguments so a control experiment can sweep them.
"""
import numpy as np

SLOPE_T = 0.18   # mean of slope/30 channel; hilly towns sit well above
BUILT_T = 0.055  # built fraction; metro fringes sit above
ROAD_T = 0.030   # road-pixel fraction; dense grids sit above

VILLAGE_T = 0.020         # built fraction below which a window is
                          # essentially unbuilt: village territory
MEGA_T = 0.220            # built fraction above which the window is a
                          # saturated metro edge: megacity territory
INFORMAL_BUILT_T = 0.100  # enough building for the ratio test to mean
                          # something
INFORMAL_RATIO_T = 0.35   # road_frac / built_frac below this reads as
                          # unplanned growth

EXPERT_NAMES = ("village", "town", "urban", "informal", "megacity")

# Substitutes in preference order, used when a deployment does not carry
# the routed expert. village falls back to town (both sparse, terrain-
# led); informal to urban then megacity (closest density regimes);
# megacity to urban. town and urban fall back to each other, which is
# exactly the v4 behaviour.
FALLBACK = {
    "village": ("town", "urban"),
    "town": ("urban",),
    "urban": ("town",),
    "informal": ("urban", "megacity", "town"),
    "megacity": ("urban", "informal", "town"),
}


def route_features(cond):
    """cond: numpy [C,G,G], C >= 4 (elev_z, slope/30, density, core
    roads, [water]). Water, if present, is ignored by the router."""
    c = np.asarray(cond)
    if c.ndim == 4:
        c = c[0]
    built = float((c[2] > 0.03).mean())
    road = float((c[3] > 0.5).mean())
    return {"slope_mean": float(c[1].mean()),
            "built_frac": built,
            "road_frac": road,
            "road_per_built": road / max(built, 1e-6)}


def route(cond, slope_t=SLOPE_T, built_t=BUILT_T, road_t=ROAD_T,
          village_t=VILLAGE_T, mega_t=MEGA_T,
          informal_built_t=INFORMAL_BUILT_T,
          informal_ratio_t=INFORMAL_RATIO_T):
    """Hard gate over five experts. Returns (expert_name, features)."""
    f = route_features(cond)
    if f["built_frac"] < village_t:
        return "village", f
    if f["built_frac"] > mega_t:
        return "megacity", f
    if (f["built_frac"] > informal_built_t
            and f["road_per_built"] < informal_ratio_t):
        return "informal", f
    if f["slope_mean"] >= slope_t:
        return "town", f
    if f["built_frac"] > built_t or f["road_frac"] > road_t:
        return "urban", f
    return "town", f


def route_available(name, available):
    """First expert in (name, *FALLBACK[name]) that is actually loaded.
    Raises if none is -- better a loud failure than a wrong model."""
    for cand in (name,) + FALLBACK.get(name, ()):
        if cand in available:
            return cand
    raise KeyError(f"no expert available for '{name}' "
                   f"(loaded: {sorted(available)})")


class MoE:
    """Holds one Diffusion wrapper per expert; delegates sampling.

    experts: dict name -> model.Diffusion, any non-empty subset of
    EXPERT_NAMES. All must share the cond format. Mixed batches are
    split by window and sampled per expert, preserving input order.
    last_log records (requested, used, features) per window, so both
    the routing decision and any fallback are inspectable after the
    fact.
    """

    def __init__(self, experts, slope_t=SLOPE_T, built_t=BUILT_T,
                 road_t=ROAD_T, village_t=VILLAGE_T, mega_t=MEGA_T,
                 informal_built_t=INFORMAL_BUILT_T,
                 informal_ratio_t=INFORMAL_RATIO_T):
        bad = set(experts) - set(EXPERT_NAMES)
        if bad or not experts:
            raise ValueError(f"experts must be a non-empty subset of "
                             f"{EXPERT_NAMES}, got {sorted(experts)}")
        self.experts = experts
        self.t = (slope_t, built_t, road_t, village_t, mega_t,
                  informal_built_t, informal_ratio_t)
        self.last_log = []

    def sample_ddim(self, cond, steps=50, eta=0.0, generator=None):
        import torch
        self.last_log = []
        used_names = []
        for i in range(cond.shape[0]):
            name, f = route(cond[i].detach().cpu().numpy(), *self.t)
            used = route_available(name, self.experts)
            used_names.append(used)
            self.last_log.append((name, used, f))
        out = [None] * len(used_names)
        for name in EXPERT_NAMES:
            idx = [i for i, n in enumerate(used_names) if n == name]
            if not idx:
                continue
            xs = self.experts[name].sample_ddim(
                cond[idx], steps=steps, eta=eta, generator=generator)
            for j, i in enumerate(idx):
                out[i] = xs[j]
        return torch.stack(out)


def load_moe(town_ckpt=None, urban_ckpt=None, device="cuda", cond_ch=4,
             out_ch=3, ckpts=None):
    """Build a MoE from checkpoint paths (EMA weights).

    New style: load_moe(ckpts={"village": p, "town": p, ...}) with any
    subset of EXPERT_NAMES. Old style, kept so v4 callers keep working:
    load_moe(town_path, urban_path).
    """
    import torch
    from model import UNet, Diffusion
    if ckpts is None:
        ckpts = {}
        if town_ckpt:
            ckpts["town"] = town_ckpt
        if urban_ckpt:
            ckpts["urban"] = urban_ckpt
    if not ckpts:
        raise ValueError("no checkpoints given")
    experts = {}
    for name, path in ckpts.items():
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

    cases = {
        "village": fake_cond(slope=0.10, built=0.005, road=0.005),
        "town": fake_cond(slope=0.40, built=0.030, road=0.012),
        "urban": fake_cond(slope=0.05, built=0.120, road=0.050),
        "informal": fake_cond(slope=0.08, built=0.150, road=0.028),
        "megacity": fake_cond(slope=0.06, built=0.300, road=0.060),
    }
    for want, cond in cases.items():
        got, f = route(cond)
        assert got == want, (want, got, f)
        print(f"route {want} ok "
              + " ".join(f"{k}={v:.3f}" for k, v in f.items()))

    # water channel must not change routing
    with_water = np.concatenate(
        [cases["town"], np.ones((1, G, G), np.float32)])
    assert route(with_water)[0] == "town"
    print("5th channel ignored by router ok")

    # fallback: a two-expert deployment behaves like v4
    assert route_available("informal", {"town", "urban"}) == "urban"
    assert route_available("village", {"town", "urban"}) == "town"
    assert route_available("megacity", {"town", "urban"}) == "urban"
    try:
        route_available("village", set())
        raise AssertionError("empty expert set should have raised")
    except KeyError:
        print("fallback chains + empty-set failure ok")

    class _FakeDiff:
        def __init__(self, tag):
            self.tag = tag

        def sample_ddim(self, cond, steps=50, eta=0.0, generator=None):
            import torch
            b = cond.shape[0]
            return torch.full((b, 3, G, G), float(self.tag))

    try:
        import torch
        tags = {n: i + 1 for i, n in enumerate(EXPERT_NAMES)}
        moe = MoE({n: _FakeDiff(t) for n, t in tags.items()})
        order = ["informal", "village", "megacity", "town", "urban"]
        cond = torch.from_numpy(np.stack([cases[n] for n in order]))
        x = moe.sample_ddim(cond, steps=4)
        assert x.shape == (5, 3, G, G)
        for i, n in enumerate(order):
            assert float(x[i, 0, 0, 0]) == tags[n], (i, n)
        assert [u for _, u, _ in moe.last_log] == order
        # partial deployment: same batch, only town+urban loaded
        moe2 = MoE({"town": _FakeDiff(2), "urban": _FakeDiff(3)})
        x2 = moe2.sample_ddim(cond, steps=4)
        used = [u for _, u, _ in moe2.last_log]
        assert used == ["urban", "town", "urban", "town", "urban"], used
        req = [r for r, _, _ in moe2.last_log]
        assert req == order, req
        print("moe order-preserving dispatch + fallback dispatch ok")
    except ImportError:
        print("torch unavailable: dispatch test skipped (router tested)")
    print("all moe self-tests passed")
