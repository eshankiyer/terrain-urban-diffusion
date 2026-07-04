"""Generate the v4 Colab notebook JSON. Run: python build_notebook_v3.py"""
import json
import os

REPO_URL = "https://github.com/eshankiyer/terrain-urban-diffusion.git"

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "outputs": [],
                  "execution_count": None,
                  "source": text.splitlines(keepends=True)})


md("""# v4: water conditioning + two-expert MoE (hard router)
The deployed model builds on lakes: cond had no water channel, so flat
water read like buildable valley floor. v4 appends a filled OSM water mask
as cond channel 5 (channels 0-3 unchanged -- the router depends on that
order) and forces targets to no-growth on water. Growth style now comes
from TWO experts sharing one architecture: **town** (hilly European towns)
and **urban** (flat metro fringes incl. 20 US windows), dispatched by an
auditable hard router on conditioning stats (`moe.py`) -- not a learned
gate, so every routing decision is inspectable.

**Runtime: ~2.5-3 h total on a T4.** Datasets ~40-70 min fresh (resumable),
2 x 300-epoch trainings ~25-30 min each, classifiers + sampling + export
the rest.""")

code(f"""import torch, sys
assert torch.cuda.is_available(), "Switch runtime to GPU: Runtime > Change runtime type"
print(torch.cuda.get_device_name(0))
!git clone -b v3 {REPO_URL} /content/repo 2>/dev/null || (cd /content/repo && git fetch && git checkout v3 && git pull)
%cd /content/repo
!pip -q install -r requirements.txt tifffile "imagecodecs==2024.9.22" "numpy==2.0.2" zarr scikit-image scikit-learn networkx scipy onnx onnxscript
sys.path.insert(0, "src")""")

md("""## 0. Smoke tests (offline, ~1 min) — must pass before the long run""")
code("""!python src/smoke_test_v2.py
!python src/smoke_test_v3.py""")

md("""## 1a. Dataset: TOWN expert (~30-60 min fresh; resumable)
Hilly small towns from `towns.py`. v4 cond format (5 channels, water
appended), so the per-town cache lives in `town_cache_v4` — old v3 caches
must not be reused. Interrupted? Just re-run this cell.""")
code("""import os
os.makedirs("data", exist_ok=True)
from towns import TOWNS
from data_v3 import build_dataset_v3
n = build_dataset_v3(TOWNS, "data/ds_town.npz", cache_dir="data/ghsl_cache",
                     town_cache_dir="data/town_cache_v4")
print("town-expert samples:", n)""")

md("""## 1b. Dataset: URBAN expert (~20-40 min fresh; resumable)
Flat metro fringes: European (`towns_urban.py`) + the 20 new US windows
(`towns_us.py`). Same GHSL and town caches, separate npz.""")
code("""from towns_urban import URBAN_TOWNS
from towns_us import US_TOWNS
from data_v3 import build_dataset_v3
n = build_dataset_v3(URBAN_TOWNS + US_TOWNS, "data/ds_urban.npz",
                     cache_dir="data/ghsl_cache",
                     town_cache_dir="data/town_cache_v4")
print("urban-expert samples:", n)""")

md("""## 2a. Train the TOWN expert, 300 epochs (~25-30 min on T4)
train.py infers cond_ch=5 / out_ch=3 from the dataset automatically.""")
code("""!python src/train.py --data data/ds_town.npz --out runs/v4_town --epochs 300 --batch 16""")

md("""## 2b. Train the URBAN expert, 300 epochs (~25-30 min on T4)""")
code("""!python src/train.py --data data/ds_urban.npz --out runs/v4_urban --epochs 300 --batch 16""")

md("""## 3. Zone classifier (leave-one-town-out macro-F1)
Trained on the combined v4 town caches (both experts' towns).""")
code("""import os, pickle
from zones import train_zone_classifier
os.makedirs("runs/v4", exist_ok=True)
zclf, f1 = train_zone_classifier("data/town_cache_v4")
print(f"mean leave-one-town-out macro-F1: {f1:.3f}")
with open("runs/v4/zone_clf.pkl", "wb") as f:
    pickle.dump(zclf, f)""")

md("""## 4. Bike-lane classifier (same stage as v2)""")
code("""import pickle
from bikelanes import train_classifier
bclf, auc = train_classifier()
print(f"mean leave-one-town-out AUC: {auc:.3f}")
with open("runs/v4/bike_clf.pkl", "wb") as f:
    pickle.dump(bclf, f)""")

md("""## 5. MoE sampling on two windows: hilly vs. US flat
Gubbio (held-out, hilly) should route to the TOWN expert; the Des Moines
NW fringe (flat gridiron) to the URBAN expert. Honesty note: Des Moines is
one of the urban expert's TRAINING windows — it is a routing/morphology
sanity check here, not a held-out evaluation. Best-of-8 with the 11-metric
scorecard; today + top-2 futures rendered per window.""")
code("""import numpy as np, torch, os
from towns import EVAL_TOWNS
from data import fetch_elevation, fetch_osm, rasterize_osm, binary_dilate, slope_from_elevation
from data_v2 import sample_density, FOOTPRINT_THR
from data_v3 import fetch_env_v3, amenity_density, water_raster
from environment import fetch_environment
from moe import load_moe, route
from zones import assign_zones
from render import render_plan
import sustainability as sus

moe = load_moe("runs/v4_town/ckpt.pt", "runs/v4_urban/ckpt.pt", cond_ch=5)
os.makedirs("runs/v4/plans", exist_ok=True)
WINDOWS = [EVAL_TOWNS[0],                                       # Gubbio, hilly
           ("Des Moines NW fringe", "US", 41.6520, -93.7800, "us_flat")]
N = 8
for name, cc, lat, lon, region in WINDOWS:
    safe = name.replace(" ", "_")
    elev = fetch_elevation(lat, lon)
    roads_now, _ = rasterize_osm(fetch_osm(lat, lon), lat, lon)
    d_now = sample_density(lat, lon, 2020, cache_dir="data/ghsl_cache")
    env = fetch_environment(lat, lon, elev)
    env_v3 = fetch_env_v3(lat, lon)
    amen_now = amenity_density(env_v3, lat, lon)
    water = water_raster(env_v3, lat, lon)

    ez = np.clip((elev - elev.mean()) / (elev.std() + 1e-6), -3, 3) / 3.0
    sl = np.clip(slope_from_elevation(elev) / 30.0, 0, 1)
    foot = binary_dilate((d_now > FOOTPRINT_THR).astype(np.uint8), 2)
    cond_np = np.stack([ez, sl, d_now.astype(np.float32),
                        (roads_now * foot).astype(np.float32),
                        water.astype(np.float32)]).astype(np.float32)

    expert, feats = route(cond_np)
    print(f"{name}: routed to '{expert}'  "
          + "  ".join(f"{k}={v:.3f}" for k, v in feats.items()))

    cond = torch.from_numpy(cond_np)[None].repeat(N, 1, 1, 1).cuda()
    with torch.no_grad():
        out = moe.sample_ddim(cond, steps=50).cpu().numpy()

    cands, amen_props = [], []
    for i in range(N):
        new_roads = (out[i, 0] > 0.0).astype(np.uint8)
        dens_new = np.clip((out[i, 1] + 1) / 2, 0, 1)
        amen_props.append(np.clip((out[i, 2] + 1) / 2, 0, 1))
        cands.append((np.maximum(roads_now, new_roads), dens_new))

    order, scores = sus.rank_samples(cands, d_now, elev, env=env,
                                     roads0=roads_now)
    print(f"  best {scores[order[0]][0]:.0f}, worst {scores[order[-1]][0]:.0f}")

    render_plan(elev, roads_now, d_now, green=env["green0"], water=env["water"],
                title=f"{name} today", path=f"runs/v4/plans/{safe}_today.png")
    for rank, i in enumerate(order[:2]):
        roads_all, dens_new = cands[i]
        amen_combined = np.maximum(amen_now, amen_props[i])
        z = assign_zones(dens_new, roads_all, elev, amen_combined, zclf, d0=d_now)
        render_plan(elev, roads_all, np.maximum(dens_new, d_now), zones=z,
                    green=env["green0"], water=env["water"],
                    amen_proposed=amen_props[i], existing_dens=d_now,
                    title=f"{name} future #{rank+1} (score {scores[i][0]:.0f})",
                    path=f"runs/v4/plans/{safe}_future{rank+1}.png")
print("plans written to runs/v4/plans/")

from IPython.display import Image as _Img, display
for fn in sorted(os.listdir("runs/v4/plans")):
    display(_Img(f"runs/v4/plans/{fn}", width=560))""")

md("""## 6. Bring your own plan
Drop a `plan.png` into /content via the file browser (black lines = roads,
red/orange = built, green = keep free, white = empty) and re-run this cell
to have the MoE CONTINUE your plan (the router picks the expert). Without
an upload it demos on a built-in sketch. Terrain is procedural here; a
sketch has no water layer, so the water channel is zeros.""")
code("""import os
import numpy as np
from PIL import Image
import plan_import
from zones import assign_zones
from render import render_plan
import sustainability as sus

src_img = (Image.open("/content/plan.png") if os.path.exists("/content/plan.png")
           else plan_import.demo_sketch())
roads_p, built_p = plan_import.load_plan_image(src_img)
cond_p, elev_p = plan_import.cond_from_plan(roads_p, built_p, seed=3)
# cond_from_plan returns the 4-channel v3 layout; v4 experts expect a 5th
# water channel. A drawn plan carries no water, so append zeros here.
cond_p = np.concatenate([cond_p, np.zeros((1,) + cond_p.shape[1:],
                                          np.float32)])
out_p, cands_p = plan_import.extend_plan(moe, cond_p, n=8)
d0_p = cond_p[2]
order_p, scores_p = sus.rank_samples(cands_p, d0_p, elev_p, roads0=roads_p)
os.makedirs("runs/v4/plans", exist_ok=True)
render_plan(elev_p, roads_p, d0_p, title="your plan (imported)",
            path="runs/v4/plans/yourplan_input.png")
for rank, i in enumerate(order_p[:2]):
    roads_all, dens_new = cands_p[i]
    amen_i = np.clip((out_p[i, 2] + 1) / 2, 0, 1) if out_p.shape[1] > 2 else \\
        np.zeros_like(dens_new)
    z = assign_zones(dens_new, roads_all, elev_p, amen_i, zclf, d0=d0_p)
    render_plan(elev_p, roads_all, np.maximum(dens_new, d0_p), zones=z,
                amen_proposed=amen_i, existing_dens=d0_p,
                title=f"continuation #{rank+1} (score {scores_p[i][0]:.0f})",
                path=f"runs/v4/plans/yourplan_future{rank+1}.png")
from IPython.display import Image as _I, display
for fn in sorted(f for f in os.listdir("runs/v4/plans") if f.startswith("yourplan")):
    display(_I(f"runs/v4/plans/{fn}", width=520))""")

md("""## 7. Export BOTH experts to ONNX for the browser demo
The web tool routes with the same hard rule, so it needs both UNets.
~52 MB fp32 each, under GitHub's file limit. Dummy cond is now
[1, 5, 128, 128].""")
code("""import os, torch
from model import UNet

class EpsWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x, cond, t):
        return self.m(x, cond, t)

for tag in ("town", "urban"):
    ck = torch.load(f"runs/v4_{tag}/ckpt.pt", map_location="cpu")
    net_cpu = UNet(cond_ch=5, out_ch=3)
    net_cpu.load_state_dict(ck["ema"])
    net_cpu.eval()
    dummy = (torch.randn(1, 3, 128, 128), torch.randn(1, 5, 128, 128),
             torch.tensor([500], dtype=torch.int64))
    path = f"runs/v4_{tag}/model_{tag}.onnx"
    torch.onnx.export(EpsWrapper(net_cpu), dummy, path, opset_version=17,
                      dynamo=False,  # legacy exporter: decomposes 3D attention
                      input_names=["x", "cond", "t"], output_names=["eps"])
    print(path, os.path.getsize(path) / 1e6, "MB")""")

md("""## 8. Pack results for download""")
code("""!zip -qr v4_results.zip runs/v4 runs/v4_town runs/v4_urban
from google.colab import files
files.download("v4_results.zip")""")

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"gpuType": "T4", "provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "v3_zoned_towns_colab.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {out}")
