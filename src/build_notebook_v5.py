"""Generate notebooks/v5_global_moe_colab.ipynb. Run: python3 build_notebook_v5.py"""
import json

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "outputs": [],
                  "execution_count": None,
                  "source": text.splitlines(keepends=True)})


md("""# v5: five-expert global MoE + amenity-scale fix
Two changes, one run. First, the two-expert mixture (hilly towns / flat
fringes) becomes five: **village**, **town**, **urban** (now with AU/NZ/CA
windows), **informal** (Lima, Nairobi, Ikorodu, Savar...), and **megacity**
(Delhi first among them; Narela is held out for eval). The router is still
a hard, auditable rule (`moe.py`), extended with one new feature:
roads-per-built, which separates unplanned frontiers from planned ones.
Scale needs no sixth expert -- built fraction already tells a village
window from a Delhi one.

Second, the amenity channel moves to an absolute scale. The old per-window
max normalization turned one isolated suburban school into a full-strength
"amenity core", which is why the zone classifier painted most of Middleburg
Heights, OH institutional. Cell 5b reproduces that window and prints the
before/after class counts.

**Runtime: budget 5-6 h on a T4.** Dataset builds ~1.5-2.5 h fresh (150+
windows through one polite Overpass semaphore -- and the amenity rescale
means NO old cache can be reused), five 300-epoch trainings ~25-30 min
each, classifiers + sampling + export the rest. Fine to run across two
sessions: everything resumes from the Drive cache.""")

code("""import torch, sys
assert torch.cuda.is_available(), "Switch runtime to GPU: Runtime > Change runtime type"
print(torch.cuda.get_device_name(0))
!git clone -b v3 https://github.com/eshankiyer/terrain-urban-diffusion.git /content/repo 2>/dev/null || (cd /content/repo && git fetch && git checkout v3 && git pull)
%cd /content/repo
!pip -q install -r requirements.txt tifffile "imagecodecs==2024.9.22" "numpy==2.0.2" zarr scikit-image scikit-learn networkx scipy onnx onnxscript
sys.path.insert(0, "src")""")

md("""## 0a. Persist to Google Drive (survives disconnects & page refreshes)
Same mechanism as v4, new folder: `terrain-urban-diffusion-v5`. The v4
folder is deliberately NOT reused -- its town caches carry the old amenity
normalization and would poison the classifier training. Safe to re-run.""")

code("""from google.colab import drive
drive.mount('/content/drive')

import os, shutil

PROJECT_DIR = '/content/drive/MyDrive/terrain-urban-diffusion-v5'
os.makedirs(PROJECT_DIR, exist_ok=True)

for sub in ('data', 'runs'):
    local = f'/content/repo/{sub}'
    remote = f'{PROJECT_DIR}/{sub}'
    if os.path.islink(local):
        continue
    os.makedirs(remote, exist_ok=True)
    if os.path.isdir(local):
        shutil.copytree(local, remote, dirs_exist_ok=True)
        shutil.rmtree(local)
    os.symlink(remote, local)

print('persisted data/ and runs/ to', PROJECT_DIR)""")

md("""## 0b. Smoke tests (offline, ~1 min) -- must pass before the long run
Now also runs the moe.py router self-tests (five-way routing, fallback
chains) and the zones.py per-class margin tests.""")

code("""!python src/smoke_test_v2.py
!python src/smoke_test_v3.py
!python src/moe.py
!python src/zones.py""")

DATASETS = [
    ("1a", "TOWN", "from towns import TOWNS",
     "TOWNS", "ds_town.npz",
     "Hilly small towns from `towns.py`, the original regime."),
    ("1b", "URBAN", "from towns_urban import URBAN_TOWNS\n"
     "from towns_us import US_TOWNS\n"
     "from towns_global import AUS_NZ_CA_TOWNS",
     "URBAN_TOWNS + US_TOWNS + AUS_NZ_CA_TOWNS", "ds_urban.npz",
     "Planned flat fringes: EU + US + the new AU/NZ/CA windows. "
     "Australian suburbs are US subdivisions with different trees, so "
     "they extend this expert instead of getting their own."),
    ("1c", "VILLAGE", "from towns_global import VILLAGES",
     "VILLAGES", "ds_village.npz",
     "Villages that demonstrably grew 1980-2020 (tourism, road access, "
     "spillover). Ones that stayed frozen would be dropped by the growth "
     "filter anyway."),
    ("1d", "INFORMAL", "from towns_global import INFORMAL_TOWNS",
     "INFORMAL_TOWNS", "ds_informal.npz",
     "Unplanned periurban frontiers. OSM road mapping is thin in some of "
     "these; the growth filter eats bad windows, and thin mapped roads "
     "are part of what this expert should learn."),
    ("1e", "MEGACITY", "from towns_global import MEGACITY_TOWNS",
     "MEGACITY_TOWNS", "ds_megacity.npz",
     "Saturated growth edges of the largest cities. Four Delhi-region "
     "windows train; Narela is held out. The two Chinese windows may be "
     "dropped by the filter (thin OSM buildings) -- that is expected."),
]

for tag, name, imports, expr, npz, blurb in DATASETS:
    md(f"## {tag}. Dataset: {name} expert (resumable)\n{blurb}\n"
       "All caches live in `town_cache_v5` -- v3/v4 caches must not be "
       "reused (amenity rescale).")
    code(f"""import os
os.makedirs("data", exist_ok=True)
{imports}
from data_v3 import build_dataset_v3
n = build_dataset_v3({expr}, "data/{npz}", cache_dir="data/ghsl_cache",
                     town_cache_dir="data/town_cache_v5", with_water=True)
print("{name.lower()}-expert samples:", n)""")

for i, tag in enumerate(("village", "town", "urban", "informal",
                         "megacity")):
    md(f"## 2{'abcde'[i]}. Train the {tag.upper()} expert, 300 epochs "
       "(~25-30 min on T4)")
    code(f"!python src/train.py --data data/ds_{tag}.npz "
         f"--out runs/v5_{tag} --epochs 300 --batch 16")

md("""## 3. Zone classifier (leave-one-town-out macro-F1)
Trained on the combined v5 caches, which now include sparse suburban,
informal, and megacity windows -- so "isolated school on a parking lot"
is finally IN the training distribution instead of an out-of-distribution
surprise.""")

code("""import os, pickle
from zones import train_zone_classifier
os.makedirs("runs/v5", exist_ok=True)
zclf, f1 = train_zone_classifier("data/town_cache_v5")
print(f"mean leave-one-town-out macro-F1: {f1:.3f}")
with open("runs/v5/zone_clf.pkl", "wb") as f:
    pickle.dump(zclf, f)""")

md("## 4. Bike-lane classifier (same stage as v2)")

code("""import pickle
from bikelanes import train_classifier
bclf, auc = train_classifier()
print(f"mean leave-one-town-out AUC: {auc:.3f}")
with open("runs/v5/bike_clf.pkl", "wb") as f:
    pickle.dump(bclf, f)""")

md("""## 5. MoE sampling across the whole scale: village to Delhi
Five windows, one per expert. Wengen (village), Gubbio (town), and
Antananarivo W / Delhi-Narela (informal / megacity) are held out; Des
Moines is a TRAINING window for the urban expert, kept as a routing
sanity check, not an evaluation. The printout shows requested vs used
expert -- with all five loaded they should match.""")

code("""import numpy as np, torch, os
from data import fetch_elevation, fetch_osm, rasterize_osm, binary_dilate, slope_from_elevation
from data_v2 import sample_density, FOOTPRINT_THR
from data_v3 import fetch_env_v3, amenity_density, water_raster
from environment import fetch_environment
from moe import load_moe, route, route_available
from zones import assign_zones
from render import render_plan
import sustainability as sus

CKPTS = {t: f"runs/v5_{t}/ckpt.pt"
         for t in ("village", "town", "urban", "informal", "megacity")
         if os.path.exists(f"runs/v5_{t}/ckpt.pt")}
print("experts loaded:", sorted(CKPTS))
moe = load_moe(ckpts=CKPTS, cond_ch=5)
os.makedirs("runs/v5/plans", exist_ok=True)
WINDOWS = [("Wengen", "CH", 46.6050, 7.9220, "village"),
           ("Gubbio", "IT", 43.3530, 12.5780, "europe"),
           ("Des Moines NW fringe", "US", 41.6520, -93.7800, "us_flat"),
           ("Antananarivo W fringe", "MG", -18.9000, 47.4600, "informal"),
           ("Delhi N Narela", "IN", 28.8400, 77.0900, "megacity")]
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
    used = route_available(expert, set(CKPTS))
    print(f"{name}: routed to '{expert}' (using '{used}') "
          + " ".join(f"{k}={v:.3f}" for k, v in feats.items()))

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
                title=f"{name} today", path=f"runs/v5/plans/{safe}_today.png")
    for rank, i in enumerate(order[:2]):
        roads_all, dens_new = cands[i]
        amen_combined = np.maximum(amen_now, amen_props[i])
        z = assign_zones(dens_new, roads_all, elev, amen_combined, zclf, d0=d_now)
        render_plan(elev, roads_all, np.maximum(dens_new, d_now), zones=z,
                    green=env["green0"], water=env["water"],
                    amen_proposed=amen_props[i], existing_dens=d_now,
                    title=f"{name} future #{rank+1} (score {scores[i][0]:.0f})",
                    path=f"runs/v5/plans/{safe}_future{rank+1}.png")
print("plans written to runs/v5/plans/")

from IPython.display import Image as _Img, display
for fn in sorted(os.listdir("runs/v5/plans")):
    display(_Img(f"runs/v5/plans/{fn}", width=560))""")

md("""## 5b. Middleburg Heights validation: is the institutional bug fixed?
The reported failure: 697/1,099 solid-growth cells and 4,869/11,345
advisory cells came back institutional on a Cleveland suburb. This cell
re-runs advisory zoning on that exact window and on Gubbio (a dense
in-distribution check that must NOT regress), three ways:

* **old-norm control** -- a classifier trained on per-window-max amenity
  features, reconstructed from the v5 cache by renormalizing each town's
  amenity map to its own max. Close to the shipped v4 classifier, not
  byte-identical (saturated cores lose a little detail in the round
  trip; noted for honesty).
* **v5** -- the absolute-scale classifier from cell 3.
* **v5 + institutional margin** -- same, sweeping `class_margins={4: m}`.

What "fixed" looks like: the institutional count on Middleburg Heights
drops from a majority to a plausible minority under v5, while Gubbio's
distribution barely moves.""")

code("""import os, shutil, tempfile
import numpy as np
from data import fetch_elevation, fetch_osm, rasterize_osm
from data_v2 import sample_density
from data_v3 import fetch_env_v3, amenity_density
from zones import train_zone_classifier, zone_potential, ZONE_NAMES

VAL = [("Middleburg Heights OH", 41.3584, -81.8039),
       ("Gubbio", 43.3530, 12.5780)]

def counts(z):
    c = np.bincount(z.ravel(), minlength=7)
    total = int((z > 0).sum())
    body = ", ".join(f"{ZONE_NAMES.get(k, k)}={c[k]}"
                     for k in sorted(ZONE_NAMES) if c[k])
    return f"typed={total} abstained={c[0]} | {body}"

# control: retrain on per-window-max features reconstructed from cache
tmp = tempfile.mkdtemp()
for fn in os.listdir("data/town_cache_v5"):
    if not fn.endswith(".npz"):
        continue
    d = dict(np.load(os.path.join("data/town_cache_v5", fn),
                     allow_pickle=True))
    a = d["amenity"]
    d["amenity"] = (a / a.max() if a.max() > 0 else a).astype(np.float32)
    np.savez_compressed(os.path.join(tmp, fn), **d)
zclf_old, f1_old = train_zone_classifier(tmp, verbose=False)
shutil.rmtree(tmp, ignore_errors=True)
print(f"control (old-norm) macro-F1 {f1_old:.3f} vs v5 {f1:.3f}")

for name, lat, lon in VAL:
    elev = fetch_elevation(lat, lon)
    roads, _ = rasterize_osm(fetch_osm(lat, lon), lat, lon)
    d_now = sample_density(lat, lon, 2020, cache_dir="data/ghsl_cache")
    amen = amenity_density(fetch_env_v3(lat, lon), lat, lon)
    amen_old = (amen / amen.max() if amen.max() > 0 else amen)

    print(f"\\n{name}")
    z = zone_potential(d_now, roads, elev, amen_old, zclf_old)
    print(f"  old-norm control : {counts(z)}")
    z = zone_potential(d_now, roads, elev, amen, zclf)
    print(f"  v5 absolute scale: {counts(z)}")
    for m in (0.5, 1.0, 2.0):
        z = zone_potential(d_now, roads, elev, amen, zclf,
                           class_margins={4: m})
        print(f"  v5 + inst margin {m:.1f}: {counts(z)}")""")

md("""## 6. Bring your own plan
Drop a `plan.png` into /content (black lines = roads, red/orange = built,
green = keep free, white = empty) and re-run; the router picks the expert.
Without an upload it demos on a built-in sketch. A sketch has no water
layer, so the water channel is zeros.""")

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
cond_p = np.concatenate([cond_p, np.zeros((1,) + cond_p.shape[1:],
                                          np.float32)])
out_p, cands_p = plan_import.extend_plan(moe, cond_p, n=8)
d0_p = cond_p[2]
order_p, scores_p = sus.rank_samples(cands_p, d0_p, elev_p, roads0=roads_p)
os.makedirs("runs/v5/plans", exist_ok=True)
render_plan(elev_p, roads_p, d0_p, title="your plan (imported)",
            path="runs/v5/plans/yourplan_input.png")
for rank, i in enumerate(order_p[:2]):
    roads_all, dens_new = cands_p[i]
    amen_i = np.clip((out_p[i, 2] + 1) / 2, 0, 1) if out_p.shape[1] > 2 else \\
        np.zeros_like(dens_new)
    z = assign_zones(dens_new, roads_all, elev_p, amen_i, zclf, d0=d0_p)
    render_plan(elev_p, roads_all, np.maximum(dens_new, d0_p), zones=z,
                amen_proposed=amen_i, existing_dens=d0_p,
                title=f"continuation #{rank+1} (score {scores_p[i][0]:.0f})",
                path=f"runs/v5/plans/yourplan_future{rank+1}.png")
from IPython.display import Image as _I, display
for fn in sorted(f for f in os.listdir("runs/v5/plans") if f.startswith("yourplan")):
    display(_I(f"runs/v5/plans/{fn}", width=520))""")

md("""## 7. Export ALL experts to ONNX for the browser demo
~52 MB fp32 each, under GitHub's file limit. The site can ship any subset;
the JS router falls back exactly like `moe.FALLBACK`.""")

code("""import os, torch
from model import UNet

class EpsWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x, cond, t):
        return self.m(x, cond, t)

for tag in ("village", "town", "urban", "informal", "megacity"):
    ck_path = f"runs/v5_{tag}/ckpt.pt"
    if not os.path.exists(ck_path):
        print("skip", tag, "(no checkpoint)")
        continue
    ck = torch.load(ck_path, map_location="cpu")
    net_cpu = UNet(cond_ch=5, out_ch=3)
    net_cpu.load_state_dict(ck["ema"])
    net_cpu.eval()
    dummy = (torch.randn(1, 3, 128, 128), torch.randn(1, 5, 128, 128),
             torch.tensor([500], dtype=torch.int64))
    path = f"runs/v5_{tag}/model_{tag}.onnx"
    torch.onnx.export(EpsWrapper(net_cpu), dummy, path, opset_version=17,
                      dynamo=False,  # legacy exporter: decomposes 3D attention
                      input_names=["x", "cond", "t"], output_names=["eps"])
    print(path, os.path.getsize(path) / 1e6, "MB")""")

md("## 8. Pack results for download")

code("""!zip -qr v5_results.zip runs/v5 runs/v5_village runs/v5_town runs/v5_urban runs/v5_informal runs/v5_megacity
from google.colab import files
files.download("v5_results.zip")""")

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

with open("v5_global_moe_colab.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("wrote v5_global_moe_colab.ipynb,", len(cells), "cells")
