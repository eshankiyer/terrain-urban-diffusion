"""Generate the v3 Colab notebook JSON. Run: python build_notebook_v3.py"""
import json

REPO_URL = "https://github.com/eshankiyer/terrain-urban-diffusion.git"

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "outputs": [],
                  "execution_count": None,
                  "source": text.splitlines(keepends=True)})


md("""# v3: typed zones + generated local centres + plan-style rendering
The diffusion target gains a third channel (amenity density), so the model
PROPOSES local centres for new growth. A post-hoc classifier types new
development into residential / commercial / industrial / institutional,
and results render as legible planning maps (hillshade, zone colours,
legend, scale bar) instead of raw rasters. Trains 300 epochs.
**Runtime: GPU (T4 is fine).**""")

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

md("""## 1. Build the v3 dataset (~30-60 min fresh; resumable)
GHSL temporal pairs + one extra Overpass query per town for landuse
polygons and amenity points. Interrupted? Just re-run this cell.""")
code("""import os
os.makedirs("data", exist_ok=True)
from towns import TOWNS
from data_v3 import build_dataset_v3
n = build_dataset_v3(TOWNS, "data/dataset_v3.npz", cache_dir="data/ghsl_cache",
                     town_cache_dir="data/town_cache_v3")
print("v3 samples:", n)""")

md("""## 2. Train, 300 epochs (~25-30 min on T4)
train.py infers cond_ch=4 / out_ch=3 from the dataset automatically.""")
code("""!python src/train.py --data data/dataset_v3.npz --out runs/v3 --epochs 300 --batch 16""")

md("""## 3. Zone classifier (leave-one-town-out macro-F1)""")
code("""import pickle
from zones import train_zone_classifier
zclf, f1 = train_zone_classifier("data/town_cache_v3")
print(f"mean leave-one-town-out macro-F1: {f1:.3f}")
with open("runs/v3/zone_clf.pkl", "wb") as f:
    pickle.dump(zclf, f)""")

md("""## 4. Bike-lane classifier (same stage as v2)""")
code("""import pickle
from bikelanes import train_classifier
bclf, auc = train_classifier()
print(f"mean leave-one-town-out AUC: {auc:.3f}")
with open("runs/v3/bike_clf.pkl", "wb") as f:
    pickle.dump(bclf, f)""")

md("""## 5. Generate futures on a held-out town, rank, type zones, RENDER
Best-of-16 with the 11-metric scorecard; each of the top 3 futures is
drawn as a planning map: grey = existing town, zone colours = typed new
growth, red stars = the model's proposed local centres.""")
code("""import numpy as np, torch, matplotlib.pyplot as plt, os
from towns import EVAL_TOWNS
from data import fetch_elevation, fetch_osm, rasterize_osm, binary_dilate, slope_from_elevation
from data_v2 import sample_density, FOOTPRINT_THR
from data_v3 import fetch_env_v3, amenity_density, zone_raster
from environment import fetch_environment
from model import UNet, Diffusion
from zones import assign_zones
from render import render_plan
import sustainability as sus

os.makedirs("runs/v3/plans", exist_ok=True)
name, cc, lat, lon, region = EVAL_TOWNS[0]
elev = fetch_elevation(lat, lon)
roads_now, _ = rasterize_osm(fetch_osm(lat, lon), lat, lon)
d_now = sample_density(lat, lon, 2020, cache_dir="data/ghsl_cache")
env = fetch_environment(lat, lon, elev)
env_v3 = fetch_env_v3(lat, lon)
amen_now = amenity_density(env_v3, lat, lon)

ez = np.clip((elev - elev.mean()) / (elev.std() + 1e-6), -3, 3) / 3.0
sl = np.clip(slope_from_elevation(elev) / 30.0, 0, 1)
foot = binary_dilate((d_now > FOOTPRINT_THR).astype(np.uint8), 2)
cond_np = np.stack([ez, sl, d_now.astype(np.float32),
                    (roads_now * foot).astype(np.float32)]).astype(np.float32)

ckpt = torch.load("runs/v3/ckpt.pt", map_location="cuda")
net = UNet(cond_ch=4, out_ch=3).cuda()
net.load_state_dict(ckpt["ema"])
diff = Diffusion(net, device="cuda")

N = 16
cond = torch.from_numpy(cond_np)[None].repeat(N, 1, 1, 1).cuda()
with torch.no_grad():
    out = diff.sample_ddim(cond, steps=50).cpu().numpy()

cands, amen_props = [], []
for i in range(N):
    new_roads = (out[i, 0] > 0.0).astype(np.uint8)
    dens_new = np.clip((out[i, 1] + 1) / 2, 0, 1)
    amen_props.append(np.clip((out[i, 2] + 1) / 2, 0, 1))
    cands.append((np.maximum(roads_now, new_roads), dens_new))

order, scores = sus.rank_samples(cands, d_now, elev, env=env,
                                 roads0=roads_now)
print(f"{name}: best {scores[order[0]][0]:.0f}, worst {scores[order[-1]][0]:.0f}")

render_plan(elev, roads_now, d_now, green=env["green0"], water=env["water"],
            title=f"{name} today", path=f"runs/v3/plans/{name}_today.png")
for rank, i in enumerate(order[:3]):
    roads_all, dens_new = cands[i]
    amen_combined = np.maximum(amen_now, amen_props[i])
    z = assign_zones(dens_new, roads_all, elev, amen_combined, zclf, d0=d_now)
    render_plan(elev, roads_all, np.maximum(dens_new, d_now), zones=z,
                green=env["green0"], water=env["water"],
                amen_proposed=amen_props[i], existing_dens=d_now,
                title=f"{name} future #{rank+1} (score {scores[i][0]:.0f})",
                path=f"runs/v3/plans/{name}_future{rank+1}.png")
print("plans written to runs/v3/plans/")

from IPython.display import Image as _Img, display
for fn in sorted(os.listdir("runs/v3/plans")):
    display(_Img(f"runs/v3/plans/{fn}", width=560))""")

md("""## 6. Bring your own plan
Drop a `plan.png` into /content via the file browser (black lines = roads,
red/orange = built, green = keep free, white = empty) and re-run this cell
to have the model CONTINUE your plan. Without an upload it demos on a
built-in sketch. Terrain is procedural here; pass a real DEM for a site.""")
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
out_p, cands_p = plan_import.extend_plan(diff, cond_p, n=8)
d0_p = cond_p[2]
order_p, scores_p = sus.rank_samples(cands_p, d0_p, elev_p, roads0=roads_p)
os.makedirs("runs/v3/plans", exist_ok=True)
render_plan(elev_p, roads_p, d0_p, title="your plan (imported)",
            path="runs/v3/plans/yourplan_input.png")
for rank, i in enumerate(order_p[:2]):
    roads_all, dens_new = cands_p[i]
    amen_i = np.clip((out_p[i, 2] + 1) / 2, 0, 1) if out_p.shape[1] > 2 else \
        np.zeros_like(dens_new)
    z = assign_zones(dens_new, roads_all, elev_p, amen_i, zclf, d0=d0_p)
    render_plan(elev_p, roads_all, np.maximum(dens_new, d0_p), zones=z,
                amen_proposed=amen_i, existing_dens=d0_p,
                title=f"continuation #{rank+1} (score {scores_p[i][0]:.0f})",
                path=f"runs/v3/plans/yourplan_future{rank+1}.png")
from IPython.display import Image as _I, display
for fn in sorted(f for f in os.listdir("runs/v3/plans") if f.startswith("yourplan")):
    display(_I(f"runs/v3/plans/{fn}", width=520))""")

md("""## 7. Export ONNX for the browser demo (GitHub Pages)
Exports the EMA UNet so the web tool at docs/ can run DDIM sampling
in-browser with onnxruntime-web. ~52 MB fp32, under GitHub's file limit.""")
code("""import torch
class EpsWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x, cond, t):
        return self.m(x, cond, t)

net_cpu = UNet(cond_ch=4, out_ch=3)
net_cpu.load_state_dict(ckpt["ema"])
net_cpu.eval()
wrap = EpsWrapper(net_cpu)
dummy = (torch.randn(1, 3, 128, 128), torch.randn(1, 4, 128, 128),
         torch.tensor([500], dtype=torch.int64))
torch.onnx.export(wrap, dummy, "runs/v3/model.onnx", opset_version=17,
                  dynamo=False,  # legacy exporter: decomposes 3D attention
                  input_names=["x", "cond", "t"], output_names=["eps"])
import os
print("model.onnx:", os.path.getsize("runs/v3/model.onnx") / 1e6, "MB")""")

md("""## 8. Pack results for download""")
code("""!zip -qr v3_results.zip runs/v3
from google.colab import files
files.download("v3_results.zip")""")

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"gpuType": "T4", "provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

with open("v3_zoned_towns_colab.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("wrote v3_zoned_towns_colab.ipynb")
