"""Generate the v2 Colab notebook JSON. Run: python build_notebook_v2.py"""
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


md("""# v2: real densification (GHSL) + sustainable 15-minute-town sampling
Trains on observed 1980-2020 built-up change instead of the erosion proxy,
then samples many futures per town and keeps the most sustainable ones
(15-minute walk coverage, infill share, land efficiency, circuity, earthwork).
Also trains the slope-aware bike-lane edge classifier on Dutch/Danish/German
towns. **Runtime: GPU (T4 is fine).** GHSL tiles are cached (~30-45 MB each).
""")

code(f"""import torch, sys
assert torch.cuda.is_available(), "Switch runtime to GPU: Runtime > Change runtime type"
print(torch.cuda.get_device_name(0))
!git clone -b v2 {REPO_URL} /content/repo 2>/dev/null || (cd /content/repo && git fetch && git checkout v2 && git pull)
%cd /content/repo
!pip -q install -r requirements.txt tifffile imagecodecs zarr scikit-image scikit-learn networkx scipy
sys.path.insert(0, "src")""")

md("""## 0. Smoke test (offline, <1 min) — must pass before the long run""")
code("""!python src/smoke_test_v2.py""")

md("""## 1. Build the v2 dataset from GHSL temporal pairs (~30-60 min)
One sample per town per consecutive epoch pair with observed growth.""")
code("""import os
os.makedirs("data", exist_ok=True)
from towns import TOWNS
from data_v2 import build_dataset_v2
n = build_dataset_v2(TOWNS, "data/dataset_v2.npz", cache_dir="data/ghsl_cache",
                     town_cache_dir="data/town_cache_v2")
print("v2 samples:", n)
# Resumable: if this cell is interrupted, just re-run it -- finished towns
# reload from data/town_cache_v2 in milliseconds.""")

md("""## 2. Train (~1.5 h on T4) — same model/trainer as v1, new data""")
code("""!python src/train.py --data data/dataset_v2.npz --out runs/v2 --epochs 120 --batch 16""")

md("""## 3. Bike-lane edge classifier (leave-one-town-out AUC)""")
code("""import pickle
from bikelanes import train_classifier
clf, auc = train_classifier()
print(f"mean leave-one-town-out AUC: {auc:.3f}")
with open("runs/v2/bike_clf.pkl", "wb") as f:
    pickle.dump(clf, f)""")

md("""## 4. Sustainable best-of-N sampling on a held-out town
Sample 16 futures, rank with the 11-metric scorecard (real amenities,
greenspace, flood/landslide proxies, congestion, access equity), show the
top 3 with walk-time maps and bike-lane annotations.""")
code("""import numpy as np, torch, matplotlib.pyplot as plt
from towns import EVAL_TOWNS
from data import binary_dilate
from data_v2 import sample_density, make_sample_v2, FOOTPRINT_THR
from data import fetch_elevation, fetch_osm, rasterize_osm
from model import UNet, Diffusion
import sustainability as sus
from bikelanes import assign_bike_lanes

name, cc, lat, lon, region = EVAL_TOWNS[0]
elev = fetch_elevation(lat, lon)
roads_now, _ = rasterize_osm(fetch_osm(lat, lon), lat, lon)
d_now = sample_density(lat, lon, 2020, cache_dir="data/ghsl_cache")
from environment import fetch_environment
env = fetch_environment(lat, lon, elev)
print("amenity categories present:",
      [c for c, p in env["amenities"].items() if p])

ez = np.clip((elev - elev.mean()) / (elev.std() + 1e-6), -3, 3) / 3.0
from data import slope_from_elevation
sl = np.clip(slope_from_elevation(elev) / 30.0, 0, 1)
foot = binary_dilate((d_now > FOOTPRINT_THR).astype(np.uint8), 2)
cond_np = np.stack([ez, sl, d_now.astype(np.float32),
                    (roads_now * foot).astype(np.float32)]).astype(np.float32)

ckpt = torch.load("runs/v2/ckpt.pt", map_location="cuda")
net = UNet(cond_ch=4, out_ch=2).cuda()
net.load_state_dict(ckpt["ema"] if "ema" in ckpt else ckpt["model"])
diff = Diffusion(net, device="cuda")

N = 16
cond = torch.from_numpy(cond_np)[None].repeat(N, 1, 1, 1).cuda()
with torch.no_grad():
    out = diff.sample_ddim(cond, steps=50).cpu().numpy()

cands = []
for i in range(N):
    new_roads = (out[i, 0] > 0.0).astype(np.uint8)
    dens_new = np.clip((out[i, 1] + 1) / 2, 0, 1)
    roads_all = np.maximum(roads_now, new_roads)
    cands.append((roads_all, dens_new))

order, scores = sus.rank_samples(cands, d_now, elev, env=env,
                                 roads0=roads_now)
print(f"{name}: best {scores[order[0]][0]:.0f}, "
      f"median {scores[order[N // 2]][0]:.0f}, worst {scores[order[-1]][0]:.0f}")
for i in order[:3]:
    print(i, {k: round(v, 2) for k, v in scores[i][1].items()
              if not k.startswith("_")})

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
for row, i in enumerate(order[:3]):
    roads_all, dens_new = cands[i]
    cov, minutes = sus.fifteen_min_coverage(np.maximum(dens_new, d_now), roads_all)
    _, _, bike = assign_bike_lanes(roads_all, elev, clf)
    axes[row, 0].imshow(np.maximum(dens_new, d_now), cmap="magma"); axes[row, 0].set_title(f"#{row+1} density (score {scores[i][0]:.0f})")
    axes[row, 1].imshow(np.where(np.isfinite(minutes), minutes, np.nan), cmap="viridis_r"); axes[row, 1].set_title(f"walk minutes (15-min cov {cov:.0%})")
    axes[row, 2].imshow(roads_all, cmap="gray"); axes[row, 2].imshow(np.ma.masked_where(bike < 0.5, bike), cmap="autumn", alpha=0.9); axes[row, 2].set_title("roads + bike-lane prob>0.5")
for ax in axes.ravel():
    ax.axis("off")
plt.tight_layout(); plt.savefig("runs/v2/sustainable_topk.png", dpi=150); plt.show()""")

md("""## 5. Pack results for download""")
code("""!zip -qr v2_results.zip runs/v2 -x "runs/v2/ghsl_cache/*"
from google.colab import files
files.download("v2_results.zip")""")

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"gpuType": "T4", "provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

with open("v2_sustainable_towns_colab.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("wrote v2_sustainable_towns_colab.ipynb")
