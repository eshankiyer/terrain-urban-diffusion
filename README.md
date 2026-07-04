# Terrain-Conditioned Diffusion for Small-Town Expansion

A conditional denoising diffusion model that generates street-level urban
expansion layouts (roads + built-up area) conditioned on terrain (elevation
and slope) and the existing settlement footprint. Trained on ~60 small towns
in hilly terrain across Europe and Asia, using OpenStreetMap and the AWS
Terrain Tiles open elevation dataset.

Two modes from one model:

- **Real mode** — feed a real town's elevation window and current footprint;
  the model proposes a plausible expansion ring.
- **Sandbox mode** — feed any heightmap (fractal, hand-drawn, or exported
  from a game map editor) and a seed settlement; loop the model so each
  output becomes the next input, growing a city iteratively.

## Quick start

```bash
pip install -r requirements.txt

# 1. build the dataset (downloads OSM + terrain; ~30-60 min, be polite to Overpass)
python src/data.py --out data/dataset.npz
python src/data.py --out data/eval.npz --eval

# 2. train (a few hours on a T4; use the Colab notebook for free GPU)
python src/train.py --data data/dataset.npz --out runs/default --epochs 300

# 3. sample
python src/sample.py --ckpt runs/default/ckpt.pt --mode sandbox --rounds 4
python src/sample.py --ckpt runs/default/ckpt.pt --mode real --lat 43.353 --lon 12.578

# 4. evaluate vs. slope-blind baseline
python src/evaluate.py --ckpt runs/default/ckpt.pt --data data/eval.npz
```

Or open `notebooks/terrain_urban_diffusion_colab.ipynb` in Google Colab
(GPU runtime) — it runs the whole pipeline end to end.

## How it works

Each training sample is a 128×128 raster window (~1.9 km at 15 m/px) centred
on a town. Conditioning channels: z-scored elevation, slope, "core" built-up
mask, "core" road raster. Target channels: ring roads and ring built-up area.
The core/ring split uses concentric erosion of today's footprint as a proxy
for historical growth (see the paper's limitations section). A ~13M-parameter
U-Net with self-attention predicts noise under a cosine schedule; sampling
uses DDIM.

Evaluation compares generated rings to real rings on held-out towns:
slope-occupancy KS distance, road connectivity to the core network,
built-up contiguity, and expansion volume ratio, against a slope-blind
random-scatter baseline.

## Repository layout

```
src/towns.py      curated town lists (train + held-out eval)
src/data.py       download + rasterize + dataset build (numpy/PIL only)
src/model.py      U-Net, diffusion, EMA
src/train.py      training loop
src/sample.py     real mode + sandbox (iterative growth) mode
src/metrics.py    evaluation metrics + random baseline
src/evaluate.py   held-out evaluation script
notebooks/        Colab end-to-end notebook
paper/            LaTeX source of the paper
```

## Data sources & licenses

- Road/building/land-use vectors: © OpenStreetMap contributors (ODbL).
- Elevation: AWS Terrain Tiles (Mapzen terrarium tiles; various public
  sources incl. SRTM). No API key required.
