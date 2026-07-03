"""Generate the Colab notebook JSON. Run: python build_notebook.py"""
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

md("""# Terrain-Conditioned Diffusion for Small-Town Expansion — end-to-end
Trains the model on real European/Asian towns (OpenStreetMap + AWS Terrain
Tiles), evaluates on held-out towns, and runs both sampling modes.
**Runtime: GPU (T4 is fine).** Full run ≈ 2–3 h; most of it is training.
""")

code(f"""import torch, sys
assert torch.cuda.is_available(), "Switch runtime to GPU: Runtime > Change runtime type"
print(torch.cuda.get_device_name(0))
!git clone {REPO_URL} repo 2>/dev/null || (cd repo && git pull)
%cd repo
!pip -q install -r requirements.txt
sys.path.insert(0, "src")""")

md("""## 0. Smoke test (~2 min) — catches bugs before the long run
Tiny synthetic dataset, 2 epochs, 1 sample. Must finish without errors.""")
code("""!python src/data.py --synthetic 4 --out data/smoke.npz
!python src/train.py --data data/smoke.npz --out runs/smoke --epochs 2 --batch 8 --ckpt_every 2
import torch, numpy as np
from sample import load_model, expand_once
d = np.load("data/smoke.npz", allow_pickle=True)
diff = load_model("runs/smoke/ckpt.pt", "cuda")
c = d["cond"][0]
r, b, raw = expand_once(diff, c[0]*300, c[2], c[3], steps=10, seed=0)
print("SMOKE OK — sample shape", raw.shape, "road px", int(r.sum()), "built px", int(b.sum()))""")

md("## 1. Build datasets (downloads OSM + terrain; ~30–50 min)")
code("""!python src/data.py --out data/dataset.npz
!python src/data.py --out data/eval.npz --eval
import numpy as np
d = np.load("data/dataset.npz", allow_pickle=True)
print("train samples:", d["cond"].shape, "eval:", np.load("data/eval.npz", allow_pickle=True)["cond"].shape)""")

md("## 2. Train")
code("""EPOCHS = 150   # ~1.5-2 h on T4; raise to 300 for the full run
!python src/train.py --data data/dataset.npz --out runs/colab --epochs $EPOCHS --batch 32""")

md("## 3. Held-out evaluation vs slope-blind baseline")
code("""!python src/evaluate.py --ckpt runs/colab/ckpt.pt --data data/eval.npz --out eval_out --n 48
import json; print(json.dumps({k: v for k, v in json.load(open("eval_out/results.json")).items() if k.startswith("model") or k=="random_baseline"}, indent=2))""")

md("## 4. Case studies: real-mode expansion on held-out towns")
code("""import numpy as np, torch
from sample import load_model, expand_once, real_expand
from towns import EVAL_TOWNS
device = "cuda"
diff = load_model("runs/colab/ckpt.pt", device)
cases = {}
for name, cc, lat, lon, region in EVAL_TOWNS[:6]:
    try:
        cases[name] = real_expand(diff, lat, lon, seed=7)
        print("ok", name)
    except Exception as e:
        print("skip", name, e)""")

md("## 5. Sandbox (game) mode: iterative growth on fractal terrain")
code("""from sample import sandbox_grow
elev_s, stages = sandbox_grow(diff, n_rounds=5, seed=3)
elev_s2, stages2 = sandbox_grow(diff, n_rounds=5, seed=11)
print("stages:", [ (int(b.sum()), int(r.sum())) for r,b in stages ])""")

md("## 6. Pack results and print as base64 (for retrieval)")
code("""import io, base64, json, numpy as np
pack = {"loss_history": np.loadtxt("runs/colab/loss_history.txt"),
        "results_json": json.dumps(json.load(open("eval_out/results.json"))),
        "sandbox_elev": elev_s, "sandbox_elev2": elev_s2}
for i,(r,b) in enumerate(stages):  pack[f"sb_r{i}"], pack[f"sb_b{i}"] = r, b
for i,(r,b) in enumerate(stages2): pack[f"s2_r{i}"], pack[f"s2_b{i}"] = r, b
for name, res in cases.items():
    key = name.replace(" ", "_")[:12]
    for f in ("elev","core_built","core_roads","real_roads","real_built","gen_roads","gen_built"):
        pack[f"case_{key}_{f}"] = res[f]
buf = io.BytesIO(); np.savez_compressed(buf, **pack); raw = buf.getvalue()
b64 = base64.b64encode(raw).decode()
print("PACKSIZE", len(raw), "B64LEN", len(b64))
CH = 60000
for i in range(0, len(b64), CH):
    print(f"===CHUNK {i//CH}===")
    print(b64[i:i+CH])
print("===END===", (len(b64)+CH-1)//CH, "chunks")""")

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

with open("terrain_urban_diffusion_colab.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("notebook written")
