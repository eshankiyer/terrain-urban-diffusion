# v2 roadmap: terrain-urban-diffusion as a planning tool

Status of this branch: `src/data_v2.py` and `src/bikelanes.py` are implemented
and smoke-tested; everything else below is a prioritized plan. Channel counts
in v2 match v1, so the existing UNet, train.py and sample.py run unchanged.

## Implemented in this branch

`data_v2.py` replaces the concentric-erosion growth proxy with real temporal
pairs from GHS-BUILT-S R2023A (3 arc-second WGS84 tiles, epochs 1980/1990/
2000/2010/2020, downloaded and cached per tile). Each sample conditions on
elevation, slope, observed built density at epoch t and roads inside the
epoch-t footprint, and targets the epoch t+1 density plus roads outside the
footprint. Density is a continuous built fraction, so infill densification and
outward expansion are both learnable; ordinal classes for evaluation come from
`density_to_classes`. Known proxy that remains: OSM has no road history, so
"roads at epoch t" is present-day roads clipped to the epoch-t footprint.

`sustainability.py` makes the sustainable-15-minute-town objective concrete
and turns the trained model into an optimizer without retraining. It scores
any (roads, density) plan on five raster metrics: 15-minute walk coverage
(Dijkstra network walk times from density-peak service centres, 80 m/min,
15-minute limit), infill share of growth versus greenfield sprawl, land
efficiency (added density per newly consumed pixel), network circuity, and
an earthwork index (slope under new roads, which doubles as an
erosion/grading proxy). `rank_samples` performs best-of-N selection: sample
many futures, keep the most sustainable. The smoke test verifies the
scorecard prefers a compact, connected candidate over a detached sprawl
candidate on the same terrain. Two honest caveats are documented in the
module: service centres are proxied by density peaks until an amenity
channel exists, and the score measures spatial form, not certified
environmental performance.

`bikelanes.py` keeps bike infrastructure out of the diffusion model (OSM
cycleway tags are near-absent in the training towns) and instead trains a
per-edge logistic classifier on twelve well-tagged flat European towns, using
transferable physical features: mean absolute grade along the edge (from the
DEM), road class, edge length, distance to centre. It includes a
skeleton-to-graph extractor so the classifier can annotate generated road
rasters, and reports leave-one-town-out AUC so the transfer claim is testable.

## Next datasets to add (from a verified 12-dataset survey)

Priority four, chosen for resolution match, small-town validity and effort:

| Dataset | Res | Use | Why first |
|---|---|---|---|
| FABDEM slope/aspect | 30m | conditioning | removes tree/building bias in current DEM; feeds bike stage |
| ESA WorldCover 2021 | 10m | conditioning | only land cover at native grid resolution; rural-validated |
| JRC river flood hazard v2.1 | ~90m | conditioning + loss penalty | the planning constraint for valley towns; treat as soft prior (misses headwater streams) |
| GHS-BUILT-H height | 100m | output or scalar | adds vertical form; check per-town variance first (weak over low-rise towns) |

Nearly free scalars worth adding in the same pass: GEM seismic PGA, WorldClim
temperature/precipitation, Global Solar Atlas GHI (one number per town each).
Deliberately skipped: wildfire (no global high-res product; US-only at 30m),
GHS-POP (circular with built-up conditioning), land value (no open global
raster), night lights and PM2.5 (too smooth or too noisy at town scale).

## Planner-facing capabilities, ranked

The four to build first are all training-free on the current DDPM:

1. Region-locked inpainting (RePaint-style compositing with resampling jumps):
   planners lock existing fabric, floodplains, unowned land; the model fills
   the rest. Foundation for everything below. ~2-4 days.
2. Multi-sample consensus and uncertainty maps: sample 32-64 futures, show
   per-pixel agreement ("built in 87% of futures") and contested frontiers.
   Cheapest conversion from image generator to planning tool. ~3 days.
3. Vector export: skeletonize roads to GeoJSON centerlines, polygonize zoning,
   georeference. Without QGIS/ArcGIS interop the tool does not exist for a
   working planner. ~3-5 days.
4. Sketch-guided completion (SDEdit, with the sketch pixels pinned via
   inpainting): planner draws the arterial, model develops around it. The
   noise level t0 doubles as a faithfulness/realism slider. ~2-3 days.

Then: a metric scorecard on the exported graph (network efficiency, walk-shed
coverage, cut-and-fill earthwork from the DEM); counterfactual edits (delete a
bridge, regenerate a buffered neighborhood, diff the metrics); one retrain that
adds FiLM scalar conditions with CFG dropout to bundle compact-vs-sprawl
sliders and growth-rate control for iterative decade rollouts; plan critique
mode (reconstruction-surprise maps over a real draft plan, scored against the
generated ensemble); and, as a stretch, distilling the scorecard into a small
critic CNN used as guidance so sampling can steer toward low earthwork or high
accessibility directly.

## Sequencing note

The single scheduling trick: the compact/sprawl knobs and the growth-rate
control are both FiLM scalars with classifier-free-guidance dropout, so they
share one retrain. Do all training-free capabilities first, prove the
interaction loop, then pay for that one retrain.

## Honest scope

This remains a scenario-generation and critique tool. Land ownership, zoning
law, budgets and politics are not in any raster; the tool's job is to let a
planner explore many terrain-consistent futures quickly and defensibly, with
uncertainty shown rather than hidden.
