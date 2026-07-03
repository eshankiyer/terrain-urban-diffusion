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

`environment.py` and the extended `scorecard_v2` close the gaps the first
scorecard proxied or skipped. The module fetches real OSM amenities in seven
categories (food, education, health, civic, recreation, transit, social),
greenspace and water masks, and derives two terrain hazard proxies: a HAND
flood mask (height above nearest water under 5 m within about 105 m of
mapped water) and slope-threshold landslide masks (25 and 35 degrees, severe
counted double). The 11-metric scorecard replaces density-peak service
centres with real fixed amenities, adds green preservation measured against
baseline functional patches of at least 0.45 ha, network-walk green access
within 300 m, flood and landslide avoidance over new development only,
a congestion proxy (1 minus the Gini of edge betweenness on the road graph),
and spatial access equity (population-weighted CV of per-cell amenity
coverage). Ranking applies a delivery factor so a do-nothing plan cannot
win best-of-N by avoiding every hazard. The design came from a STORM-style
multi-perspective spec (ecologist, transport engineer, geotechnical
engineer, equity planner) and the implementation survived two rounds of an
adversarial prover-verifier loop, which caught one critical defect
(pre-existing roads scored as new development) and three bugs before
acceptance. Income, tenure, affordability and displacement risk are named
human inputs, deliberately not fabricated from rasters.

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

## v3 direction: the transit layer

v2 generates urban fabric and audits walkability; transit is currently only
a destination category in the 15-minute metric (bus stops and stations count
as amenities, but no transit infrastructure is ever generated). A v3 would
add that layer, and it decomposes into three different kinds of problem.

Surface bus lanes and tram alignments are edge labeling on the road graph,
the same shape as the bike-lane stage: train on well-tagged European cities
(OSM busway, lanes:bus, railway=tram/light_rail) with corridor width, edge
betweenness, and density-along-edge as features, then annotate generated
arterials. Dedicated right-of-way and route planning is optimization, not
generation: choose the corridor that maximizes residents within a 400 m
walk of stops under a length budget, solved on the extracted graph and
scored with a new transit-corridor term in the scorecard. Subways are the
outlier: underground, decoupled from the surface raster, and only
meaningful at city scale.

That last point is the real commitment. Trams and metros exist in mid-size
cities, not 5k-60k mountain towns, so v3 means a new city-scale dataset:
larger windows (512 px or more), a city selection with usable transit
tagging, and correspondingly more compute. The staging that keeps it
tractable: bus-lane edge labeling first (works at current scale on
arterial-rich towns), then the corridor optimizer, then the city-scale
dataset jump only when the first two prove out.

## v3 direction: typed zoning

The model currently generates one undifferentiated density field; a planner
needs to know WHERE residential, commercial, industrial and institutional
uses go. Labels exist (OSM landuse polygons: residential, retail,
commercial, industrial, plus amenity clusters for institutional), but they
are patchy outside Europe and heavily imbalanced toward residential, so the
staging mirrors the bike-lane decision. First a post-hoc stage (zones.py):
classify generated density pixels into use classes from features the model
already produces, such as distance to centre, arterial adjacency, slope,
local density, and neighbouring existing uses, trained only where OSM
labels exist. Then, at city scale, fold zoning into the diffusion output
as K one-hot channels with a masked cross-entropy loss over labeled pixels.

Typed zones unlock the scorecard metrics that make 15-minute claims real:
land-use mix entropy within each walk-shed, a jobs-housing balance proxy
(commercial+industrial area vs residential area reachable in 15 minutes),
and hard compatibility penalties (new residential adjacent to industrial
scores zero on a new nuisance metric). They also unlock the most
interesting generative upgrade: an amenity-placement channel. v2 scores
access against EXISTING amenities only, which correctly penalizes remote
growth but cannot express "grow here AND put a school there." Letting the
model emit a proposed-local-centre channel (trained on OSM amenity density)
turns 15-minute planning fully generative, with the scorecard reporting
access both with and without the proposed amenities so the assumption
stays visible.

## v4 direction: a natural-language planning agent

The end state is conversational: a planner says "this district needs more
commercial and a school within a ten-minute walk" and the map updates.
The right architecture keeps the LLM out of the pixel business. An
open-weights model with function calling (Llama, Qwen, or Mistral served
locally via Ollama or llama.cpp) parses intent into a small operation
schema: REGENERATE(region_mask, constraints), SET_ZONE_BIAS(class,
weight, region), ADD_AMENITY_TARGET(category, region), PROTECT(mask),
RERANK(weights_override). An executor maps those onto tools that already
exist in this repo: masked RePaint-style regeneration on the trained
DDPM (the one new sampler this requires), zone-classifier reweighting,
amenity-channel biasing, scorecard re-ranking, and plan rendering. The
LLM never invents geometry; it routes verified operations, and every
result still passes through the same scorecard, so a request that would
pave the floodplain gets a low score the planner can see. A browser-only
variant is possible with WebLLM running a 3B model over WebGPU next to
the ONNX sampler already on the site.

## Honest scope

This remains a scenario-generation and critique tool. Land ownership, zoning
law, budgets and politics are not in any raster; the tool's job is to let a
planner explore many terrain-consistent futures quickly and defensibly, with
uncertainty shown rather than hidden.
