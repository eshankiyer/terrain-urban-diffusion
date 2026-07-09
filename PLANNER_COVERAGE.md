# Planner duty coverage

This file is the accounting the paper refers to: which duties of a
municipal planner this pipeline covers, which it assists, and which it
does not attempt. Duty definitions follow published planner job
descriptions (APA sample descriptions and municipal postings). The
classification of each duty is our judgment and is stated so it can be
argued with, duty by duty.

A duty is COVERED when the pipeline performs it end to end with a human
reviewing the output. ASSISTED means the pipeline contributes real work
but a person does most of it. HUMAN means the duty rests on discretion,
negotiation, or legal authority and is out of scope on purpose.

## Covered

1. Growth alternative generation. Five terrain-conditioned diffusion
   experts routed by conditioning statistics (src/moe.py, src/model.py).
2. Plan evaluation and comparison. Eleven-metric scorecard, best-of-N
   selection (src/sustainability.py).
3. Zoning annotation of proposed growth. Typed classifier with
   abstention and per-class margins, plus the advisory layer over
   expansion land (src/zones.py).
4. Demand and land-need forecasting. Linear and logistic fits over the
   GHSL epochs with growth budgets for candidate selection
   (src/forecast.py).
5. Development review at raster scale. Hazard placement, adjacency,
   intensity, and leapfrog checks under jurisdiction rule packs
   (src/compliance.py, src/jurisdiction.py).
6. Public comment intake and synthesis. Stance and topic tagging,
   frequent-request extraction into validated plan operations
   (src/public_comment.py).
7. Staff report drafting. Deterministic assembly from pipeline
   artifacts with a disclosed recommendation rule (src/staff_report.py).
8. Plan revision from stated intent. Closed operation schema over
   region locking, zone bias, amenity targets, and reranking
   (src/planner_agent.py, src/inpaint.py), with a pluggable language
   backend defaulting to a local Gemma model (src/llm_adapter.py).

## Assisted

9. Comprehensive plan updates. The pipeline supplies growth scenarios,
   forecasts, and maps; the plan document, its policies, and its
   adoption process are human work.
10. Site suitability and special studies. Scorecard layers and
    compliance checks inform them; study design and interpretation do
    not come from this pipeline.
11. Grant applications and capital improvement documentation. Outputs
    are quotable; the argument is not automated.
12. GIS data maintenance. The pipeline consumes and exports standard
    formats (GeoJSON) but does not manage a municipal GIS.

## Human

13. Discretionary approvals, variances, and appeals.
14. Negotiation with developers, agencies, and elected officials.
15. Public meeting facilitation and community relationships.
16. Legal interpretation and enforcement.
17. Ethics, equity judgments, and political accountability.

Two boundaries apply across every covered duty. First, each covered
output is a draft for a named human step: reports carry a review and
signature line, forecasts carry their fit residual, compliance findings
carry pack provenance, and comment summaries state their own blind
spots. Second, the rule packs that localize the compliance checks are
approximations that cite their instruments and demand local
verification; no output of this pipeline is a legal determination.
