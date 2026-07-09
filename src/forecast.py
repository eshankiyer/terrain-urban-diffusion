"""Built-area forecasting and growth budgets from GHSL epoch series.

A planner sizing a land-use plan starts from a demand number: how much
development the next decade will actually bring. The diffusion model has
no opinion about volume; best-of-N selection as shipped rewards plans
that deliver growth well, not plans that deliver the right amount of it.
This module supplies the missing number. It fits the five GHSL epochs a
window already carries (1975 through 2020 built pixel counts), projects
the series to a target year, and turns the projection into a growth
budget in pixels that candidate plans can be scored against.

Two candidate models are fitted and the better one wins on residual sum
of squares: a straight line, which suits steady incremental towns, and a
logistic curve, which suits windows that boomed and are now saturating.
Five points cannot support anything richer, and the module refuses to
extrapolate beyond twenty years for that reason. Forecasts from five
points are rough by construction; the output carries the fit residual so
downstream text can qualify the number instead of laundering it.

Dependencies: numpy only.
"""

import numpy as np

MAX_HORIZON_YEARS = 20


def _fit_linear(years, built):
    t = np.asarray(years, float)
    y = np.asarray(built, float)
    a, b = np.polyfit(t, y, 1)
    pred = a * t + b
    sse = float(((y - pred) ** 2).sum())
    return {"kind": "linear", "a": float(a), "b": float(b), "sse": sse}


def _fit_logistic(years, built):
    """Fit y = K / (1 + exp(-r (t - t0))) by linearizing at fixed K.

    K is grid searched between just above the observed maximum and four
    times it. For each K the transform log((K - y) / y) is linear in t,
    so ordinary least squares recovers r and t0. Zero counts are floored
    at one pixel to keep the transform finite.
    """
    t = np.asarray(years, float)
    y = np.maximum(np.asarray(built, float), 1.0)
    best = None
    top = float(y.max())
    for K in np.linspace(top * 1.05, top * 4.0, 24):
        z = np.log((K - y).clip(1e-9) / y)
        slope, intercept = np.polyfit(t, z, 1)
        r = -slope
        if r <= 0:
            continue
        pred = K / (1.0 + np.exp(intercept + slope * t))
        sse = float(((np.asarray(built, float) - pred) ** 2).sum())
        if best is None or sse < best["sse"]:
            best = {"kind": "logistic", "K": float(K), "r": float(r),
                    "t0": float(-intercept / slope), "sse": sse}
    return best


def fit_series(years, built):
    """Fit both models, return (chosen, alternatives).

    The (year, built) pairs are sorted by year first, so callers may
    pass epochs in any order. Selection is by SSE with a 1.5 penalty
    factor on the logistic, which has more effective parameters and can
    overfit a five-point series that a line describes just as well.
    """
    pairs = sorted(zip(years, built))
    years = [p[0] for p in pairs]
    built = [p[1] for p in pairs]
    lin = _fit_linear(years, built)
    log = _fit_logistic(years, built)
    fits = [f for f in (lin, log) if f is not None]
    fits.sort(key=lambda f: f["sse"] * (1.5 if f["kind"] == "logistic"
                                        else 1.0))
    return fits[0], fits


def _predict(fit, year):
    if fit["kind"] == "linear":
        return fit["a"] * year + fit["b"]
    return fit["K"] / (1.0 + np.exp(-fit["r"] * (year - fit["t0"])))


def forecast(years, built, target_year):
    """Project built pixels to target_year.

    Returns {"model", "pred_px", "new_px", "annual_px", "sse", "note"}.
    new_px is the increment over the last observation, floored at zero:
    GHSL revisions can make a series wobble downward, and a negative
    demand number is not a plan input anyone can use. Epochs may arrive
    in any order; pairs are sorted by year before fitting.
    """
    pairs = sorted(zip(years, built))
    years = [p[0] for p in pairs]
    built = [p[1] for p in pairs]
    if len(years) < 3:
        raise ValueError("need at least three epochs to fit")
    horizon = target_year - years[-1]
    if not 0 < horizon <= MAX_HORIZON_YEARS:
        raise ValueError(f"target year must be 1 to {MAX_HORIZON_YEARS} "
                         f"years past {years[-1]}")
    fit, _ = fit_series(years, built)
    pred = float(_predict(fit, target_year))
    new_px = max(0.0, pred - built[-1])
    return {"model": fit["kind"], "pred_px": pred, "new_px": new_px,
            "annual_px": new_px / horizon, "sse": fit["sse"],
            "note": ("five-point fit; treat as an order of magnitude, "
                     "not a target")}


def epoch_series(dens_by_epoch, thr=0.05):
    """(years, built_px) from {epoch: density raster}. Sorted by year."""
    years = sorted(dens_by_epoch)
    built = [int((np.asarray(dens_by_epoch[y]) > thr).sum()) for y in years]
    return years, built


def budget_alignment(candidates, d0, budget_px, thr=0.05, tol=0.5):
    """Score candidates by closeness of delivered growth to the budget.

    For each (roads, dens) candidate, delivered growth is the count of
    pixels newly above thr relative to d0. The score is 1 at the budget
    and falls linearly to 0 at a relative error of tol on either side.
    The relative error denominator is floored at 1/tol so that a
    near-zero positive budget cannot zero out every candidate; with the
    floor, the absolute error needed to drive a score to 0 is at least
    one pixel. Returned as a list aligned with candidates, for use as a
    rerank feature next to the sustainability scorecard, not as a
    replacement for it: matching the demand number says nothing about
    where the growth went.
    """
    base = np.asarray(d0) > thr
    out = []
    for _roads, dens in candidates:
        new = int(((np.asarray(dens) > thr) & ~base).sum())
        if budget_px <= 0:
            out.append(1.0 if new == 0 else 0.0)
            continue
        rel_err = abs(new - budget_px) / max(budget_px, 1.0 / tol)
        out.append(float(max(0.0, 1.0 - rel_err / tol)))
    return out


if __name__ == "__main__":
    years = [1975, 1990, 2000, 2015, 2020]

    # a saturating boom town should pick the logistic and flatten out
    K, r, t0 = 4000.0, 0.12, 1998.0
    boom = [K / (1 + np.exp(-r * (t - t0))) for t in years]
    f = forecast(years, boom, 2035)
    assert f["model"] == "logistic", f["model"]
    assert f["pred_px"] <= K * 1.05
    assert f["new_px"] < 2.0 * (boom[-1] - boom[-2])

    # a steady town should pick the line and keep its slope
    steady = [200 + 30 * (t - 1975) for t in years]
    f2 = forecast(years, steady, 2030)
    assert f2["model"] == "linear", f2["model"]
    assert abs(f2["annual_px"] - 30) < 1.5, f2["annual_px"]

    # unsorted epochs give the same forecast as sorted ones
    perm = [2015, 1975, 2020, 1990, 2000]
    steady_perm = [200 + 30 * (t - 1975) for t in perm]
    fp = forecast(perm, steady_perm, 2030)
    assert fp["model"] == f2["model"]
    assert abs(fp["pred_px"] - f2["pred_px"]) < 1e-6, fp["pred_px"]

    # declining series: demand floors at zero rather than going negative
    decline = [900, 880, 870, 855, 850]
    f3 = forecast(years, decline, 2030)
    assert f3["new_px"] == 0.0

    # horizon guard
    try:
        forecast(years, steady, 2080)
        raise AssertionError("accepted a 60 year horizon")
    except ValueError:
        pass

    # budget alignment prefers the candidate nearest the demand number
    g = 64
    d0 = np.zeros((g, g), np.float32)
    d0[:10, :10] = 0.5
    small = d0.copy(); small[20:22, 20:25] = 0.5      # 10 new px
    right = d0.copy(); right[30:40, 30:40] = 0.5      # 100 new px
    huge = d0.copy(); huge[:, 40:] = 0.5              # ~1400 new px
    scores = budget_alignment([(None, small), (None, right), (None, huge)],
                              d0, budget_px=100)
    assert scores[1] == max(scores) and scores[1] > 0.9
    assert scores[2] == 0.0

    # floored denominator: a tiny positive budget cannot zero everything
    one = d0.copy(); one[50, 50] = 0.5                # 1 new px
    frac = budget_alignment([(None, one)], d0, budget_px=0.1)
    assert frac[0] > 0.0, frac

    ys, built = epoch_series({1990: d0, 2020: right})
    assert ys == [1990, 2020] and built[1] > built[0]
    print("forecast self-tests passed")
