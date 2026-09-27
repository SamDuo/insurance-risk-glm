"""Validation metrics used in insurance pricing and credit scoring.

Pure numpy functions, so they are easy to test and to reuse on Spark output collected to pandas.
Conventions: `exposure` is policy-years, `loss` is total claim cost, `pred` is predicted cost per
policy-year (pure premium) unless stated otherwise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------- deviance
def poisson_deviance(y, mu, weight=None) -> float:
    """Mean Poisson deviance. y are counts, mu expected counts (already times exposure)."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0) - (y - mu)
    return float(np.average(2 * term, weights=weight))


def gamma_deviance(y, mu, weight=None) -> float:
    """Mean Gamma deviance for positive y (severities)."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    return float(np.average(2 * (-np.log(y / mu) + (y - mu) / mu), weights=weight))


# ----------------------------------------------------------------------------- ranking
def lorenz(loss, pred, exposure):
    """Ordered Lorenz curve: sort by predicted rate (low to high), accumulate exposure and loss.

    Returns (cumulative exposure share, cumulative loss share), both starting at 0.
    """
    order = np.argsort(pred, kind="mergesort")
    e = np.concatenate([[0.0], np.cumsum(np.asarray(exposure, float)[order])])
    l = np.concatenate([[0.0], np.cumsum(np.asarray(loss, float)[order])])
    return e / e[-1], l / l[-1]


def gini(loss, pred, exposure) -> float:
    """Gini index of the ordered Lorenz curve: 1 - 2 * area under it.

    0 means the model ranks no better than random; higher means riskier policies are ranked
    higher. This is the ranking metric usually reported for pricing models on this data.
    """
    x, y = lorenz(loss, pred, exposure)
    return float(1 - 2 * np.trapezoid(y, x))


def lift_table(loss, pred, exposure, n: int = 10) -> pd.DataFrame:
    """Exposure-weighted deciles of predicted pure premium: actual vs predicted loss cost."""
    df = pd.DataFrame({"loss": loss, "pred": pred, "exposure": exposure}).sort_values("pred", kind="mergesort")
    df["bucket"] = np.minimum((df.exposure.cumsum() / df.exposure.sum() * n).astype(int), n - 1) + 1
    df["pred_loss"] = df.pred * df.exposure
    t = df.groupby("bucket")[["exposure", "loss", "pred_loss"]].sum()
    t["actual_pure_premium"] = t.loss / t.exposure
    t["predicted_pure_premium"] = t.pred_loss / t.exposure
    t["actual_to_expected"] = t.loss / t.pred_loss
    return t.reset_index()


def double_lift(loss, pred_a, pred_b, exposure, n: int = 10) -> pd.DataFrame:
    """Sort by the ratio of model B to model A. Where B prices higher than A, which one is right?

    Each model is first rescaled to the same total premium as actual losses, so the chart
    compares ranking, not overall level.
    """
    loss, exposure = np.asarray(loss, float), np.asarray(exposure, float)
    a = np.asarray(pred_a, float) * loss.sum() / (np.asarray(pred_a) * exposure).sum()
    b = np.asarray(pred_b, float) * loss.sum() / (np.asarray(pred_b) * exposure).sum()
    df = pd.DataFrame({"loss": loss, "a": a * exposure, "b": b * exposure, "exposure": exposure, "ratio": b / a})
    df = df.sort_values("ratio", kind="mergesort")
    df["bucket"] = np.minimum((df.exposure.cumsum() / df.exposure.sum() * n).astype(int), n - 1) + 1
    t = df.groupby("bucket")[["exposure", "loss", "a", "b"]].sum()
    for c in ["loss", "a", "b"]:
        t[c + "_per_exposure"] = t[c] / t.exposure
    return t.reset_index()


def dislocation(pred_old, pred_new, exposure, threshold: float = 0.10) -> float:
    """Share of policies whose premium moves by more than `threshold` when switching models,
    after rescaling both to the same total premium."""
    old = np.asarray(pred_old, float)
    new = np.asarray(pred_new, float) * (old * exposure).sum() / (np.asarray(pred_new, float) * exposure).sum()
    return float(np.mean(np.abs(new / old - 1) > threshold))


def grouped_bootstrap(stat, groups, reps: int = 500, seed: int = 11) -> np.ndarray:
    """Resample whole groups with replacement and evaluate stat(index_array) each time.

    Rows that share a group (here: identical driver-and-car profiles) move together, so the
    interval reflects how many independent risks there really are.
    """
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(pd.Series(groups))
    order = np.argsort(codes, kind="mergesort")
    starts = np.searchsorted(codes[order], np.arange(len(uniq)))
    ends = np.append(starts[1:], len(codes))
    out = np.empty(reps)
    for r in range(reps):
        pick = rng.integers(len(uniq), size=len(uniq))
        idx = np.concatenate([order[starts[g]:ends[g]] for g in pick])
        out[r] = stat(idx)
    return out


# ----------------------------------------------------------------------------- credit
def ks_statistic(y, score) -> float:
    """Maximum gap between the score distributions of bads (y=1) and goods (y=0)."""
    y, score = np.asarray(y), np.asarray(score, float)
    thresholds = np.unique(score)
    bad = np.searchsorted(np.sort(score[y == 1]), thresholds, side="right") / (y == 1).sum()
    good = np.searchsorted(np.sort(score[y == 0]), thresholds, side="right") / (y == 0).sum()
    return float(np.max(np.abs(bad - good)))


def woe_table(bins, y) -> pd.DataFrame:
    """Weight of evidence and information value per bin; WoE = ln(share of goods / share of bads)."""
    df = pd.DataFrame({"bin": bins, "bad": np.asarray(y, int)})
    t = df.groupby("bin", observed=True).bad.agg(["count", "sum"]).rename(columns={"sum": "bads"})
    t["goods"] = t["count"] - t.bads
    good_share = (t.goods + 0.5) / (t.goods.sum() + 0.5)   # +0.5 keeps empty cells finite
    bad_share = (t.bads + 0.5) / (t.bads.sum() + 0.5)
    t["bad_rate"] = t.bads / t["count"]
    t["woe"] = np.log(good_share / bad_share)
    t["iv"] = (good_share - bad_share) * t.woe
    return t


def psi(expected, actual, bins: int = 10) -> float:
    """Population stability index between two score samples, using the expected sample's deciles."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))
