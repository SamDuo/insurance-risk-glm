"""Champion / challenger validation on the grouped holdout.

    DATA_DIR=/path python src/validate.py       (after spark_glm.py and challenger.py)

Compares the GLM (champion), the GLM with three extra terms (GLM+) and XGBoost (challenger) on
frequency deviance, pure-premium ranking (Gini, with risk-group bootstrap intervals), calibration by
decile, double-lift, and premium dislocation. Writes results/validation.json and charts.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from metrics import (dislocation, double_lift, gamma_deviance, gini, grouped_bootstrap, lift_table,  # noqa: E402
                     lorenz, poisson_deviance)

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DATA_DIR", ROOT / "data"))
RESULTS, CHARTS = ROOT / "results", ROOT / "charts"
MODELS = {"glm": "GLM (champion)", "glmplus": "GLM+ (GBM-informed)", "gbm": "XGBoost (challenger)"}
# GLM+ changes the frequency model only: its terms came from frequency misfits. (spark_glm.py also fits
# a severity model with the same terms; its holdout deviance is reported as a variant.)
SEVERITY_OF = {"glm": "glm", "glmplus": "glm", "gbm": "gbm"}


def load() -> pd.DataFrame:
    df = pd.read_parquet(DATA / "glm_predictions.parquet").merge(pd.read_parquet(DATA / "gbm_predictions.parquet"), on="IDpol")
    train = df.split == "train"
    for m in MODELS:
        df[f"{m}_pp"] = df[f"{m}_freq"] * df[f"{SEVERITY_OF[m]}_sev"]      # pure premium per policy-year
        # Off-balance: frequency x severity is not level-balanced (the Gamma log link is not the
        # canonical link), so rescale the base rate until predicted loss equals actual loss on TRAIN.
        factor = df.loc[train, "loss"].sum() / (df.loc[train, f"{m}_pp"] * df.loc[train, "Exposure"]).sum()
        df[f"{m}_pp"] *= factor
        df.attrs[f"{m}_off_balance"] = float(factor)
    return df


def main() -> None:
    CHARTS.mkdir(exist_ok=True)
    df = load()
    test = df[df.split == "test"]
    # Priced on paid claims: policies whose claims closed without payment have a known loss of 0.
    pp = test.reset_index(drop=True)
    sev = test[test.n_amounts > 0]
    out = {"holdout": {"policies": len(test), "claims_closed_without_payment": int(test.amounts_missing.sum()),
                       "actual_loss": float(pp.loss.sum()), "exposure": float(pp.Exposure.sum())}}

    for m, label in MODELS.items():
        out[m] = {
            "label": label,
            "off_balance_factor_from_train": df.attrs[f"{m}_off_balance"],
            "frequency_deviance": poisson_deviance(test.claims_paid, test[f"{m}_freq"] * test.Exposure),
            "severity_deviance": gamma_deviance(sev.avg_severity, sev[f"{SEVERITY_OF[m]}_sev"], sev.n_amounts),
            "claims_actual_over_predicted": float(test.claims_paid.sum() / (test[f"{m}_freq"] * test.Exposure).sum()),
            "gini": gini(pp.loss, pp[f"{m}_pp"], pp.Exposure),
            "predicted_over_actual_loss": float((pp[f"{m}_pp"] * pp.Exposure).sum() / pp.loss.sum()),
            "decile_actual_to_expected": lift_table(pp.loss, pp[f"{m}_pp"], pp.Exposure).actual_to_expected.round(3).tolist(),
        }

    out["glmplus"]["severity_variant_with_plus_terms_deviance"] = gamma_deviance(
        sev.avg_severity, sev.glmplus_sev, sev.n_amounts)
    big = pp.loss[pp.n_amounts > 0] / pp.n_amounts[pp.n_amounts > 0]
    train = df[df.split == "train"]
    tbig = train.loss[train.n_amounts > 0] / train.n_amounts[train.n_amounts > 0]
    out["holdout"]["mean_paid_severity"] = float(pp.loss.sum() / pp.n_amounts.sum())
    out["holdout"]["train_mean_paid_severity"] = float(train.loss.sum() / train.n_amounts.sum())
    out["holdout"]["share_of_loss_from_claims_over_50k"] = float(big[big > 50_000].sum() / big.sum())
    out["holdout"]["train_share_of_loss_from_claims_over_50k"] = float(tbig[tbig > 50_000].sum() / tbig.sum())

    loss, expo, groups = pp.loss.to_numpy(), pp.Exposure.to_numpy(), pp.risk_group.to_numpy()
    preds = {m: pp[f"{m}_pp"].to_numpy() for m in MODELS}

    def diff(a, b):
        return lambda i: gini(loss[i], preds[a][i], expo[i]) - gini(loss[i], preds[b][i], expo[i])

    for a, b in [("gbm", "glm"), ("glmplus", "glm"), ("gbm", "glmplus")]:
        draws = grouped_bootstrap(diff(a, b), groups, reps=300)
        out[f"gini_{a}_minus_{b}"] = {"estimate": out[a]["gini"] - out[b]["gini"],
                                      "ci_95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]}
    for new in ["glmplus", "gbm"]:
        out[f"dislocation_glm_to_{new}"] = {
            f"over_{int(t * 100)}pct": dislocation(pp.glm_pp, pp[f"{new}_pp"], pp.Exposure, t) for t in (0.10, 0.25)}

    (RESULTS / "validation.json").write_text(json.dumps(out, indent=2, default=float))
    charts(pp)
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict) or "decile_actual_to_expected" not in v}, indent=2, default=float))
    for m in MODELS:
        print(m, {k: round(v, 4) for k, v in out[m].items() if isinstance(v, float)})


INK, MUTED, LINE = "#1f262b", "#65767f", "#d5dde2"
COLORS = {"glm": "#65767f", "glmplus": "#1c5c84", "gbm": "#c05a2b"}


def tidy(ax, title):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(LINE)
    ax.tick_params(colors=MUTED)


def charts(pp: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for m, label in MODELS.items():
        t = lift_table(pp.loss, pp[f"{m}_pp"], pp.Exposure)
        ax.plot(t.bucket, t.actual_to_expected, marker="o", color=COLORS[m], lw=2, label=label)
    ax.axhline(1, color=LINE, lw=1)
    tidy(ax, "Actual over predicted loss, by decile of predicted risk (holdout)")
    ax.set_xlabel("Decile of predicted pure premium (1 = lowest risk)", color=MUTED)
    ax.set_ylabel("Actual / predicted loss", color=MUTED)
    ax.set_xticks(range(1, 11))
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(CHARTS / "calibration_by_decile.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, new in zip(axes, ["gbm", "glmplus"]):
        t = double_lift(pp.loss, pp.glm_pp, pp[f"{new}_pp"], pp.Exposure)
        ax.plot(t.bucket, t.loss_per_exposure, color=INK, lw=2, marker="o", label="Actual")
        ax.plot(t.bucket, t.a_per_exposure, color=COLORS["glm"], lw=2, ls="--", label=MODELS["glm"])
        ax.plot(t.bucket, t[f"b_per_exposure"], color=COLORS[new], lw=2, label=MODELS[new])
        tidy(ax, f"{MODELS[new].split(' (')[0]} vs GLM")
        ax.set_xlabel(f"Decile of {MODELS[new].split(' (')[0]} / GLM price ratio", color=MUTED)
        ax.set_xticks(range(1, 11))
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Loss per policy-year (€)", color=MUTED)
    fig.suptitle("Double lift: where the models disagree, which one matches actual losses?", x=0.02, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout(); fig.savefig(CHARTS / "double_lift.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], color=LINE, lw=1)
    for m, label in MODELS.items():
        x, y = lorenz(pp.loss, pp[f"{m}_pp"], pp.Exposure)
        step = max(1, len(x) // 2000)
        ax.plot(x[::step], y[::step], color=COLORS[m], lw=2,
                label=f"{label}: Gini {gini(pp.loss, pp[f'{m}_pp'], pp.Exposure):.3f}")
    tidy(ax, "Ordered Lorenz curves (holdout)")
    ax.set_xlabel("Share of exposure, lowest predicted risk first", color=MUTED)
    ax.set_ylabel("Share of losses", color=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(CHARTS / "lorenz.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    main()
