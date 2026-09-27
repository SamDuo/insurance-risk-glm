"""Credit scorecard: weight-of-evidence logistic GLM with adverse-action reason codes and a fairness check.

    DATA_DIR=/path python src/credit.py

Data: UCI "default of credit card clients" (30,000 Taiwan cardholders, 2005; target = default the next
month). Sex, age, education and marital status are NOT model inputs; they are used only afterwards, to
check how approvals and accuracy differ across groups.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from metrics import ks_statistic, psi, woe_table  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DATA_DIR", ROOT / "data"))
RESULTS, CHARTS = ROOT / "results", ROOT / "charts"

PDO, BASE_SCORE, BASE_ODDS = 20, 600, 50      # 600 points = 50:1 good:bad odds; +20 points doubles the odds
APPROVAL_RATE = 0.75                          # the fairness check approves the top 75% of scores
PAY = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL = [f"BILL_AMT{i}" for i in range(1, 7)]
PAID = [f"PAY_AMT{i}" for i in range(1, 7)]

REASONS = {
    "recent_status": "Recent payment is past due",
    "months_late_6m": "Number of late payments in the last 6 months",
    "worst_late_6m": "Serious delinquency in the last 6 months",
    "utilization": "Balance is high relative to credit limit",
    "credit_limit": "Credit limit is low",
    "payment_ratio_3m": "Recent payments are small relative to balances",
}
# Predictive but left out: default risk is U-shaped in balance trend (flat balances default most),
# so no single adverse-action reason could describe it truthfully to a consumer.
EXCLUDED = {"balance_trend": "U-shaped: flat balances default most; not explainable as a decline reason"}


def characteristics(raw: pd.DataFrame) -> pd.DataFrame:
    """Seven characteristics built only from the account's own credit behaviour."""
    df = pd.DataFrame(index=raw.index)
    df["recent_status"] = raw.PAY_0.clip(-2, 3)                          # -2 no use, -1 paid in full, 0 revolving, 1+ months late
    df["months_late_6m"] = (raw[PAY] >= 1).sum(axis=1)
    df["worst_late_6m"] = raw[PAY].max(axis=1).clip(-2, 4)
    df["utilization"] = (raw.BILL_AMT1 / raw.LIMIT_BAL).clip(-0.1, 1.5)
    df["credit_limit"] = raw.LIMIT_BAL
    prior_bills = raw[["BILL_AMT2", "BILL_AMT3", "BILL_AMT4"]].sum(axis=1).clip(lower=1)
    df["payment_ratio_3m"] = (raw[["PAY_AMT1", "PAY_AMT2", "PAY_AMT3"]].sum(axis=1) / prior_bills).clip(0, 2)
    df["balance_trend"] = (raw.BILL_AMT1 / raw[BILL[1:]].mean(axis=1).clip(lower=1)).clip(-1, 3)
    return df


DISCRETE = {"recent_status", "months_late_6m", "worst_late_6m"}


def monotone_bins(x: pd.Series, y: pd.Series, max_bins: int = 8) -> list[float]:
    """Quantile cut points, merged until the bad rate moves in one direction across bins."""
    edges = list(np.unique(np.quantile(x, np.linspace(0, 1, max_bins + 1))))
    edges[0], edges[-1] = -np.inf, np.inf
    while len(edges) > 3:
        rates = y.groupby(pd.cut(x, edges), observed=True).mean().to_numpy()
        steps = np.diff(rates)
        if (steps >= 0).all() or (steps <= 0).all():
            break
        direction = np.sign(steps.sum())
        worst = int(np.argmin(steps * direction))      # the step that most violates the trend
        del edges[worst + 1]                          # merge that pair of bins
    return edges


def bin_all(chars: pd.DataFrame, y: pd.Series) -> dict:
    spec = {}
    for c in chars:
        if c in DISCRETE:
            values = sorted(chars[c].unique())
            spec[c] = [-np.inf] + [v + 0.5 for v in values[:-1]] + [np.inf]
            # merge sparse levels into their neighbour so every bin has at least 1% of accounts
            counts = chars[c].value_counts(normalize=True).sort_index()
            spec[c] = [e for e, v in zip(spec[c], [None] + list(counts.index)) if v is None or counts.get(v, 0) >= 0.01] + [np.inf]
            spec[c] = sorted(set(spec[c]))
        else:
            spec[c] = monotone_bins(chars[c], y)
    return spec


def apply_bins(chars: pd.DataFrame, spec: dict) -> pd.DataFrame:
    return pd.DataFrame({c: pd.cut(chars[c], spec[c]).astype(str) for c in chars}, index=chars.index)


def main() -> None:
    raw = pd.read_parquet(DATA / "credit.parquet")
    y = raw["default payment next month"].astype(int)
    chars = characteristics(raw)
    idx_train, idx_test = train_test_split(raw.index, test_size=0.3, stratify=y, random_state=7)

    spec = bin_all(chars.loc[idx_train], y.loc[idx_train])
    bins = apply_bins(chars, spec)
    woe = {c: woe_table(bins.loc[idx_train, c], y.loc[idx_train]) for c in chars}
    iv = {c: float(t.iv.sum()) for c, t in woe.items()}
    X = pd.DataFrame({c: bins[c].map(woe[c].woe).fillna(0.0) for c in chars})

    # Keep characteristics whose coefficient has the expected sign (higher WoE = safer) and p < 0.01.
    kept = [c for c in chars if iv[c] >= 0.02 and c not in EXCLUDED]
    while True:
        fit = sm.Logit(y.loc[idx_train], sm.add_constant(X.loc[idx_train, kept])).fit(disp=0)
        bad = [c for c in kept if fit.params[c] >= 0 or fit.pvalues[c] > 0.01]
        if not bad:
            break
        kept.remove(max(bad, key=lambda c: fit.pvalues[c]))

    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    base_points = offset - factor * fit.params["const"]
    points = pd.DataFrame({c: -factor * fit.params[c] * X[c] for c in kept})
    score = base_points + points.sum(axis=1)
    p_bad = fit.predict(sm.add_constant(X[kept]))

    # Adverse-action reason codes: only characteristics that count AGAINST the account (negative
    # points, i.e. a worse bad rate than average), largest first. Ranking by "points below the best
    # attribute" instead would tell a customer who paid in full that a payment was past due.
    def reasons(i: int) -> list[str]:
        s = points.loc[i].sort_values()
        return [REASONS[c] for c in s.index[:4] if s[c] < -0.5]

    # Challenger: gradient-boosted trees on the same seven characteristics (raw values).
    dtr = xgb.DMatrix(chars.loc[idx_train], label=y.loc[idx_train])
    booster = xgb.train({"objective": "binary:logistic", "eta": 0.05, "max_depth": 3, "subsample": 0.8,
                         "min_child_weight": 20, "seed": 7}, dtr, 400)
    p_gbm = booster.predict(xgb.DMatrix(chars.loc[idx_test]))

    yt = y.loc[idx_test]
    st = score.loc[idx_test]
    auc = roc_auc_score(yt, p_bad.loc[idx_test])
    auc_gbm = roc_auc_score(yt, p_gbm)

    # Fairness: approve the top 75% of scores and compare groups the model never saw.
    cutoff = float(np.quantile(score.loc[idx_train], 1 - APPROVAL_RATE))
    groups = pd.DataFrame({
        "sex": raw.SEX.map({1: "male", 2: "female"}),
        "age": pd.cut(raw.AGE, [0, 29, 44, 200], labels=["under 30", "30-44", "45+"]).astype(str)}, index=raw.index)
    fairness = {}
    for g in groups:
        t = pd.DataFrame({"group": groups.loc[idx_test, g], "approved": st >= cutoff, "bad": yt,
                          "p": p_bad.loc[idx_test]})
        rows = t.groupby("group").apply(lambda d: pd.Series({
            "accounts": len(d), "approval_rate": d.approved.mean(),
            "default_rate_if_approved": d.bad[d.approved].mean(),
            "actual_default_rate": d.bad.mean(), "predicted_default_rate": d.p.mean()}), include_groups=False)
        rows["approval_ratio_vs_highest"] = rows.approval_rate / rows.approval_rate.max()
        fairness[g] = rows.round(4).to_dict(orient="index")

    declined = st[st < cutoff].sort_values().index
    examples = [{"score": round(float(score.loc[i])), "reasons": reasons(i)}
                for i in [declined[len(declined) // 10], declined[len(declined) // 2], declined[-1]]]

    for c in kept:
        woe[c] = woe[c].loc[sorted(woe[c].index, key=lambda b: float(b.split(",")[0].strip("(")))]
    scorecard = pd.concat([pd.DataFrame({"characteristic": c, "bin": woe[c].index, "accounts": woe[c]["count"].to_numpy(),
                                         "bad_rate": woe[c].bad_rate.round(4).to_numpy(), "woe": woe[c].woe.round(4).to_numpy(),
                                         "points": (-factor * fit.params[c] * woe[c].woe).round(1).to_numpy()})
                           for c in kept])
    scorecard.to_csv(RESULTS / "credit_scorecard.csv", index=False)
    out = {
        "accounts": len(raw), "default_rate": float(y.mean()), "train": len(idx_train), "test": len(idx_test),
        "information_value": {c: round(v, 3) for c, v in iv.items()},
        "characteristics_kept": kept, "excluded_for_explainability": EXCLUDED, "base_points": float(base_points),
        "coefficients": {c: round(float(fit.params[c]), 4) for c in kept},
        "holdout": {"auc": auc, "gini": 2 * auc - 1, "ks": ks_statistic(yt, -st),
                    "xgboost_auc": auc_gbm, "xgboost_gini": 2 * auc_gbm - 1,
                    "score_psi_train_vs_test": psi(score.loc[idx_train], st)},
        "cutoff_score": cutoff, "fairness_at_75pct_approval": fairness, "example_declines": examples,
    }
    (RESULTS / "credit.json").write_text(json.dumps(out, indent=2, default=float))
    chart(st, yt, groups.loc[idx_test, "sex"], cutoff)
    print(json.dumps(out, indent=2, default=float))


def chart(score, bad, sex, cutoff) -> None:
    bands = pd.qcut(score, 10, labels=False) + 1
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for g, color in [("male", "#1c5c84"), ("female", "#c05a2b")]:
        rate = bad[sex == g].groupby(bands[sex == g]).mean()
        ax.plot(rate.index, 100 * rate.to_numpy(), marker="o", lw=2, color=color, label=g.capitalize())
    ax.set_title("Default rate by score decile, men and women scored by the same card", loc="left", fontsize=12, color="#1f262b")
    ax.set_xlabel("Score decile (1 = lowest score)", color="#65767f")
    ax.set_ylabel("Actual default rate, %", color="#65767f")
    ax.set_xticks(range(1, 11))
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(CHARTS / "credit_default_by_decile_and_sex.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    main()
