"""Challenger model: gradient-boosted trees (XGBoost) for frequency and severity.

    DATA_DIR=/path python src/challenger.py      (after src/spark_glm.py)

Uses the same policies, split and rating fields as the GLM. Also runs the split experiment:
how much a random row split, which lets identical driver profiles sit on both sides, flatters
the trees relative to the GLM. Writes XGBoost predictions and the strongest pairwise interactions.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb

from metrics import poisson_deviance

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DATA_DIR", ROOT / "data"))
RESULTS = ROOT / "results"

FEATURES = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "logDensity", "VehBrand", "VehGas", "Region"]
CATEGORICAL = ["VehBrand", "VehGas", "Region"]
PARAMS = {"tree_method": "hist", "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
          "min_child_weight": 50, "seed": 7, "nthread": 8}


def frame() -> pd.DataFrame:
    df = pd.read_parquet(DATA / "glm_predictions.parquet")
    df["logDensity"] = np.log(df.Density)
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df


def matrix(df, label=None, margin=None, weight=None) -> xgb.DMatrix:
    return xgb.DMatrix(df[FEATURES], label=label, base_margin=margin, weight=weight, enable_categorical=True)


def fit_frequency(train: pd.DataFrame, depth: int) -> tuple[xgb.Booster, float]:
    """Poisson boosting with log(exposure) as the offset; early stopping on held-back risk groups."""
    val = (train.risk_group % 4) == 0
    tr, va = train[~val], train[val]
    dtr = matrix(tr, tr.claims_paid, np.log(tr.Exposure))
    dva = matrix(va, va.claims_paid, np.log(va.Exposure))
    booster = xgb.train(PARAMS | {"objective": "count:poisson", "max_depth": depth}, dtr, 3000,
                        evals=[(dva, "val")], early_stopping_rounds=100, verbose_eval=False)
    pred = booster.predict(dva, iteration_range=(0, booster.best_iteration + 1))
    return booster, poisson_deviance(va.claims_paid, pred)


def fit_severity(train: pd.DataFrame, depth: int) -> tuple[xgb.Booster, float]:
    sev = train[train.n_amounts > 0]
    val = (sev.risk_group % 4) == 0
    tr, va = sev[~val], sev[val]
    dtr = matrix(tr, tr.avg_severity, weight=tr.n_amounts)
    dva = matrix(va, va.avg_severity, weight=va.n_amounts)
    booster = xgb.train(PARAMS | {"objective": "reg:gamma", "max_depth": depth}, dtr, 3000,
                        evals=[(dva, "val")], early_stopping_rounds=100, verbose_eval=False)
    return booster, float(booster.best_score)


def rate(booster: xgb.Booster, df: pd.DataFrame) -> np.ndarray:
    """Expected claims per policy-year: predict with a zero offset."""
    return booster.predict(matrix(df, margin=np.zeros(len(df))), iteration_range=(0, booster.best_iteration + 1))


def glm_design(df: pd.DataFrame) -> np.ndarray:
    """The GLM's design (same bands as spark_glm.py), for the quick statsmodels refits below."""
    bands = pd.DataFrame({
        "VehPowerBand": np.minimum(df.VehPower, 9).astype(str),
        "VehAgeBand": pd.cut(df.VehAge, [-1, 0, 5, 10, np.inf], labels=["0", "1-5", "6-10", "11+"]).astype(str),
        "DrivAgeBand": pd.cut(df.DrivAge, [0, 20, 25, 30, 40, 50, 70, np.inf]).astype(str),
        "VehBrand": df.VehBrand.astype(str), "VehGas": df.VehGas.astype(str), "Region": df.Region.astype(str)})
    X = pd.get_dummies(bands, drop_first=True, dtype=float)
    X["logBonusMalus"] = np.log(np.minimum(df.BonusMalus, 150))
    X["logDensity"] = df.logDensity
    return sm.add_constant(X).to_numpy()


def split_experiment(df: pd.DataFrame, depth: int) -> dict:
    """Trees vs GLM on a grouped split and on a naive random split of the same policies."""
    out = {}
    X = glm_design(df)
    for name, test in [("grouped_split", (df.risk_group % 5) == 0),
                       ("random_split", (pd.util.hash_pandas_object(df.IDpol, index=False) % 5) == 0)]:
        tr, te = df[~test], df[test]
        glm = sm.GLM(tr.claims_paid, X[~test.to_numpy()], family=sm.families.Poisson(),
                     offset=np.log(tr.Exposure)).fit()
        glm_dev = poisson_deviance(te.claims_paid, glm.predict(X[test.to_numpy()], offset=np.log(te.Exposure)))
        booster, _ = fit_frequency(tr, depth)
        gbm_dev = poisson_deviance(te.claims_paid, rate(booster, te) * te.Exposure)
        twins = te.risk_group.isin(set(tr.risk_group)).mean()
        out[name] = {"glm_deviance": glm_dev, "gbm_deviance": gbm_dev,
                     "gbm_improvement_pct": 100 * (glm_dev - gbm_dev) / glm_dev,
                     "test_rows_with_a_twin_in_train": float(twins)}
    return out


def top_interactions(booster: xgb.Booster, df: pd.DataFrame, n: int = 20_000) -> pd.DataFrame:
    """Mean absolute SHAP interaction value for every feature pair, on a sample of policies."""
    sample = df.sample(n, random_state=1)
    vals = booster.predict(matrix(sample, margin=np.zeros(n)), pred_interactions=True,
                           iteration_range=(0, booster.best_iteration + 1))
    strength = np.abs(vals[:, :-1, :-1]).mean(axis=0)
    rows = [(FEATURES[i], FEATURES[j], 2 * strength[i, j]) for i in range(len(FEATURES)) for j in range(i + 1, len(FEATURES))]
    return pd.DataFrame(rows, columns=["feature_a", "feature_b", "mean_abs_interaction"]).sort_values(
        "mean_abs_interaction", ascending=False)


def main() -> None:
    df = frame()
    train = df[df.split == "train"]

    tuning = {}
    for depth in [3, 5, 7]:
        tuning[f"frequency_depth_{depth}"] = fit_frequency(train, depth)[1]
    freq_depth = min([3, 5, 7], key=lambda d: tuning[f"frequency_depth_{d}"])
    for depth in [2, 3, 4]:
        tuning[f"severity_depth_{depth}"] = fit_severity(train, depth)[1]
    sev_depth = min([2, 3, 4], key=lambda d: tuning[f"severity_depth_{d}"])

    freq_model, _ = fit_frequency(train, freq_depth)
    sev_model, _ = fit_severity(train, sev_depth)
    df["gbm_freq"] = rate(freq_model, df)
    df["gbm_sev"] = sev_model.predict(matrix(df), iteration_range=(0, sev_model.best_iteration + 1))
    df[["IDpol", "gbm_freq", "gbm_sev"]].to_parquet(DATA / "gbm_predictions.parquet", index=False)

    inter = top_interactions(freq_model, df)
    inter.to_csv(RESULTS / "gbm_top_interactions.csv", index=False)
    info = {"tuning_validation_deviance": tuning, "frequency_depth": freq_depth, "severity_depth": sev_depth,
            "frequency_rounds": freq_model.best_iteration + 1, "severity_rounds": sev_model.best_iteration + 1,
            "split_experiment": split_experiment(df, freq_depth),
            "top_interactions": inter.head(5).to_dict(orient="records")}
    (RESULTS / "challenger_fit.json").write_text(json.dumps(info, indent=2, default=float))
    print(json.dumps(info, indent=2, default=float))


if __name__ == "__main__":
    main()
