"""Champion rating model in PySpark: Poisson frequency GLM x Gamma severity GLM.

    JAVA_HOME=/path/to/jdk-21 DATA_DIR=/path python src/spark_glm.py    (Spark 4 runs on Java 17 or 21)

Reads freq.parquet and sev.parquet (from src/load.py), fits both GLMs on a train split that keeps
identical driver-and-car profiles together, cross-checks the frequency coefficients against
statsmodels, and writes predictions for every policy plus the rating relativities.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DATA_DIR", ROOT / "data"))
RESULTS = ROOT / "results"

CLAIM_CAP = 4              # a handful of policies report 5 to 16 claims a year: treated as data errors
EXPOSURE_CAP = 1.0         # exposure above one policy-year is a recording error
LARGE_LOSS_CAP = 200_000   # per claim; the top 0.1% of claims are over 160k and one is 4.1M
TEST_SHARE_BUCKETS = 5     # 1 in 5 risk groups is held out

# The raw rating fields. Rows that agree on all of them are treated as one risk (often the same
# driver on several contracts), so they always land on the same side of the split.
RISK_FIELDS = ["Area", "VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density", "Region"]
CATEGORICAL = ["VehPowerBand", "VehAgeBand", "DrivAgeBand", "VehBrand", "VehGas", "Region"]
NUMERIC = ["logBonusMalus", "logDensity"]
# GLM+: guided by the challenger's strongest SHAP interactions (driver age x bonus-malus, vehicle
# age x brand) and by the GLM's actual-to-expected misfits on the TRAINING data. Bonus-malus becomes
# rating bands instead of a log-linear slope, plus three indicators a rating filing can explain.
PLUS_CATEGORICAL = CATEGORICAL + ["BonusMalusBand"]
PLUS_NUMERIC = ["logDensity", "b12_new_vehicle", "above_floor_older_driver", "entry_level_bm_mature_driver"]


def spark_session() -> SparkSession:
    return (SparkSession.builder.master("local[8]").appName("insurance-glm")
            .config("spark.driver.memory", "10g")
            .config("spark.sql.shuffle.partitions", "16")
            .config("spark.local.dir", str(DATA / "spark_tmp"))
            .config("spark.ui.enabled", "false")
            .getOrCreate())


def build_frame(spark: SparkSession) -> DataFrame:
    """Policies with capped claims and exposure, joined to capped claim amounts, plus rating bands."""
    freq = spark.read.parquet(str(DATA / "freq.parquet"))
    sev = spark.read.parquet(str(DATA / "sev.parquet"))
    amounts = sev.groupBy("IDpol").agg(
        F.count("*").alias("n_amounts"),
        F.sum(F.least(F.col("ClaimAmount"), F.lit(float(LARGE_LOSS_CAP)))).alias("loss"))
    df = (freq.join(amounts, "IDpol", "left")
          .withColumn("ClaimNb", F.least("ClaimNb", F.lit(CLAIM_CAP)).cast("double"))
          .withColumn("Exposure", F.least("Exposure", F.lit(EXPOSURE_CAP)))
          .withColumn("n_amounts", F.coalesce("n_amounts", F.lit(0)).cast("double"))
          .withColumn("loss", F.coalesce("loss", F.lit(0.0)))
          # 9,116 policies report claims but have no payment in the severity file. Each policy has
          # either all its claims paid or none, so these are read as claims closed without payment.
          # Pricing uses paid claims, so frequency and severity describe the same claims (reported
          # claims x paid severity overstated holdout losses by 48%).
          .withColumn("amounts_missing", (F.col("ClaimNb") > 0) & (F.col("n_amounts") == 0))
          .withColumn("claims_paid", F.least("n_amounts", F.lit(float(CLAIM_CAP))))
          .withColumn("log_exposure", F.log("Exposure"))
          .withColumn("avg_severity", F.when(F.col("n_amounts") > 0, F.col("loss") / F.col("n_amounts")))
          .withColumn("VehPowerBand", F.least("VehPower", F.lit(9)).cast("string"))
          .withColumn("VehAgeBand", F.when(F.col("VehAge") == 0, "0").when(F.col("VehAge") <= 5, "1-5")
                      .when(F.col("VehAge") <= 10, "6-10").otherwise("11+"))
          .withColumn("DrivAgeBand", F.when(F.col("DrivAge") < 21, "18-20").when(F.col("DrivAge") < 26, "21-25")
                      .when(F.col("DrivAge") < 31, "26-30").when(F.col("DrivAge") < 41, "31-40")
                      .when(F.col("DrivAge") < 51, "41-50").when(F.col("DrivAge") < 71, "51-70").otherwise("71+"))
          .withColumn("logBonusMalus", F.log(F.least("BonusMalus", F.lit(150))))
          .withColumn("logDensity", F.log("Density"))
          .withColumn("b12_new_vehicle", ((F.col("VehBrand") == "B12") & (F.col("VehAge") == 0)).cast("double"))
          .withColumn("BonusMalusBand", F.when(F.col("BonusMalus") <= 50, "50").when(F.col("BonusMalus") <= 55, "51-55")
                      .when(F.col("BonusMalus") <= 60, "56-60").when(F.col("BonusMalus") <= 70, "61-70")
                      .when(F.col("BonusMalus") <= 80, "71-80").when(F.col("BonusMalus") <= 90, "81-90")
                      .when(F.col("BonusMalus") <= 100, "91-100").when(F.col("BonusMalus") <= 125, "101-125")
                      .otherwise("126+"))
          # An older driver not yet at the 50 floor has had recent claims.
          .withColumn("above_floor_older_driver",
                      ((F.col("BonusMalus") > 50) & (F.col("BonusMalus") <= 80) & (F.col("DrivAge") >= 50)).cast("double"))
          .withColumn("entry_level_bm_mature_driver",
                      ((F.col("BonusMalus") > 80) & (F.col("BonusMalus") <= 100) & (F.col("DrivAge") > 30)).cast("double"))
          .withColumn("risk_group", F.xxhash64(*RISK_FIELDS))
          .withColumn("split", F.when(F.pmod("risk_group", F.lit(TEST_SHARE_BUCKETS)) == 0, "test").otherwise("train")))
    return df


def design_pipeline(categorical: list[str], numeric: list[str]) -> Pipeline:
    """One-hot encode with the most common level as the reference (frequencyAsc + dropLast)."""
    idx = [StringIndexer(inputCol=c, outputCol=c + "_idx", stringOrderType="frequencyAsc") for c in categorical]
    ohe = OneHotEncoder(inputCols=[c + "_idx" for c in categorical], outputCols=[c + "_ohe" for c in categorical],
                        dropLast=True)
    assemble = VectorAssembler(inputCols=[c + "_ohe" for c in categorical] + numeric, outputCol="features")
    return Pipeline(stages=idx + [ohe, assemble])


def feature_names(df: DataFrame) -> list[str]:
    attrs = df.schema["features"].metadata["ml_attr"]["attrs"]
    flat = sorted((a for kind in attrs.values() for a in kind), key=lambda a: a["idx"])
    return [a["name"] for a in flat]


def coefficient_table(model, names: list[str]) -> pd.DataFrame:
    s = model.summary
    coefs = list(model.coefficients) + [model.intercept]
    return pd.DataFrame({"term": names + ["(intercept)"], "coef": coefs,
                         "se": s.coefficientStandardErrors, "p_value": s.pValues,
                         "relativity": np.exp(coefs)})


def fit_frequency(train: DataFrame):
    glm = GeneralizedLinearRegression(family="poisson", link="log", labelCol="claims_paid",
                                      offsetCol="log_exposure", featuresCol="features", maxIter=50)
    return glm.fit(train)


def fit_severity(train: DataFrame):
    glm = GeneralizedLinearRegression(family="gamma", link="log", labelCol="avg_severity",
                                      weightCol="n_amounts", featuresCol="features", maxIter=50)
    return glm.fit(train.where(F.col("n_amounts") > 0))


def cross_check(train: DataFrame, spark_model, names: list[str]) -> float:
    """Refit the frequency GLM with statsmodels on the same design matrix; return the largest gap."""
    pdf = train.select(vector_to_array("features").alias("x"), "claims_paid", "log_exposure").toPandas()
    X = sm.add_constant(np.vstack(pdf.x.to_numpy()), prepend=False)
    fit = sm.GLM(pdf.claims_paid.to_numpy(), X, family=sm.families.Poisson(), offset=pdf.log_exposure.to_numpy()).fit()
    spark_coefs = np.append(np.asarray(spark_model.coefficients), spark_model.intercept)
    return float(np.max(np.abs(fit.params - spark_coefs)))


def fit_pair(df: DataFrame, categorical: list[str], numeric: list[str], tag: str):
    """Fit frequency and severity GLMs on one design; return (scored frame, frequency model, names)."""
    design = design_pipeline(categorical, numeric).fit(df.where(F.col("split") == "train"))
    d = design.transform(df.drop("features")).cache()
    names = feature_names(d)
    train = d.where(F.col("split") == "train")
    freq_model, sev_model = fit_frequency(train), fit_severity(train)
    coefficient_table(freq_model, names).to_csv(RESULTS / f"{tag}_frequency_relativities.csv", index=False)
    coefficient_table(sev_model, names).to_csv(RESULTS / f"{tag}_severity_relativities.csv", index=False)
    scored = (freq_model.transform(d).withColumnRenamed("prediction", "pred_claims")
              .transform(lambda x: sev_model.transform(x).withColumnRenamed("prediction", "pred_severity"))
              .select("IDpol", (F.col("pred_claims") / F.col("Exposure")).alias(f"{tag}_freq"),
                      F.col("pred_severity").alias(f"{tag}_sev")))
    fit = {"terms": len(names) + 1, "deviance": freq_model.summary.deviance, "aic": freq_model.summary.aic,
           "iterations": freq_model.summary.numIterations,
           "severity_deviance": sev_model.summary.deviance, "severity_dispersion": sev_model.summary.dispersion}
    return scored, freq_model, names, train, fit


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    spark = spark_session()
    df = build_frame(spark).cache()

    glm_scored, glm_freq, names, train, glm_fit = fit_pair(df, CATEGORICAL, NUMERIC, "glm")
    gap = cross_check(train, glm_freq, names)
    plus_scored, _, _, _, plus_fit = fit_pair(df, PLUS_CATEGORICAL, PLUS_NUMERIC, "glmplus")

    out = (df.select("IDpol", "risk_group", "split", "Exposure", "ClaimNb", "claims_paid", "n_amounts", "loss",
                     "amounts_missing", "avg_severity", *RISK_FIELDS)
           .join(glm_scored, "IDpol").join(plus_scored, "IDpol"))
    out.write.mode("overwrite").parquet(str(DATA / "glm_predictions.parquet"))

    counts = df.groupBy("split").agg(F.count("*").alias("policies"), F.sum("Exposure").alias("exposure"),
                                     F.sum("ClaimNb").alias("claims_reported"), F.sum("claims_paid").alias("claims_paid"),
                                     F.countDistinct("risk_group").alias("risk_groups"),
                                     F.sum(F.col("amounts_missing").cast("int")).alias("amounts_missing")).toPandas()
    info = {"splits": counts.set_index("split").to_dict(orient="index"),
            "glm": glm_fit, "glm_plus": plus_fit,
            "spark_vs_statsmodels_max_coef_gap": gap}
    (RESULTS / "glm_fit.json").write_text(json.dumps(info, indent=2, default=float))
    print(json.dumps(info, indent=2, default=float))
    spark.stop()


if __name__ == "__main__":
    main()
