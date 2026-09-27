"""Convert the downloaded source files to Parquet.

    DATA_DIR=/path python src/load.py

freMTPL2freq / freMTPL2sev (OpenML 41214 / 41215) are ARFF files: a header, then CSV rows with
single-quoted strings. The UCI credit-card default file is an .xls with a title row above the header.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DATA_DIR", ROOT / "data"))


def read_arff(path: Path) -> pd.DataFrame:
    head, rows = path.read_text().split("@data", 1)
    cols = [line.split()[1] for line in head.splitlines() if line.lower().startswith("@attribute")]
    return pd.read_csv(io.StringIO(rows.strip()), header=None, names=cols, quotechar="'")


def main() -> None:
    freq = read_arff(DATA / "freMTPL2freq.arff")
    freq["IDpol"] = freq["IDpol"].astype("int64")
    sev = read_arff(DATA / "freMTPL2sev.arff")
    sev["IDpol"] = sev["IDpol"].astype("int64")
    credit = pd.read_excel(DATA / "default of credit card clients.xls", header=1)
    freq.to_parquet(DATA / "freq.parquet", index=False)
    sev.to_parquet(DATA / "sev.parquet", index=False)
    credit.to_parquet(DATA / "credit.parquet", index=False)
    print(f"policies {len(freq):,}  claim amounts {len(sev):,}  credit accounts {len(credit):,}")


if __name__ == "__main__":
    main()
