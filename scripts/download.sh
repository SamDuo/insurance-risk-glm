#!/usr/bin/env bash
# Download the three public source files (about 42 MB) into $DATA_DIR.
set -euo pipefail
DATA_DIR="${DATA_DIR:-data}"
mkdir -p "$DATA_DIR"
curl -sSfL -o "$DATA_DIR/freMTPL2freq.arff" https://openml.org/data/v1/download/20649148/freMTPL2freq.arff
curl -sSfL -o "$DATA_DIR/freMTPL2sev.arff"  https://openml.org/data/v1/download/20649149/freMTPL2sev.arff
curl -sSfL -o "$DATA_DIR/taiwan_credit.zip" "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
(cd "$DATA_DIR" && unzip -o -q taiwan_credit.zip)
echo "downloaded to $DATA_DIR"
