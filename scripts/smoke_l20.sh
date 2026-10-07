#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python -m mm_opsd.doctor
python -m mm_opsd.prepare --output data/processed
python -m mm_opsd.train --config configs/smoke.yaml
python -m mm_opsd.evaluate --config configs/smoke.yaml --benchmark geometry3k \
  --adapter runs/smoke --checkpoint latest --max-samples 8 --output runs/smoke-eval
