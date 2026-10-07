#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python -m mm_opsd.doctor
python -m mm_opsd.prepare --output data/processed
python -m mm_opsd.train --config configs/l20.yaml
for benchmark in geometry3k mathvista; do
  python -m mm_opsd.evaluate --config configs/l20.yaml --benchmark "$benchmark" \
    --output "runs/eval/base-$benchmark"
  python -m mm_opsd.evaluate --config configs/l20.yaml --benchmark "$benchmark" \
    --adapter runs/mm-opsd --checkpoint best --output "runs/eval/opsd-$benchmark"
  python -m mm_opsd.report --baseline "runs/eval/base-$benchmark" \
    --trained "runs/eval/opsd-$benchmark" --output "runs/reports/$benchmark.json"
done
