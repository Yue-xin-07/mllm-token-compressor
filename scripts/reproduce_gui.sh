#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/gui
for mode in native working_set; do
    .venv/bin/python scripts/benchmark_e2e.py --mode "$mode" --output "outputs/gui/perf-$mode.json"
    .venv/bin/python scripts/evaluate_gui.py --mode "$mode" --protocol point --dataset data/screenspot \
        --output "outputs/gui/screen-$mode.json"
done
.venv/bin/python scripts/evaluate_gui_replay.py --protocol point --split test --dataset data/screenspot \
    --output outputs/gui/replay-test.json
.venv/bin/python scripts/report_gui.py
