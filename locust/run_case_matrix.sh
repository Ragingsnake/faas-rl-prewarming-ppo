#!/bin/bash
# Runs 4 load cases with agent inactive and active, then saves per-case charts
# Run from inside locust/: bash run_case_matrix.sh

set -euo pipefail

RL_SERVER=${RL_SERVER:-http://127.0.0.1:8000}
CASES=(stable_low stable_high gradual_ramp sudden_spike)
MODES=(inactive active)

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install locust --quiet

mkdir -p results

get_steps() {
    curl -s "$RL_SERVER/status" | python3 -c "import json,sys; print(json.load(sys.stdin).get('steps', 0))"
}

set_mode() {
    local mode="$1"
    curl -s -X POST "$RL_SERVER/set-mode?mode=$mode" > /dev/null
}

set_learning() {
    local enabled="$1"
    curl -s -X POST "$RL_SERVER/set-learning?enabled=$enabled" > /dev/null
}

save_chart_for_range() {
    local mode="$1"
    local case_name="$2"
    local start_step="$3"
    local end_step="$4"
    local out="checkpoints/chart_${case_name}_${mode}.png"
    local title="Case=${case_name}_Mode=${mode}_Steps=${start_step}-${end_step}"
    curl -s -X POST \
      "$RL_SERVER/save-metrics?start_step=${start_step}&end_step=${end_step}&output=${out}&title=${title}" \
      > /dev/null
    echo "$out"
}

echo "Resetting in-memory metrics..."
curl -s -X POST "$RL_SERVER/reset-metrics" > /dev/null
echo "Disabling online learning for fair matrix comparison..."
set_learning false

echo "mode,case,start_step,end_step,chart_path" > results/case_ranges.csv

for mode in "${MODES[@]}"; do
    echo "=== MODE: $mode ==="
    set_mode "$mode"
    for case_name in "${CASES[@]}"; do
        echo "--- CASE: $case_name ---"
        start_step=$(get_steps)
        LOAD_CASE="$case_name" bash run_loadtest.sh
        end_step=$(get_steps)
        chart_path=$(save_chart_for_range "$mode" "$case_name" "$start_step" "$end_step")
        echo "${mode},${case_name},${start_step},${end_step},${chart_path}" >> results/case_ranges.csv
        echo "Saved ${chart_path} for step range ${start_step}-${end_step}"
    done
done

set_mode active
set_learning true
echo "All runs complete. Metadata: locust/results/case_ranges.csv"
