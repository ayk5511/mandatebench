#!/bin/zsh
# Full study, run in sequence. Fastest model first so the analysis can be
# built on real data while the larger models run. Each run is resumable.
cd "$(dirname "$0")/.."
set -u
run() { python3 scripts/run_model.py --model "$1" --out "results/runs/$2.jsonl" >> "results/runs/$2.log" 2>&1; }
run llama3.1:8b-instruct-q8_0     llama3.1-8b
run qwen2.5-coder:32b             qwen2.5-coder-32b
run qwen2.5:72b-instruct-q4_K_M   qwen2.5-72b
run qwen3:32b                     qwen3-32b
# Repeatability: an independent second pass of the fastest model. With
# temperature 0 and a fixed seed, any disagreement measures run-to-run noise in
# the serving stack rather than in the scoring.
run llama3.1:8b-instruct-q8_0     llama3.1-8b.rerun
echo "ALL DONE $(date)" >> results/runs/ALL_DONE
