#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}" || exit 1
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Add or remove experiment configuration files here.
CONFIGS=(
  "configs/ppo_bzip2_simple.json"
)

# Add or remove seeds here.
SEEDS=(
  0
  10
)

FAILED=0

for config in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "Running config=${config}, seed=${seed}"

    if ! python3 -m experiment.run_series \
      --config "${config}" \
      --seed "${seed}" \
      --summary "results/series_summary.json"; then
      echo "Experiment failed: config=${config}, seed=${seed}" >&2
      FAILED=1
    fi
  done
done

if [[ "${FAILED}" -ne 0 ]]; then
  echo "One or more experiments failed." >&2
  exit 1
fi

echo "All experiments completed."
