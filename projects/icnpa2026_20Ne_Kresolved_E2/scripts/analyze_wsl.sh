#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(cd "${project_dir}/../.." && pwd)"
response_tool="${repo_dir}/Utils/Strength_Calculation"
export PYTHONPATH="/home/abhishek/.local/share/sky3d-response-deps:${response_tool}"
export MPLBACKEND=Agg

python3 "${project_dir}/scripts/prepare_analysis_prefixes.py"
for component in k0 k1 k2 k0_half; do
  python3 "${response_tool}/Fourier.py" analyze \
    --config "${project_dir}/inputs/response_${component}.toml"
done
python3 "${project_dir}/scripts/summarize_results.py"
