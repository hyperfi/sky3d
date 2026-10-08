#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
prefix="${HIGHRES_PREFIX:?Set HIGHRES_PREFIX, for example e0 or no_boost}"
source_case="${HIGHRES_SOURCE_CASE:?Set HIGHRES_SOURCE_CASE to the completed starting case}"
source_time="${HIGHRES_SOURCE_TIME:?Set HIGHRES_SOURCE_TIME to 2000, 4000, 8000, or 12000}"

case "${source_time}" in
    2000) targets=(4000 8000 12000 18000) ;;
    4000) targets=(8000 12000 18000) ;;
    8000) targets=(12000 18000) ;;
    12000) targets=(18000) ;;
    *) printf 'Unsupported HIGHRES_SOURCE_TIME=%s\n' "${source_time}" >&2; exit 2 ;;
esac

for target_time in "${targets[@]}"; do
    target_case="${prefix}_t${target_time}_n24"
    input_file="${project_dir}/configs/restart_t${target_time}_n24.in"
    expected_time="$(printf '%.2f' "${target_time}")"
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    SKY3D_CASE="${target_case}" \
    SKY3D_SOURCE_CASE="${source_case}" \
    SKY3D_INPUT="${input_file}" \
    SKY3D_EXPECTED_TIME="${expected_time}" \
        "${project_dir}/scripts/run_response_restart_wsl.sh"
    source_case="${target_case}"
done
