#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
threads="${OMP_NUM_THREADS:-18}"
source_case="e2k0_sly5_t18000"
recovery_case="e2k0_sly5_t18000_recovery2"
recovery_dir="${project_dir}/runs/${recovery_case}"

if [[ -s "${recovery_dir}/energies.res" ]] && \
   [[ "$(awk '!/^#/ && NF {value=$1} END {print value}' "${recovery_dir}/energies.res")" == "18000.00" ]]; then
    printf 'Skipping completed recovery case %s\n' "${recovery_case}"
elif [[ -d "${recovery_dir}" ]] && find "${recovery_dir}" -mindepth 1 -print -quit | grep -q .; then
    printf 'Recovery directory exists but is incomplete; refusing to overwrite: %s\n' "${recovery_dir}" >&2
    exit 11
else
    OMP_NUM_THREADS="${threads}" \
    SKY3D_CASE="${recovery_case}" \
    SKY3D_SOURCE_CASE="${source_case}" \
    SKY3D_INPUT="${project_dir}/configs/restart_e2k0_sly5_t18000.in" \
    SKY3D_EXPECTED_TIME="18000.00" \
        "${project_dir}/scripts/run_response_restart_wsl.sh"
fi

OMP_NUM_THREADS="${threads}" \
SKY3D_SLY5_E2_CASE="${recovery_case}" \
    "${project_dir}/scripts/run_two_edf_stage2_wsl.sh"
