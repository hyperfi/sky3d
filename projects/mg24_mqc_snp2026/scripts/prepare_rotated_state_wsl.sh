#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_wf="${project_dir}/runs/static_svbas/mg24_svbas.tdhf"
output_dir="${project_dir}/runs/static_svbas_rotated"
output_wf="${output_dir}/mg24_svbas_zaxis.tdhf"
metadata="${project_dir}/data/processed/axis_rotation.json"

if [[ ! -f "${source_wf}" ]]; then
    printf 'Missing source wave function: %s\n' "${source_wf}" >&2
    exit 2
fi
mkdir -p "${output_dir}"
python3 "${project_dir}/scripts/rotate_wf_axis.py" \
    "${source_wf}" "${output_wf}" --metadata "${metadata}"

