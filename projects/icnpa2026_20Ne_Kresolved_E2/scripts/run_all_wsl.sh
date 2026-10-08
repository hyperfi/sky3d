#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(cd "${project_dir}/../.." && pwd)"
mkdir -p "${project_dir}/logs" "${project_dir}/response" \
  "${project_dir}/figures" "${project_dir}/summary"
{
  printf 'git_commit=%s\n' "$(git -C "${repo_dir}" rev-parse HEAD)"
  printf 'git_describe=%s\n' "$(git -C "${repo_dir}" describe --always --dirty)"
  printf 'executable=%s\n' "${repo_dir}/projects/mg24_mqc_snp2026/build/sky3d.omp"
  sha256sum "${repo_dir}/projects/mg24_mqc_snp2026/build/sky3d.omp"
  date --iso-8601=seconds
  uname -a
} >"${project_dir}/logs/provenance.txt"

if [[ ! -s "${project_dir}/static/20ne_sly5_zaxis.tdhf" ]]; then
  if [[ ! -s "${project_dir}/static/20ne_sly5.tdhf" ]]; then
    "${project_dir}/scripts/run_static_wsl.sh"
  fi
  python3 "${repo_dir}/projects/mg24_mqc_snp2026/scripts/rotate_wf_axis.py" \
    "${project_dir}/static/20ne_sly5.tdhf" \
    "${project_dir}/static/20ne_sly5_zaxis.tdhf" \
    --metadata "${project_dir}/summary/axis_rotation.json"
fi

for case_name in reference_K0 K0 K1 K2 K0_half; do
  if [[ ! -s "${project_dir}/td/${case_name}/energies.res" ]]; then
    "${project_dir}/scripts/run_td_case_wsl.sh" "${case_name}"
  fi
done

"${project_dir}/scripts/analyze_wsl.sh"
printf 'Raw calculations and response analysis complete.\n'
