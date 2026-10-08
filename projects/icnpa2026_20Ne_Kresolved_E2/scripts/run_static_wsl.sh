#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(cd "${project_dir}/../.." && pwd)"
run_dir="${project_dir}/static"
input="${project_dir}/inputs/static_20ne_sly5.in"
exe="${repo_dir}/projects/mg24_mqc_snp2026/build/sky3d.omp"

if [[ -d "${run_dir}" ]] && find "${run_dir}" -mindepth 1 -print -quit | grep -q .; then
  printf 'Refusing to overwrite non-empty static directory: %s\n' "${run_dir}" >&2
  exit 3
fi
mkdir -p "${run_dir}" "${project_dir}/logs"
cp "${input}" "${run_dir}/for005"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-18}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"
(
  cd "${run_dir}"
  /usr/bin/time -v -o "${project_dir}/logs/static.time" "${exe}"
) >"${project_dir}/logs/static.log" 2>&1
test -s "${run_dir}/20ne_sly5.tdhf"
printf 'Static calculation completed: %s\n' "${run_dir}/20ne_sly5.tdhf"

