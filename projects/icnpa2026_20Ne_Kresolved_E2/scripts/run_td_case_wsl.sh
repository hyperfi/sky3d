#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(cd "${project_dir}/../.." && pwd)"
case_name="${1:?usage: run_td_case_wsl.sh K0|K1|K2|K0_half|reference_K0}"
case_key="$(printf '%s' "${case_name}" | tr '[:upper:]' '[:lower:]')"
input="${project_dir}/inputs/td_${case_key}.in"
run_dir="${project_dir}/td/${case_name}"
exe="${repo_dir}/projects/mg24_mqc_snp2026/build/sky3d.omp"

test -x "${exe}"
test -s "${project_dir}/static/20ne_sly5_zaxis.tdhf"
test -f "${input}"
if [[ -d "${run_dir}" ]] && find "${run_dir}" -mindepth 1 -print -quit | grep -q .; then
  printf 'Refusing to overwrite non-empty TD directory: %s\n' "${run_dir}" >&2
  exit 3
fi
mkdir -p "${run_dir}" "${project_dir}/logs"
cp "${input}" "${run_dir}/for005"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-18}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"
(
  cd "${run_dir}"
  /usr/bin/time -v -o "${project_dir}/logs/${case_name}.time" "${exe}"
) >"${project_dir}/logs/${case_name}.log" 2>&1
final_time="$(awk '!/^#/ && NF {v=$1} END {print v}' "${run_dir}/energies.res")"
python3 - "${final_time}" <<'PY'
import math, sys
if not math.isclose(float(sys.argv[1]), 6000.0, rel_tol=0.0, abs_tol=1e-6):
    raise SystemExit(f"unexpected final time {sys.argv[1]} fm/c")
PY
printf 'TD calculation completed: %s at %s fm/c\n' "${case_name}" "${final_time}"
