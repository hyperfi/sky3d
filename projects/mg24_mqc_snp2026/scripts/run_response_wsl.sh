#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case_name="${SKY3D_CASE:?Set SKY3D_CASE to the response case name}"
input_file="${SKY3D_INPUT:?Set SKY3D_INPUT to an absolute WSL input path}"
source_wf="${SKY3D_SOURCE_WF:-${project_dir}/runs/static_svbas_rotated/mg24_svbas_zaxis.tdhf}"
run_dir="${project_dir}/runs/${case_name}"
log_file="${project_dir}/logs/${case_name}.log"
time_file="${project_dir}/logs/${case_name}.time"
executable="${project_dir}/build/sky3d.omp"

if [[ ! -x "${executable}" ]]; then
    printf 'Missing executable: %s\n' "${executable}" >&2
    exit 2
fi
if [[ ! -f "${input_file}" ]]; then
    printf 'Missing input: %s\n' "${input_file}" >&2
    exit 2
fi
if [[ ! -f "${source_wf}" ]]; then
    printf 'Missing source wave function: %s\n' "${source_wf}" >&2
    exit 2
fi
if [[ -d "${run_dir}" ]] && find "${run_dir}" -mindepth 1 -print -quit | grep -q .; then
    printf 'Refusing to overwrite non-empty run directory: %s\n' "${run_dir}" >&2
    exit 3
fi

mkdir -p "${run_dir}"
cp "${input_file}" "${run_dir}/for005"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

printf 'Starting response case %s with OMP_NUM_THREADS=%s\n' "${case_name}" "${OMP_NUM_THREADS}"
(
    cd "${run_dir}"
    /usr/bin/time -v -o "${time_file}" "${executable}"
) > "${log_file}" 2>&1
printf 'Completed response case %s. Log: %s\n' "${case_name}" "${log_file}"

