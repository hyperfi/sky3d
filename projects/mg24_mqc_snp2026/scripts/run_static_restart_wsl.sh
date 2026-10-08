#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_wf="${project_dir}/runs/static_svbas/mg24_svbas.tdhf"
run_dir="${project_dir}/runs/static_svbas_restart"
log_file="${project_dir}/logs/static_svbas_restart.log"
time_file="${project_dir}/logs/static_svbas_restart.time"
input_file="${project_dir}/configs/static_24mg_svbas_restart.in"
executable="${project_dir}/build/sky3d.omp"

if [[ ! -x "${executable}" ]]; then
    printf 'Missing executable: %s\n' "${executable}" >&2
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
cp "${source_wf}" "${run_dir}/mg24_svbas_restart.tdhf"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

printf 'Starting static restart with x0dmp=0.10 and OMP_NUM_THREADS=%s\n' "${OMP_NUM_THREADS}"
(
    cd "${run_dir}"
    /usr/bin/time -v -o "${time_file}" "${executable}"
) > "${log_file}" 2>&1
printf 'Completed static restart. Log: %s\n' "${log_file}"

