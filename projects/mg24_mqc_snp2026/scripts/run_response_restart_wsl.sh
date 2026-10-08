#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case_name="${SKY3D_CASE:?Set SKY3D_CASE to the new response case name}"
source_case="${SKY3D_SOURCE_CASE:?Set SKY3D_SOURCE_CASE to the completed source case}"
input_file="${SKY3D_INPUT:?Set SKY3D_INPUT to an absolute WSL restart input path}"
expected_time="${SKY3D_EXPECTED_TIME:-}"
source_dir="${project_dir}/runs/${source_case}"
run_dir="${project_dir}/runs/${case_name}"
log_file="${project_dir}/logs/${case_name}.log"
time_file="${project_dir}/logs/${case_name}.time"
executable="${project_dir}/build/sky3d.omp"
restart_preparer="${project_dir}/scripts/prepare_response_restart.py"

if [[ ! -x "${executable}" ]]; then
    printf 'Missing executable: %s\n' "${executable}" >&2
    exit 2
fi
if [[ ! -f "${input_file}" ]]; then
    printf 'Missing input: %s\n' "${input_file}" >&2
    exit 2
fi
if [[ ! -d "${source_dir}" ]]; then
    printf 'Missing source run directory: %s\n' "${source_dir}" >&2
    exit 2
fi
if [[ ! -f "${restart_preparer}" ]]; then
    printf 'Missing restart preparation helper: %s\n' "${restart_preparer}" >&2
    exit 2
fi
if [[ -d "${run_dir}" ]] && find "${run_dir}" -mindepth 1 -print -quit | grep -q .; then
    printf 'Refusing to overwrite non-empty run directory: %s\n' "${run_dir}" >&2
    exit 3
fi

mapfile -t restart_files < <(find "${source_dir}" -maxdepth 1 -type f -name '*restart.tdhf' -print)
if [[ "${#restart_files[@]}" -ne 1 ]]; then
    printf 'Expected exactly one restart wave function in %s; found %s\n' \
        "${source_dir}" "${#restart_files[@]}" >&2
    exit 4
fi

python3 "${restart_preparer}" \
    --source-dir "${source_dir}" \
    --run-dir "${run_dir}" \
    --restart-file "${restart_files[0]}"
cp "${input_file}" "${run_dir}/for005"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

printf 'Restarting %s from %s with OMP_NUM_THREADS=%s\n' \
    "${case_name}" "${source_case}" "${OMP_NUM_THREADS}"
(
    cd "${run_dir}"
    /usr/bin/time -v -o "${time_file}" "${executable}"
) > "${log_file}" 2>&1

if [[ ! -s "${run_dir}/restart.tdhf" ]]; then
    printf 'Restart wave function was not written: %s\n' "${run_dir}/restart.tdhf" >&2
    exit 5
fi
if [[ -n "${expected_time}" ]]; then
    final_time="$(awk '!/^#/ && NF {value=$1} END {print value}' "${run_dir}/energies.res")"
    if [[ "${final_time}" != "${expected_time}" ]]; then
        printf 'Expected final time %s fm/c, found %s fm/c\n' \
            "${expected_time}" "${final_time}" >&2
        exit 6
    fi
fi
python3 - "${run_dir}/energies.res" <<'PY'
import sys
from pathlib import Path

times = []
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    stripped = line.lstrip()
    if not stripped or stripped.startswith("#"):
        continue
    try:
        times.append(float(stripped.split()[0]))
    except (ValueError, IndexError):
        continue
if not times or any(right <= left for left, right in zip(times, times[1:])):
    raise SystemExit("Recovered energies.res is not strictly increasing")
PY
printf 'Completed response case %s. Log: %s\n' "${case_name}" "${log_file}"
