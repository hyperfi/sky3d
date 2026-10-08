#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target=6000
cases=(reference_K0 K0 K1 K2 K0_half)
mkdir -p "${project_dir}/logs"

while :; do
  remaining=0
  for case_name in "${cases[@]}"; do
    run_dir="${project_dir}/td/${case_name}"
    energy_file="${run_dir}/energies.res"
    if [[ ! -f "${energy_file}" ]]; then
      remaining=$((remaining + 1))
      continue
    fi
    final_time="$(awk '!/^#/ && NF {value=$1} END {print value+0}' "${energy_file}")"
    if awk -v value="${final_time}" -v target="${target}" 'BEGIN {exit !(value >= target)}'; then
      for proc_dir in /proc/[0-9]*; do
        pid="${proc_dir##*/}"
        [[ -r "${proc_dir}/comm" ]] || continue
        [[ "$(<"${proc_dir}/comm")" == "sky3d.omp" ]] || continue
        cwd="$(readlink -f "${proc_dir}/cwd" 2>/dev/null || true)"
        if [[ "${cwd}" == "${run_dir}" ]]; then
          kill -TERM "${pid}"
          printf '%s case=%s endpoint=%s pid=%s action=TERM\n' \
            "$(date --iso-8601=seconds)" "${case_name}" "${final_time}" "${pid}" \
            >>"${project_dir}/logs/stop_at_6000.log"
        fi
      done
    else
      remaining=$((remaining + 1))
    fi
  done
  [[ "${remaining}" -eq 0 ]] && break
  sleep 1
done

printf '%s all cases reached %s fm/c\n' "$(date --iso-8601=seconds)" "${target}" \
  >>"${project_dir}/logs/stop_at_6000.log"
