#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(cd "${project_dir}/../.." && pwd)"
mkdir -p "${project_dir}/logs" "${project_dir}/td"

if [[ ! -s "${project_dir}/static/20ne_sly5_zaxis.tdhf" ]]; then
  printf 'Missing aligned static state; run scripts/run_all_wsl.sh through alignment first.\n' >&2
  exit 2
fi

cases=(reference_K0 K0 K1 K2 K0_half)
pids=()
for case_name in "${cases[@]}"; do
  if [[ -s "${project_dir}/td/${case_name}/energies.res" ]] && \
     awk '!/^#/ && NF {v=$1} END {exit !(v==6000.00000)}' \
       "${project_dir}/td/${case_name}/energies.res"; then
    printf 'Already complete: %s\n' "${case_name}"
    continue
  fi
  OMP_NUM_THREADS="${OMP_THREADS_PER_CASE:-4}" \
    "${project_dir}/scripts/run_td_case_wsl.sh" "${case_name}" \
    >"${project_dir}/logs/${case_name}.driver.log" 2>&1 &
  pids+=("$!")
  printf 'Launched %s as PID %s\n' "${case_name}" "$!"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
