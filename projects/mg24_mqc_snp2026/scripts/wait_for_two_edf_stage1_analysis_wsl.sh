#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cases=(
    no_boost_sly5_t18000
    e0_sly5_t18000
    no_boost_skms_t18000
    e0_skms_t18000
)

while true; do
    complete=1
    for case_name in "${cases[@]}"; do
        time_file="${project_dir}/logs/${case_name}.time"
        energy_file="${project_dir}/runs/${case_name}/energies.res"
        if ! grep -q 'Exit status: 0' "${time_file}" 2>/dev/null; then
            complete=0
            continue
        fi
        final_time="$(awk '!/^#/ && NF {value=$1} END {print value}' "${energy_file}")"
        if [[ "${final_time}" != "18000.00" ]]; then
            printf '%s has a clean exit but ended at %s fm/c\n' \
                "${case_name}" "${final_time}" >&2
            exit 12
        fi
    done
    if [[ "${complete}" -eq 1 ]]; then
        break
    fi
    if ! pgrep -f 'run_two_edf_stage1_wsl.sh|sky3d.omp' >/dev/null 2>&1; then
        printf 'Stage-1 production stopped before all four cases completed.\n' >&2
        exit 13
    fi
    sleep 60
done

# Avoid racing a stage-1 launcher that includes its own final analysis.
while pgrep -f '[r]un_two_edf_stage1_wsl.sh' >/dev/null 2>&1; do
    sleep 10
done

PYTHONPATH="/home/abhishek/.local/share/sky3d-response-deps:${project_dir}/../../Utils/Strength_Calculation" \
python3 "${project_dir}/scripts/analyze_highres_convergence.py" \
    --sly5-e0 "${project_dir}/runs/e0_sly5_t18000" \
    --sly5-reference "${project_dir}/runs/no_boost_sly5_t18000" \
    --skms-e0 "${project_dir}/runs/e0_skms_t18000" \
    --skms-reference "${project_dir}/runs/no_boost_skms_t18000" \
    --output "${project_dir}/data/processed/highres"

python3 "${project_dir}/scripts/summarize_two_edf_e0_gate.py"
printf 'Stage-1 analysis is ready for review; stage 2 was not started.\n'
