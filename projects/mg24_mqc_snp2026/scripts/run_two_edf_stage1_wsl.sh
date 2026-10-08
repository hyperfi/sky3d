#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runner="${project_dir}/scripts/run_response_wsl.sh"
threads="${OMP_NUM_THREADS:-18}"

run_case() {
    local case_name="$1"
    local input_name="$2"
    local source_wf="$3"
    local run_dir="${project_dir}/runs/${case_name}"
    if [[ -s "${run_dir}/energies.res" ]]; then
        local final_time
        final_time="$(awk '!/^#/ && NF {value=$1} END {print value}' "${run_dir}/energies.res")"
        if [[ "${final_time}" == "18000.00" ]]; then
            printf 'Skipping completed case %s\n' "${case_name}"
            return
        fi
    fi
    OMP_NUM_THREADS="${threads}" \
    SKY3D_CASE="${case_name}" \
    SKY3D_INPUT="${project_dir}/configs/${input_name}" \
    SKY3D_SOURCE_WF="${source_wf}" \
        "${runner}"
    local final_time
    final_time="$(awk '!/^#/ && NF {value=$1} END {print value}' "${run_dir}/energies.res")"
    if [[ "${final_time}" != "18000.00" ]]; then
        printf '%s ended at %s fm/c, expected 18000.00 fm/c\n' \
            "${case_name}" "${final_time}" >&2
        exit 4
    fi
}

sly5_wf="${project_dir}/runs/static_sly5_rotated/mg24_sly5_zaxis.tdhf"
skms_wf="${project_dir}/runs/static_skms_rotated/mg24_skms_zaxis.tdhf"

run_case no_boost_sly5_t18000 no_boost_highres_24mg_sly5.in "${sly5_wf}"
run_case e0_sly5_t18000 e0_highres_24mg_sly5.in "${sly5_wf}"
run_case no_boost_skms_t18000 no_boost_highres_24mg_skms.in "${skms_wf}"
run_case e0_skms_t18000 e0_highres_24mg_skms.in "${skms_wf}"

PYTHONPATH="/home/abhishek/.local/share/sky3d-response-deps:${project_dir}/../../Utils/Strength_Calculation" \
python3 "${project_dir}/scripts/analyze_highres_convergence.py" \
    --sly5-e0 "${project_dir}/runs/e0_sly5_t18000" \
    --sly5-reference "${project_dir}/runs/no_boost_sly5_t18000" \
    --skms-e0 "${project_dir}/runs/e0_skms_t18000" \
    --skms-reference "${project_dir}/runs/no_boost_skms_t18000" \
    --output "${project_dir}/data/processed/highres"

printf 'Completed and analyzed the two-EDF E0 comparison stage.\n'
