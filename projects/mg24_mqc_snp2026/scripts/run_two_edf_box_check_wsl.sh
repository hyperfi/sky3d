#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
static_runner="${project_dir}/scripts/run_static_wsl.sh"
response_runner="${project_dir}/scripts/run_response_wsl.sh"
threads="${OMP_NUM_THREADS:-18}"
gate="${project_dir}/data/processed/highres/core_stage_go.flag"
sly5_e2_case="${SKY3D_SLY5_E2_CASE:-e2k0_sly5_t18000_recovery2}"
run_skms_n32="${SKY3D_RUN_SKMS_N32:-0}"

if [[ ! -s "${gate}" ]] || ! grep -qx 'GO' "${gate}"; then
    printf 'The box check is gated until the two-EDF core and linearity audit passes.\n' >&2
    exit 11
fi

run_static_case() {
    local case_name="$1"
    local input_name="$2"
    local wavefunction="$3"
    if [[ -s "${project_dir}/runs/${case_name}/${wavefunction}" ]]; then
        printf 'Skipping completed static case %s\n' "${case_name}"
        return
    fi
    OMP_NUM_THREADS="${threads}" \
    SKY3D_CASE="${case_name}" \
    SKY3D_INPUT="${project_dir}/configs/${input_name}" \
        "${static_runner}"
}

run_response_case() {
    local case_name="$1"
    local input_name="$2"
    local source_wf="$3"
    local run_dir="${project_dir}/runs/${case_name}"
    if [[ -s "${run_dir}/energies.res" ]]; then
        local final_time
        final_time="$(awk '!/^#/ && NF {value=$1} END {print value}' "${run_dir}/energies.res")"
        if [[ "${final_time}" == "8000.00" ]]; then
            printf 'Skipping completed response case %s\n' "${case_name}"
            return
        fi
    fi
    OMP_NUM_THREADS="${threads}" \
    SKY3D_CASE="${case_name}" \
    SKY3D_INPUT="${project_dir}/configs/${input_name}" \
    SKY3D_SOURCE_WF="${source_wf}" \
        "${response_runner}"
}

run_static_case static_sly5_n32 static_24mg_sly5_n32.in mg24_sly5_zaxis_n32.tdhf

sly5_n32_wf="${project_dir}/runs/static_sly5_n32/mg24_sly5_zaxis_n32.tdhf"
run_response_case no_boost_sly5_n32_t8000 no_boost_highres_24mg_sly5_n32.in "${sly5_n32_wf}"
run_response_case e0_sly5_n32_t8000 e0_highres_24mg_sly5_n32.in "${sly5_n32_wf}"

large_args=(
    --sly5-large-e0 "${project_dir}/runs/e0_sly5_n32_t8000"
    --sly5-large-reference "${project_dir}/runs/no_boost_sly5_n32_t8000"
)
if [[ "${run_skms_n32}" == "1" ]]; then
    run_static_case static_skms_n32 static_24mg_skms_n32.in mg24_skms_zaxis_n32.tdhf
    skms_n32_wf="${project_dir}/runs/static_skms_n32/mg24_skms_zaxis_n32.tdhf"
    run_response_case no_boost_skms_n32_t8000 no_boost_highres_24mg_skms_n32.in "${skms_n32_wf}"
    run_response_case e0_skms_n32_t8000 e0_highres_24mg_skms_n32.in "${skms_n32_wf}"
    large_args+=(
        --skms-large-e0 "${project_dir}/runs/e0_skms_n32_t8000"
        --skms-large-reference "${project_dir}/runs/no_boost_skms_n32_t8000"
    )
fi

PYTHONPATH="/home/abhishek/.local/share/sky3d-response-deps:${project_dir}/../../Utils/Strength_Calculation" \
python3 "${project_dir}/scripts/analyze_highres_convergence.py" \
    --sly5-e0 "${project_dir}/runs/e0_sly5_t18000" \
    --sly5-reference "${project_dir}/runs/no_boost_sly5_t18000" \
    --skms-e0 "${project_dir}/runs/e0_skms_t18000" \
    --skms-reference "${project_dir}/runs/no_boost_skms_t18000" \
    --sly5-e2 "${project_dir}/runs/${sly5_e2_case}" \
    --skms-e2 "${project_dir}/runs/e2k0_skms_t18000" \
    --sly5-half "${project_dir}/runs/e0_half_sly5_t18000" \
    --skms-half "${project_dir}/runs/e0_half_skms_t18000" \
    "${large_args[@]}" \
    --output "${project_dir}/data/processed/highres"

printf 'Completed and analyzed the requested 32^3 box checks at T=8000 fm/c.\n'
