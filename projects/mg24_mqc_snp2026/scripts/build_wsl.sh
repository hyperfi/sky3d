#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${project_dir}/build"
jobs="${SKY3D_BUILD_JOBS:-2}"

make -C "${build_dir}" -j "${jobs}" all
make -C "${build_dir}" provenance > "${project_dir}/logs/build_provenance.log"

printf 'Built %s\n' "${build_dir}/sky3d.omp"
printf 'Provenance: %s\n' "${project_dir}/logs/build_provenance.log"

