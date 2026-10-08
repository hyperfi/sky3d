#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
printf 'This legacy entry point now dispatches only the gated SLy5/SkM* stage.\n'
exec "${project_dir}/scripts/run_two_edf_stage2_wsl.sh"
