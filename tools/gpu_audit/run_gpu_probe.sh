#!/usr/bin/env bash
set -euo pipefail
tool_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="$(mktemp -d "$HOME/.cache/sky3d-derivative-probe-XXXXXX")"
mkdir -p "$tool_dir/evidence"
# Explicit toolkit path avoids the older distro nvcc and libraries.
/usr/local/cuda/bin/nvcc -O3 -arch=sm_120 -Xcompiler=-fopenmp \
  "$tool_dir/derivative_probe.cu" -lfftw3 -lcufft \
  -Xlinker=-rpath -Xlinker=/usr/local/cuda/lib64 -o "$build_dir/derivative_probe"
printf 'points,states,repetitions,cpu_threads,cpu_ms,gpu_resident_ms,gpu_transfer_ms,resident_speedup,transfer_speedup,relative_l2,max_abs\n' > "$tool_dir/evidence/gpu.csv"
for shape in '24 1 20' '24 20 10' '32 48 5' '48 48 3'; do
  read -r points states repetitions <<< "$shape"
  "$build_dir/derivative_probe" "$points" "$states" "$repetitions" >> "$tool_dir/evidence/gpu.csv"
done
printf 'Probe executable retained at %s\n' "$build_dir/derivative_probe"
cat "$tool_dir/evidence/gpu.csv"
