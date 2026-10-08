#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${project_dir}/data/experimental/source"
output_file="${output_dir}/fig5_bahini2022_arxiv.eps"
expected_sha="92009487c140d8b8169de8fe3810f281619df0dfc014f021bb6dab77ae3fdba3"

if [[ -f "${output_file}" ]]; then
    actual_sha="$(sha256sum "${output_file}" | awk '{print $1}')"
    if [[ "${actual_sha}" != "${expected_sha}" ]]; then
        printf 'Refusing to replace source with unexpected existing hash: %s\n' "${actual_sha}" >&2
        exit 3
    fi
    printf 'Source already present with expected SHA-256: %s\n' "${output_file}"
    exit 0
fi

temporary_dir="$(mktemp -d /tmp/mg24-bahini-source.XXXXXX)"
trap 'rm -rf -- "${temporary_dir}"' EXIT
curl -fsSL https://export.arxiv.org/e-print/2111.07105 \
    -o "${temporary_dir}/source.tar"
tar -xf "${temporary_dir}/source.tar" -C "${temporary_dir}" fig5.eps
actual_sha="$(sha256sum "${temporary_dir}/fig5.eps" | awk '{print $1}')"
if [[ "${actual_sha}" != "${expected_sha}" ]]; then
    printf 'Downloaded source hash mismatch: %s\n' "${actual_sha}" >&2
    exit 4
fi

mkdir -p "${output_dir}"
install -m 0644 "${temporary_dir}/fig5.eps" "${output_file}"
printf 'Preserved %s\n' "${output_file}"
