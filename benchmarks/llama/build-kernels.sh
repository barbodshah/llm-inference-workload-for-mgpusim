#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
set -euo pipefail
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
compiler=${CLANG:-clang-14}
linker=${LLD:-ld.lld-14}
device_libs=${ROCM_DEVICE_LIB_PATH:-/usr/lib/x86_64-linux-gnu/amdgcn/bitcode}
if [[ ! -f "$device_libs/opencl.bc" ]]; then
    echo "Set ROCM_DEVICE_LIB_PATH to the ROCm device-library bitcode directory." >&2
    exit 1
fi
build_dir=$(mktemp -d)
trap 'rm -rf -- "$build_dir"' EXIT
options=(-target amdgcn-amd-amdhsa -mcpu=gfx803 -mcode-object-version=2
    -x cl -cl-std=CL1.2 -Xclang -finclude-default-header
    "--rocm-device-lib-path=$device_libs" -O2 -cl-fast-relaxed-math -ffp-contract=off)
"$compiler" "${options[@]}" -c "$source_dir/kernels.cl" -o "$build_dir/kernels.o"
"$linker" -shared "$build_dir/kernels.o" -o "$build_dir/kernels.hsaco"
"$compiler" "${options[@]}" -S "$source_dir/kernels.cl" -o "$build_dir/kernels.disasm"
cp -- "$build_dir/kernels.hsaco" "$source_dir/kernels.hsaco"
cp -- "$build_dir/kernels.disasm" "$source_dir/kernels.disasm"
"$compiler" --version | head -n 1
sha256sum "$source_dir/kernels.cl" "$source_dir/kernels.hsaco"
