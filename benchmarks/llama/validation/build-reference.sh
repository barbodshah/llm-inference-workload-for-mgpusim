#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
set -euo pipefail
if [[ $# != 2 ]]; then
    echo "usage: bash build-reference.sh /path/to/upstream/run.c /path/to/output" >&2
    exit 2
fi
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
reference=$(realpath -- "$1")
"${CC:-gcc}" -O2 -ffp-contract=off "-DLLAMA_REFERENCE_SOURCE=\"$reference\"" \
    "$source_dir/reference.c" -lm -o "$2"
