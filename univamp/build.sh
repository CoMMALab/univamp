#!/usr/bin/env bash
# Build the UniVAMP managed-memory allocator shared library for Jetson Thor (sm_110).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/libunivamp_managed.so"
nvcc -O3 -shared -Xcompiler -fPIC \
     -gencode arch=compute_110,code=sm_110 \
     -o "$OUT" "$HERE/csrc/managed_allocator.cu"
echo "built $OUT"

# Merged seed-expansion kernel (raw managed VAMP waypoints -> expanded device seed, zero-copy).
OUT_SE="$HERE/libunivamp_seedexpand.so"
nvcc -O3 -shared -Xcompiler -fPIC \
     -gencode arch=compute_110,code=sm_110 \
     -o "$OUT_SE" "$HERE/csrc/seed_expand.cu"
echo "built $OUT_SE"
