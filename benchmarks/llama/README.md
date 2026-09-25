# Llama FP32 inference benchmark

Single-GPU inference is implemented and validated in MGPUSim 3.0.2. The GCN3
kernel binary is embedded in Go: ordinary builds and runs need no ROCm, WSL,
external HSACO, or hardware AMD GPU. WSL/Linux is used to rebuild kernels and
run the independent upstream C reference.

This ports the sequential forward pass from
[llama-inference-hip](https://github.com/Uzair-90/llama-inference-hip).
See NOTICE, LICENSE, ROCM-DEVICE-LIBS-LICENSE, and VALIDATION.md.

## Run on this Windows checkout

From the Projects directory in PowerShell:

```powershell
.\mgpusim-3.0.2\samples\llama\llama.exe -timing -verify
.\mgpusim-3.0.2\samples\llama\llama.exe -checkpoint .\.build-cache\stories15M.bin -tokens 1,2,3 -verify
.\mgpusim-3.0.2\samples\llama\llama.exe -timing -report-all -phase-metrics llama-phases.csv
```

To build from source, use Go 1.22+ from the mgpusim-3.0.2 directory:

```bash
go build -o samples/llama/llama.exe ./samples/llama
go run ./samples/llama -verify
go test -v ./benchmarks/llama ./emu ./insts
```

Go is installed at `C:\Program Files\Go\bin\go.exe` on this machine.
If absent from PATH, invoke that path with PowerShell's `&` operator.

## Scope

- One physical simulated GPU; optional `-use-unified-memory` allocations.
- FP32 legacy llama2.c checkpoints, shared or separate classifier weights.
- RMSNorm, RoPE, grouped-query attention, SwiGLU, residuals, persistent KV caches.
- Fixed input token IDs at contiguous positions starting at zero.
- Vocabulary logits and argmax logging after each token.
- No checkpoint supplied: a deterministic, untrained two-layer synthetic model.
- `-kernels` optionally overrides the embedded binary for development.

This is sequential token inference, not optimized batched prefill.
Tokenization, text generation/sampling, GGUF, quantization, batching, and
multi-GPU partitioning are not implemented. Input IDs and context length are
validated. The synthetic model does not generate meaningful language.

## Rebuild kernels in WSL

Validated: Ubuntu 22.04.5, Clang/LLD 14.0.0-1ubuntu1.1, and
rocm-device-libs 5.0.0-1.

```bash
sudo apt-get update
sudo apt-get install clang-14 lld-14 rocm-device-libs gcc make
cd '/mnt/d/Barbod/Hardware Lab/Projects/mgpusim-3.0.2/benchmarks/llama'
bash build-kernels.sh
```

The script emits kernels.hsaco and kernels.disasm for gfx803, code-object v2.
It uses `-O2 -cl-fast-relaxed-math -ffp-contract=off`, following the upstream
HIP project's fast-math policy. Results are tolerance-checked rather than
bit-identical to CPU math. CLANG, LLD, and ROCM_DEVICE_LIB_PATH can override
tool locations. Rebuild the Go executable to update its embedded kernel copy.

Clang removes unused implicit arguments. Validated argument sizes are 88 bytes
for matmul/RoPE, 80 for add/SwiGLU, 28 for RMSNorm, and 64 for attention.
Kernels using global IDs retain seven implicit 64-bit slots, all zero here.
Static scratch is 256 floats for RMSNorm/softmax and 128 for attention.
The loader rejects incompatible headers and argument sizes.

## Validation

Default Go tests execute the embedded kernels, including irregular shapes,
cross-wavefront reductions, in-place RMSNorm, and GQA through position 256.
Full synthetic inference tests run three tokens using normal and unified
allocations. MGPUSIM_LLAMA_HSACO can override the binary under test.

`-verify` checks each token's logits and full KV cache against the Go CPU
reference using `abs(error) <= 1e-4 + 1e-3 * abs(reference)`, rejecting NaN/Inf.
Synthetic detailed timing and real stories15M functional runs have passed.

For an independent comparison to the original C implementation, in Linux:

```bash
bash benchmarks/llama/validation/build-reference.sh \
  /path/to/llama-inference-hip-main/run.c /tmp/llama-reference
MGPUSIM_LLAMA_REFERENCE=/tmp/llama-reference \
MGPUSIM_LLAMA_CHECKPOINT=/path/to/stories15M.bin \
  go test -v ./benchmarks/llama -run TestUpstreamReference
```

This optional test checks both synthetic classifier layouts and the real
checkpoint when supplied. Linux Go can run it directly; alternatively Windows
Go can cross-compile the test binary with GOOS=linux, GOARCH=amd64, CGO_ENABLED=0
for WSL execution. See VALIDATION.md for tested inputs and limitations.

## Metrics

Use `-timing -report-all` for runner metrics and `-phase-metrics path.csv` for
initialization/per-token simulated seconds and simulator wall-clock seconds.
Functional emulation timestamps are not performance estimates.

Initialization includes weights and cache setup. A forward phase includes the
embedding copy, dispatches, and logits readback. The driver implements the
embedding copy as a kernel. CPU tokenization/sampling is not modeled.

Verification copies KV caches and perturbs subsequent cache state. It has a
separate phase, but subtracting its duration cannot undo these effects.
Use a separate run without `-verify` for performance measurements.

KV allocation in bytes is:
`2 * layers * sequence_length * (dim / heads * kv_heads) * 4`.
The full checkpoint context is allocated even for short sequences. Large
models may exceed host/simulated memory. Full real-model detailed timing has
not been validated and can take substantially longer than functional runs.
