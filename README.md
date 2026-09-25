# LLM inference workload for MGPUSim

A validated, single-GPU FP32 Llama inference workload for
[MGPUSim](https://github.com/sarchlab/mgpusim). It executes transformer inference
inside MGPUSim's functional or detailed-timing GPU simulation and reports the
simulator's instruction, cache, memory, and timing metrics.

The repository is self-contained for normal use. Its precompiled `gfx803`
kernels are embedded in the Go benchmark, so running it does not require ROCm,
an AMD GPU, WSL, or an external kernel binary.

The Go module path intentionally remains `github.com/sarchlab/mgpusim/v3`.
Keeping the upstream module identity lets this repository build its embedded
MGPUSim fork without rewriting internal imports.

## What is included

- Legacy `llama2.c` FP32 checkpoint loading.
- Token embedding, RMSNorm, Q/K/V projections, RoPE, grouped-query attention,
  persistent KV caches, SwiGLU, residual connections, and vocabulary logits.
- Functional simulation and MGPUSim detailed-timing simulation.
- Fixed-token workloads for repeatable architectural experiments.
- CPU verification of every token's logits and complete KV-cache state.
- A deterministic synthetic model that runs without downloading model weights.
- A phase CSV separating initialization, inference, and verification time.
- Focused extensions to MGPUSim's GCN3 emulator for instructions emitted by
  the validated Clang toolchain.

This is a benchmark workload rather than a chat application. Prompt
tokenization, sampling, quantized/GGUF models, batching, and multi-GPU model
partitioning are outside its current scope.

## Requirements

- Go 1.22 or newer.
- Windows, Linux, or macOS for functional simulation. Detailed simulation does
  not require physical GPU hardware.

WSL/Linux and Clang are required only if you edit and rebuild the OpenCL kernels.

## Quick start

Clone the repository, enter its root directory, and run the embedded synthetic
workload:

```bash
go run ./samples/llama -verify
```

Run the same workload using detailed timing and collect MGPUSim metrics:

```bash
go run ./samples/llama \
  -timing \
  -report-all \
  -metric-file-name llama-metrics \
  -phase-metrics llama-phases.csv
```

Use separate correctness and performance runs. `-verify` copies the full KV
cache to the host after every token and therefore changes traffic and cache
state. For performance measurements, omit it:

```bash
go run ./samples/llama -timing -report-all -phase-metrics llama-phases.csv
```

## Run a real checkpoint

The workload accepts the original unquantized FP32 `llama2.c` `.bin` format.
Model files are intentionally excluded from Git.

Download Karpathy's public Stories 15M checkpoint on Linux/macOS:

```bash
curl -L \
  https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin \
  -o models/stories15M.bin
```

Or in PowerShell:

```powershell
Invoke-WebRequest `
  -Uri "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin" `
  -OutFile "models/stories15M.bin"
```

Run three fixed token IDs in functional mode with full verification:

```bash
go run ./samples/llama \
  -checkpoint models/stories15M.bin \
  -tokens 1,2,3 \
  -verify
```

The token IDs are benchmark inputs, not a text prompt. Each ID must be within
the checkpoint's vocabulary, and the sequence must fit its context length.

## Command-line options

The workload-specific options are:

| Option | Meaning |
| --- | --- |
| `-checkpoint PATH` | Legacy FP32 checkpoint; empty selects the synthetic model |
| `-tokens IDS` | Comma-separated token IDs, starting at position zero |
| `-phase-metrics PATH` | Write per-phase simulated and wall-clock times to CSV |
| `-kernels PATH` | Override the embedded HSACO for kernel development |

Standard MGPUSim runner flags remain available, including `-timing`, `-verify`,
`-report-all`, `-report-inst-count`, and cache/DRAM reporting flags. The workload
currently accepts one physical simulated GPU. `-use-unified-memory` changes its
allocation mode without partitioning work across GPUs.

## Build and test

```bash
go build ./samples/llama
go test ./benchmarks/llama ./emu ./insts ./driver ./kernels ./timing/cu
go vet ./benchmarks/llama ./samples/llama
```

The benchmark tests execute the embedded GPU instructions through MGPUSim; they
do not silently replace GPU execution with the CPU reference. They cover kernel
edge cases, cross-wavefront reductions, long-context GQA, full inference, and
normal and unified-memory allocations.

## Rebuild the GPU kernels

The committed HSACO was built and validated using Ubuntu 22.04, Clang/LLD 14,
and ROCm device libraries 5.0.0:

```bash
sudo apt-get update
sudo apt-get install clang-14 lld-14 rocm-device-libs make
cd benchmarks/llama
bash build-kernels.sh
cd ../..
go test ./benchmarks/llama
```

Rebuild the Go executable after changing the kernels because `kernels.hsaco` is
embedded at compile time. Changing the compiler or flags requires renewed ABI,
instruction, and numerical validation.

## Repository layout

| Path | Purpose |
| --- | --- |
| `benchmarks/llama/` | Model loader, CPU reference, GPU host orchestration, kernels, and tests |
| `samples/llama/` | Command-line entry point using the standard MGPUSim runner |
| `emu/` and `insts/` | MGPUSim GCN3 emulator and the required ISA compatibility extensions |
| `samples/runner/` | Functional/timing platform construction and metric reporting |
| `docs/MGPUSIM_UPSTREAM_README.md` | Original MGPUSim project overview and citation |

Detailed benchmark internals are documented in
[`benchmarks/llama/README.md`](benchmarks/llama/README.md). Reproducibility data,
hashes, test evidence, and limitations are in
[`benchmarks/llama/VALIDATION.md`](benchmarks/llama/VALIDATION.md).

## Validation status

The workload has passed:

- Complete three-token synthetic inference in functional and detailed timing.
- Normal and unified-memory allocation modes.
- Stories 15M functional inference with logits and KV-cache verification.
- Numerical comparison with the original upstream C implementation.
- MGPUSim emulator, decoder, driver, kernel-loader, and compute-unit tests.
- A regression run of MGPUSim's existing FIR workload in detailed timing.

The validated Stories 15M input IDs `[1, 2, 3]` produced argmax IDs
`[9038, 471, 471]`. Full Stories 15M detailed timing, larger models, and
physical-GPU performance calibration have not been completed.

## Attribution and licensing

The surrounding MGPUSim source remains under its original MIT license in
[`LICENSE`](LICENSE). The Llama benchmark adapts
[`Uzair-90/llama-inference-hip`](https://github.com/Uzair-90/llama-inference-hip)
and is distributed under GPL-3.0-only, with its license and attribution in
`benchmarks/llama/`. The embedded kernel links ROCm device libraries; their
redistribution notices are included beside the benchmark.

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) before redistributing the
repository. If you publish research results, cite MGPUSim as described in
[`CITATION.md`](CITATION.md).
