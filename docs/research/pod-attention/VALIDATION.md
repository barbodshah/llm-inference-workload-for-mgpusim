# Validation record: steps 1-3

Executed on 2026-10-31. Research base revision:
`9fec2aa06001b21ddd6b83089189667150699baa`, with the uncommitted changes and new
files identified by `results/source-sha256.txt`.

## Environment

- Windows Go 1.27.1, module `github.com/sarchlab/mgpusim/v3`.
- Kernel build: existing Ubuntu 22.04 WSL distribution, Ubuntu Clang 14.0.0,
  LLD 14, ROCm device libraries in the existing system toolchain.
- Target: legacy gfx803, wave64, code-object v2; `-O2 -ffp-contract=off`.
- No physical GPU was used and no hardware performance calibration is claimed.

`results/source-sha256.txt` fingerprints source files and the embedded code
object. The kernel build script also prints the Clang version and kernel hashes.
The disassembly is regenerable with that script and is ignored by Git.

## Executed checks

The full command and successful output are retained in `results/tests.txt`:

```powershell
go test ./benchmarks/attention ./emu ./benchmarks/llama ./timing/cp/... ./timing/cu ./driver ./kernels ./insts -count=1 -v
go vet ./benchmarks/attention ./samples/attention ./samples/attention-audit ./emu
```

External-checkpoint/reference tests retain their existing optional skip rules;
passing this command does not mean new real-model timing validation.

New attention tests execute the embedded GPU instructions and compare against a
stable FP64 CPU reference with `abs(error) <= 1e-4 + 1e-3 * abs(reference)`.
NaN and infinite outputs fail. Coverage includes:

- Query/KV head sharing ratios 1, 2, and 4.
- Head dimensions 1, 7, 32, and 64.
- Contexts through 257 tokens and ragged decode batches with poisoned padding.
- Single-token contexts, causal multi-query prefill, partial query/key tiles.
- Serial/concurrent execution and isolated prefill/decode phases.
- Normal and unified allocations.
- Large QK scores that would overflow an unstabilized exponential.
- Kernel ABI and scratch-free code-object validation.
- Invalid configurations and numerical-verification rejection of NaN.
- Detailed concurrent attention, with all 16 workgroup receipt/completion
  intervals paired and no pending trace entries.

The emulation fix has dedicated destination-routing and failed-send retry tests.
Existing Llama, emulator/decoder, driver, dispatcher, resource, and CU tests pass.

## Concurrency evidence

Default probe configuration: four workgroups/kernel, 64 threads/workgroup,
512 dependent integer iterations/lane. Two kernels are launched in each of two
rounds, with exact output verification after each round.

| Mode | Verified launches | Completed workgroups | CUs with mixed-kernel residency across both rounds | Pending intervals |
| --- | ---: | ---: | ---: | ---: |
| Same queue | 4 | 16 | 0 | 0 |
| Separate queues | 4 | 16 | 8 | 0 |

Raw simulation-clock intervals are in `results/same-queue.csv` and
`results/separate-queues.csv`. Kernel identity is the allocated kernel-object
address, so separate launches of the same probe are distinguishable.

## Initial resource-profile evidence

Fresh detailed platform for each isolated phase, verification disabled:
8 query heads, 2 KV heads, dimension 32, prefill chunk/context 128, and 32 decode
requests each with context 128. Each isolated phase launches 256 workgroups.
Prefill groups have four wavefronts; decode groups have one.

| Phase | CP kernel interval | Mean SIMD pipeline busy fraction |
| --- | ---: | ---: |
| Prefill | 144.137 microseconds | 16.95% |
| Decode | 141.287 microseconds | 11.75% |

These are measurements of this initial workload, not expected production
performance. `profile-summary.csv` is derived from the retained full runner
metrics by `summarize_profiles.py`. Its busy fraction divides summed pipeline
busy time by all 256 SIMD units and the command-processor kernel interval. It
does not count useful FLOPs or measure lane efficiency.

The DRAM counters include setup/uploads, so no kernel HBM bandwidth estimate is
reported. Small contexts may be cache-resident. Low pipeline activity also
leaves possible instruction, scalar, LDS, barrier, and occupancy bottlenecks.
Consequently, **resource complementarity and bandwidth saturation have not been
established**. More representative contexts, shape sweeps, and kernel tuning
are required before scheduler gains would support the proposed research claim.

## Reproduce

From the repository root in PowerShell with Go and Python on PATH:

```powershell
./docs/research/pod-attention/reproduce.ps1
```

This runs the tests, vet, sample builds, a functional CLI verification, both
concurrency controls, both isolated timing profiles, summary generation, and
source fingerprinting. It uses the committed binary and does not rebuild
kernels or require WSL unless the OpenCL source changes. Generated executables
are ignored by Git. The default timing runner starts its existing local monitor.

Current sandboxed Go invocations can print a telemetry upload-token access
warning. The executed checks still return success; this is unrelated to GPU
simulation correctness. Simulator wall-clock duration is not a performance
metric for the modeled GPU.
