# Standalone attention research benchmark

Start with [the research scope and concurrency audit](../../docs/research/pod-attention/RESEARCH.md).

From the repository root:

```powershell
go test ./benchmarks/attention -v
go run ./samples/attention -verify -mode serial
go run ./samples/attention -verify -mode concurrent `
  -heads 4 -kv-heads 2 -head-dim 7 -chunk 5 `
  -prefill-context 17 -decode-contexts 17,1,9
```

For detailed timing, use a separate run without verification:

```powershell
go run ./samples/attention -timing -report-all -mode concurrent `
  -metric-file-name attention-concurrent
```

`-phase prefill` and `-phase decode` isolate the kernels for profiling. The
default `-phase both` executes both. Context lengths include current query K/V;
prefill context must be at least the chunk size. Head dimensions are 1..64 and
positive KV heads must divide query heads. Decode batch size is the number of
entries in `-decode-contexts`. Requests can have different lengths.

```powershell
go run ./samples/attention-audit -audit-mode same-queue `
  -residency-csv same-queue.csv
go run ./samples/attention-audit -audit-mode separate-queues `
  -residency-csv separate-queues.csv
```

Both audit modes use detailed timing regardless of the standard runner's
`-timing` flag. Their default serial simulation engine makes residency tracing
deterministic. Do not interpret functional timestamps as performance estimates.

The committed code object is embedded. Kernel rebuilding requires the existing
WSL/Linux Clang 14 and ROCm device libraries:

```bash
bash benchmarks/attention/build-kernels.sh
```

Rebuild Go binaries after changing the code object. The loader validates gfx803,
wave64, entry offset, scratch usage, and host/kernel argument-size agreement.
Volatile local arrays avoid combined LDS instructions missing from this emulator.
No simulator instruction support was added for this benchmark.

The benchmark is an initial research workload, not a POD implementation or an
optimized vendor-library baseline. See the scope document for the remaining
resource-profile gate.
