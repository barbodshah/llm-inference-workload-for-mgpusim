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

### Seeded mixed-context placement replay

From the repository root, build the timing profiler and run the trace driver
with Python 3.9 or newer. Choose a new output filename for each run.

```powershell
go build -o attention-study.exe ./samples/attention-study
python docs/research/pod-attention/run_trace_study.py ./attention-study.exe trace.jsonl --seed 42 --seed 101
```

The driver draws prompt lengths from a mixture of two discrete uniform ranges.
Use `--short`, `--long`, and `--long-fraction` to set the ranges and mixture,
and `--outputs` to set the output length range. Prompts enter in trace order as
the previous prompt finishes prefill. Each round runs at most one prefill chunk
and one decode step for every eligible request. The decode kernel uses each
request's own KV length, which grows by one per step.

The JSONL records the generated requests, the measured rounds, and per-seed
summaries. It compares shared dispatch, five fixed splits for mixed rounds,
and two adaptive policies. Both adaptive policies use CTA counts and the
previous round's fraction of CU time with any resident CTA. The context-aware
policy also weights CTA counts by the mean visible KV length for each phase.
Both share CUs if the previous round had less than 50% CU occupancy or either
phase has fewer than 16 CTAs. Otherwise they choose the closest tested split
to the estimated prefill share of work.
Compare it with the CTA-only policy to isolate the value of that context
estimate. Compare both with shared dispatch and the best fixed split across
evaluation seeds to see whether adaptation helps.

Each round starts a fresh timing simulation and ends before the next round.
The reported throughput, first decode latency, and inter-decode latency are
attention-only measurements. They exclude arrival queueing, persistent cache
state across rounds, and full-model execution. Decisions apply between rounds.
