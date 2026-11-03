# Attention bottleneck study

The [wider GQA follow-up](WIDE-GQA-STUDY.md) traces the original gap and tests
additional head dimensions.

The decision for this single-GPU GCN3 study is **no-go on a hardware scheduler prototype**. The measured decode kernels remain sensitive to SIMD width and show little sensitivity to moderate DRAM column throttling. Existing dispatch already co-locates both phases on all 64 CUs in the mixed cases.

## Method

The study uses one 64-CU R9 Nano timing model with a 2 MiB L2, FP32, eight query and KV heads, and head dimension eight. A fresh platform serves each run. The small case uses a 16-token prefill chunk in a 64-token context and eight 64-token decode requests. The balanced case uses a 64-token chunk in a 96-token context and sixteen 64-token decode requests. The large-KV case keeps the same prefill and increases each decode context to 320 tokens. Decode KV footprints are 0.25, 0.50, and 2.50 MiB. The smaller footprints can fit in L2, but these runs begin cold.

The profiler uses the command processor's launch receipt and completion events to delimit each kernel window. Setup copies use the simulator's direct storage path. DRAM request bytes count traffic accepted by the memory controllers during the kernel windows, including instruction or translation traffic reaching those controllers. The counts describe requests and can omit dirty data written back after a window closes. Decode completion is the decode kernel's completion relative to the first launch, and covers the entire decode batch.

The experiment varies the modeled SIMD lanes per cycle from 16 to 8 or 32, or changes DRAM read/write column command spacing from 1 to 8 cycles while keeping SIMD width fixed. An additional 64-cycle control tests a much stronger throttle. The column setting changes one constraint on memory service rate. It does not scale end-to-end DRAM bandwidth by a known factor. The CPI counters are simulator activity categories. The raw logs also record barrier instruction counts and time-weighted resident wavefront counts. Barrier instruction counts do not measure time blocked at barriers.

Both decode kernels implement the same attention math. The compact variant specializes for dimension eight and reduces LDS per workgroup from 4,392 to 584 bytes. It also changes the instruction stream and register allocation. The experiment measures the combined effect of those changes.

## Reproduce

Run from the repository root with Go and Python 3.10 or later. The compiled kernel is embedded in the Go executable.

```powershell
go build -buildvcs=false -o attention-study.exe ./samples/attention-study
python docs/research/pod-attention/run_study.py ./attention-study.exe docs/research/pod-attention/results/bottleneck-study.jsonl
python docs/research/pod-attention/run_study.py ./attention-study.exe docs/research/pod-attention/results/column-stress.jsonl --column-stress
./attention-study.exe -phase both -mode concurrent -chunk 5 -prefill-context 17 -batch 3 -decode-context 19 -compact-decode -verify-outputs
```

The last command uses ordinary transfers and verifies both outputs against the CPU reference. The Go tests also cover compact decode with ragged contexts and grouped query heads. The study runs used Go 1.27.1 on Windows and gfx803 code compiled with Clang 14 in WSL.

## Results

All runtimes are simulated microseconds. Each row changes one hardware parameter at a time.

| Case | Phase and kernel | Default | 8 SIMD lanes | 32 SIMD lanes | 8-cycle DRAM spacing |
| --- | --- | ---: | ---: | ---: | ---: |
| small | prefill | 25.459 | 29.540 | 24.634 | 25.352 |
| small | decode | 46.161 | 52.949 | 42.613 | 46.165 |
| small | decode compact | 20.577 | 24.107 | 19.063 | 20.567 |
| balanced | prefill | 50.935 | 68.460 | 42.461 | 50.024 |
| balanced | decode | 48.337 | 54.773 | 45.110 | 48.738 |
| balanced | decode compact | 20.559 | 24.169 | 19.153 | 20.595 |
| large kv | decode | 219.443 | 249.803 | 204.062 | 218.694 |
| large kv | decode compact | 83.469 | 99.571 | 75.971 | 83.535 |

Balanced prefill takes 34.4% longer with half the SIMD width and runs 16.6% faster with double the width. Large-KV compact decode takes 19.3% longer with half the width and runs 9.0% faster with double the width. The 8-cycle column spacing changes compact large-KV decode by under 0.1%. The 64-cycle stress control raises its runtime to 99.755 us, up 19.5%. This shows the compact kernel can respond to a sufficiently strong throttle. At default settings, the tested decode is not demonstrably limited by DRAM bandwidth.

Large-KV decode accepts about 2.51 MiB of DRAM reads in both variants. Compact tiling cuts its time from 219.443 to 83.469 us with comparable traffic. The benchmark also records only about two resident wavefronts per CU in isolated large-KV decode, so a shortage of in-flight work remains a plausible factor.

## Mixed execution and idealized overlap

The idealized experiment aligns the two isolated kernel timelines at time zero and assumes no interference. Its makespan is the longer isolated duration. This is an overlap reference, since a real placement may change kernel durations, memory behavior, and occupancy.

| Case | Decode | Serial makespan | Concurrent makespan | Concurrent decode completion | Ideal overlap | Remaining reduction to reference |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| balanced | baseline | 99.069 | 61.979 | 61.979 | 50.935 | 17.8% |
| balanced | compact | 70.821 | 54.045 | 31.867 | 50.935 | 5.8% |
| large kv | baseline | 271.087 | 229.039 | 229.039 | 219.443 | 4.2% |
| large kv | compact | 133.763 | 104.276 | 104.276 | 83.469 | 20.0% |

The compact balanced case is close to the overlap reference. The compact large-KV case still has material headroom in that model despite co-residency on all CUs. These results do not attribute the gap to admission or show that a different placement can recover it. A scheduler study would need a workload with a better established compute/memory split and evidence that the residual delay follows admission decisions.

Raw data are in `results/bottleneck-study.jsonl` and `results/column-stress.jsonl`.
The embedded kernel SHA-256 is `927ec93249f3f081c2e7a54895630ecd16c5f47ed692dc1fde252f888ba462b8`.
