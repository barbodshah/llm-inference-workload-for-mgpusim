# Wider GQA and workgroup residency

This follow-up tests FP32 grouped-query attention with eight query heads, two KV heads, and head dimensions of 32 and 64. Both cases use a 32-token prefill chunk in a 96-token context and 16 decode requests. Decode context lengths are 96 and 544 respectively, for cold KV footprints of 0.75 and 2.125 MiB on the model's 2 MiB L2. The two shapes change both head dimension and decode context, so comparisons between shapes cannot isolate either factor. The study also repeats the original dimension-eight compact decode case to locate the earlier mixed-execution gap.

Each isolated shape runs at the default 16 SIMD lanes, at 32 lanes with default DRAM timings, and at the default SIMD width with eight-cycle DRAM column spacing. This spacing is a service constraint and its change is not a calibrated bandwidth scaling factor. Serial and concurrent mixed runs use identical kernels and default hardware settings.

For each kernel, the profiler finds the workgroup that finishes last. Its launch-to-completion time splits into three intervals:

1. Command processor receipt to CU receipt of the workgroup. This includes grid dispatch and transfer as well as any resource admission delay.
2. CU receipt to workgroup completion. This includes all time resident, including stalls and contention.
3. Workgroup completion to kernel completion at the command processor.

The breakdown locates elapsed time. It cannot distinguish competition for SIMD issue, cache, memory, or other resources within the resident interval. Placement choices can still affect those resources.

## Reproduce

```powershell
go build -buildvcs=false -o attention-wide.exe ./samples/attention-study
python docs/research/pod-attention/run_wide_study.py ./attention-wide.exe docs/research/pod-attention/results/wide-gqa.jsonl
python docs/research/pod-attention/run_wide_study.py ./attention-wide.exe docs/research/pod-attention/results/original-gap-residency.jsonl --original-gap --jobs 2
./attention-wide.exe -phase both -mode concurrent -heads 8 -kv-heads 2 -head-dim 64 -chunk 5 -prefill-context 17 -batch 3 -decode-context 19 -verify-outputs
```

The final command checks both wider outputs against the CPU reference with ordinary memory copies. Performance runs use direct setup copies and a fresh timing platform.

## Where the original 21 microseconds went

| Compact dimension-eight decode | Alone | Concurrent | Change |
| --- | ---: | ---: | ---: |
| Kernel duration (µs) | 83.469 | 104.276 | +20.807 |
| Before the last-finishing workgroup reached its CU (µs) | 0.377 | 0.617 | +0.240 |
| Last-finishing workgroup's CU residence (µs) | 83.087 | 103.654 | +20.567 |
| Completion tail (µs) | 0.005 | 0.005 | 0.000 |

About 99% of the decode completion penalty accumulated after CU admission.
The average decode workgroup's residence also rose from 82.842 to 97.189 µs.
The last-finishing workgroup can differ across runs. The comparison decomposes
kernel completion time and uses the workgroup average as a cross-check.

## Sensitivity with wider heads

| Head dimension | Isolated phase | Default (µs) | 32 SIMD lanes (µs) | Eight-cycle DRAM spacing (µs) |
| ---: | --- | ---: | ---: | ---: |
| 32 | Prefill | 44.585 | 43.201 | 44.847 |
| 32 | Decode | 82.098 | 75.817 | 82.926 |
| 64 | Prefill | 60.970 | 59.139 | 61.086 |
| 64 | Decode | 534.335 | 494.907 | 533.971 |

Doubling SIMD width reduces decode runtime by roughly 7–8% in both shapes.
Increasing DRAM column spacing from one to eight cycles changes every
isolated runtime by about 1% or less. The wider workloads still provide
insufficient evidence for the compute-heavy prefill and bandwidth-heavy
decode pairing that motivated the scheduler.

## Concurrent execution

| Head dimension | Serial batch (µs) | Concurrent batch (µs) | Decode duration alone → concurrent (µs) | Last decode group before CU receipt (µs) | Last decode group resident (µs) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 127.202 | 86.933 | 82.098 → 86.932 | 0.352 → 0.377 | 81.682 → 86.501 |
| 64 | 598.439 | 544.707 | 534.335 → 544.706 | 0.382 → 0.517 | 533.936 → 544.180 |

Both concurrent runs co-locate the phases on all 64 CUs. The decode penalty accumulates primarily during CU residence in both wider cases. Together with the sensitivity measurements, this leaves the hardware admission prototype at no-go for the tested workloads.

The 16 wider runs and three original-case runs are saved as JSONL in the results directory. The profiler's Go tests, vet, and the dimension-64 CPU-reference smoke test passed.
