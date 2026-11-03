# Shared versus separate CUs

This experiment reuses the original FP32 compact-decode case on one 64-CU GCN3
timing model. Prefill processes a 64-token chunk in a 96-token context. Decode
processes 16 requests with 320 KV tokens each. Both phases use eight heads and
head dimension eight.

Ordinary concurrent dispatch lets both kernels use all 64 CUs. Fixed splits
restrict each kernel to disjoint ranges of CU resource-pool indices. The two
ranges share L2 and DRAM. For each split, the study runs each phase alone on
its assigned CU range, then runs both concurrently. These controls distinguish
the cost of a smaller CU allocation from the effect of running together. Batch
makespan and decode completion both start at the first kernel launch. Decode
completion covers the full decode batch.

The placement restriction is optional and applies to round-robin dispatch.
The study identifies the benchmark kernels by their workgroup sizes, 256 for
prefill and 64 for decode. This is specific to these kernels. The simulator
checks that partitioned kernels have no overlapping residency on any CU. A
CPU-reference smoke test also passed with the 24/40 split, and the dispatcher
test checks that the CU filter is cleared between launches.

## Comparison with POD-Attention

Kamath et al., *POD-Attention: Unlocking Full Prefill-Decode Overlap for Faster
LLM Inference*, ASPLOS 2025, arXiv:2410.18038v2, §5.1 and Figure 11, report
28% mean and 59% peak attention speedup over serial FlashAttention. Their
stream-based baseline can run separate kernels concurrently, but has no
guarantee of same-SM placement. The fused POD kernel schedules prefill and
decode blocks on the same SM, tunes decode tiles to reduce redundant compute,
and limits prefill splitting to control memory contention (§4). The paper's
offline serving results are a separate metric (§5.2, Figure 12).

The closest comparison here is the direction of the whole-attention change
from serial to concurrent dispatch. This experiment uses two separate OpenCL
kernels on a simulated GCN3 GPU. The paper evaluates tuned, fused attention on
NVIDIA A100 hardware with model-derived workloads. The paper's Figure 11 does
not provide a matched fixed-CU split baseline. The measurements here report
runtime reduction explicitly as `1 - concurrent / serial`.

## Reproduce

```powershell
go build -buildvcs=false -o attention-placement.exe ./samples/attention-study
python docs/research/pod-attention/run_placement_study.py ./attention-placement.exe docs/research/pod-attention/results/cu-placement.jsonl
python docs/research/pod-attention/run_placement_study.py ./attention-placement.exe docs/research/pod-attention/results/cu-placement-confirm.jsonl --confirm
./attention-placement.exe -phase both -mode concurrent -prefill-cus 24 -compact-decode -chunk 5 -prefill-context 17 -batch 4 -decode-context 19 -verify-outputs
```

## Results

| Prefill/decode placement | Prefill alone (µs) | Decode alone (µs) | Concurrent decode completion (µs) | Concurrent batch (µs) | CUs with overlapping residency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shared 64 | 50.935 | 83.469 | 104.451 | 104.451 | 64 |
| 16/48 | 154.713 | 86.759 | 89.015 | 149.120 | 0 |
| 32/32 | 92.563 | 86.951 | 86.395 | 91.936 | 0 |
| 48/16 | 62.035 | 114.619 | 117.851 | 117.851 | 0 |

The serial all-CU batch took 133.763 µs. Ordinary overlap reduced batch time
by 21.9%. The 32/32 split reduced batch time by 31.3% relative to serial and
12.0% relative to ordinary overlap. Its decode completion was 17.3% earlier
than ordinary overlap. Restricting isolated decode to 32 CUs costs about
3.5 µs, but mixed decode on those CUs completes close to its matched solo
time. The last-finishing decode workgroup's resident interval fell from
103.828 µs under shared dispatch to 85.814 µs with the split. The interval
before CU receipt changed from 0.617 to 0.577 µs.

Two repeat runs put shared dispatch at 103.479 and 104.265 µs. Both repeats
of 32/32 returned 91.936 µs for the batch and 86.395 µs for decode.

The 16/48 split improved decode completion while hurting batch time, and
48/16 worsened both. Both the shared and split mixed runs recorded 2,717,440
DRAM read request bytes. Equal byte counts do not establish equal cache
behavior or memory timing. Static placement also changes occupancy and how
many workgroups from each kernel compete on each CU. The results establish
an effective placement for this workload, with the exact contention mechanism
still unresolved.

The positive overlap result agrees with the paper's broad finding that
prefill and decode can benefit from concurrent execution. The advantage of
separating these two GCN3 kernels addresses a different question from whether
POD's tuned fused kernel benefits from same-SM execution on A100. The fixed
split is a candidate policy for this simulated workload. Generalizing it
requires a sweep across kernel shapes and prefill/decode balance.

## Workload sweep

The follow-up compares shared dispatch against 16/48, 24/40, 32/32, 40/24,
and 48/16 splits. Each workload uses the same kernels and inputs across its
six placements. The selected shapes cover three compact dimension-eight
workloads and three wider GQA workloads. These are a targeted FP32 sweep on
the same GCN3 model. Each point is one timing run, with the separate
original-case confirmation runs above.

| Shape | Head dimension | KV heads | Prefill chunk / context | Decode batch / context |
| --- | ---: | ---: | --- | --- |
| Compact prefill-heavy | 8 | 8 | 96 / 128 | 8 / 64 |
| Compact short mix | 8 | 8 | 64 / 96 | 16 / 64 |
| Compact decode-heavy | 8 | 8 | 16 / 64 | 24 / 192 |
| GQA32 prefill-heavy | 32 | 2 | 64 / 96 | 8 / 64 |
| GQA32 decode-heavy | 32 | 2 | 32 / 96 | 16 / 96 |
| GQA64 mix | 64 | 2 | 64 / 96 | 16 / 96 |

```powershell
python docs/research/pod-attention/run_placement_sweep.py ./attention-placement.exe docs/research/pod-attention/results/cu-placement-sweep.jsonl
```

A separate dimension-64 GQA smoke test passed CPU-reference verification with
the 40/24 split. The sweep checks every partitioned run for overlapping CU
residency.

| Shape | Shared batch (µs) | 32/32 batch (µs) | Lowest measured batch | Fastest decode (placement, µs) |
| --- | ---: | ---: | --- | --- |
| compact-prefill-heavy | 69.010 | 151.346 | Shared, 69.010 µs | 24/40, 20.417 |
| compact-short-mix | 54.045 | 92.316 | Shared, 54.045 µs | 24/40, 21.345 |
| compact-decode-heavy | 56.833 | 64.385 | 16/48, 55.047 µs | 16/48, 55.047 |
| gqa32-prefill-heavy | 67.025 | 111.719 | Shared, 67.025 µs | 16/48, 56.051 |
| gqa32-decode-heavy | 86.933 | 84.045 | 24/40, 83.358 µs | 16/48, 82.932 |
| gqa64-mix | 113.025 | 137.154 | 40/24, 111.829 µs | 24/40, 103.024 |

32/32 did not produce the lowest makespan in any of the six new shapes.
In the compact prefill-heavy case it more than doubled batch time, even
though decode completed earlier. Shared dispatch led both prefill-heavy
shapes and the compact short mix. In the decode-heavy shapes, 16/48
improved compact batch time by 3.1% and 24/40 improved dimension-32
GQA batch time by 4.1%. The 40/24 result for dimension-64 GQA was
only 1.1% ahead of shared dispatch in one run, so treat those as close.

Two compact decode-heavy repeats returned 56.841 and 56.839 µs shared
versus 54.997 and 55.093 µs at 16/48. These repeats corroborate that
specific split benefit. The remaining new shapes are single-run points.

This supports sharing as a default and choosing a fixed split only
where the workload and latency objective warrant it. The original
compact long-context case still favors 32/32. The sweep provides no
universal partition and does not measure the cost of choosing or
switching policies at runtime.

Raw measurements: `results/cu-placement-sweep.jsonl` and
`results/cu-placement-sweep-confirm.jsonl`.

```powershell
python docs/research/pod-attention/run_placement_sweep.py ./attention-placement.exe docs/research/pod-attention/results/cu-placement-sweep-confirm.jsonl --workload compact-decode-heavy --split 0 --split 16 --repeats 2 --jobs 2
```
