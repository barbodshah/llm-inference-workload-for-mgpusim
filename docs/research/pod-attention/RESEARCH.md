# Prefill/decode co-scheduling research

## Scope and hypotheses (step 1)

The [follow-up bottleneck study](BOTTLENECK-STUDY.md) records the measured
sensitivity, concurrency baselines, and decision on prototyping a scheduler.

Research base: this repository's MGPUSim 3.0.2 fork, initially one GCN3 GPU,
FP32 attention, independent requests, and non-preemptive workgroup admission.
The default detailed platform has 64 CUs, four SIMDs per CU, and eight command
processor dispatchers sharing one CU resource pool. This is not an A100 model.

The proposed future scheduler would coordinate prefill/decode admission, retain
each kernel's actual resource footprint, and optionally adjust its admission mix
using delayed resource-pressure feedback. It has **not been implemented**.

| Hypothesis | Required comparison | Falsification condition |
| --- | --- | --- |
| Coordinated placement adds value | Ordinary concurrent kernels versus coordinated admission, same footprints | Negligible improvement across representative mixes |
| Distinct resource footprints improve packing | Controlled common-footprint versus actual-footprint admission | No meaningful occupancy or completion-time improvement |
| Feedback improves robustness | Adaptive admission versus a separately tuned practical fixed policy | Gains disappear with controller overhead or fixed-policy tuning |

An offline best-fixed-policy sweep and an explicitly idealized scheduler are
diagnostic bounds, not implementable baselines. Hardware decision cost, finite
dispatch bandwidth, telemetry delay, starvation protection, and workload changes
must be included when the scheduler is implemented. Existing heterogeneous
allocation and concurrent-kernel support are baseline capabilities, not novelty.

Initial success criteria are numerical correctness, evidence of real same-CU
co-residency, and measured attention resource profiles. A scheduler claim needs
representative workloads and ablations; an attention microbenchmark cannot
establish TTFT, serving throughput, request latency, or A100 speedups.

## Existing concurrency audit (step 2)

The existing execution path is:

1. Driver command queues execute in order within each queue.
2. Each launch normally uploads its code, arguments, and dispatch packet.
3. The command processor selects an available dispatcher (eight by default).
4. Each dispatcher runs round-robin workgroup placement using the shared resource
   pool. Reservation already accounts for each code object's SGPR, VGPR, and LDS
   requirements, including allocation granularity.
5. Detailed CUs complete each workgroup to its originating dispatcher. The
   dispatcher frees reservations and returns the kernel completion to the driver.

No new placement policy or hardware scheduler was added.

### Experimental controls

`samples/attention-audit` runs two independent integer probe kernels twice on a
fresh detailed platform. Every output has an exact unsigned-integer CPU
reference. Repeated rounds check continued progress after resource reclamation.
Existing CU request-receipt/completion hooks record residency intervals using the
simulation clock. The tracer never changes scheduling.

The **same-queue** negative control must have zero overlap between distinct
kernel allocations. The **separate-queues** run must exhibit overlapping
workgroup residency on the same CU. This measures residency, not productive
compute/memory overlap. Four launches of four workgroups each give 16 completed
workgroups per run; the recorded separate-queue run touches eight mixed CUs
across its two rounds. It does not mean eight mixed CUs simultaneously.

`benchmarks/attention/launch.go` prepares uploads before exposing launch commands
to execution queues, using the public driver queue API. The original upload path
can skew short-kernel launch timing; the driver also conservatively marks all
context buffers dirty at launch. No driver cache policy was changed. These are
prepared-launch experiments, not measurements of an unmodified application's
launch overhead.

### Discovered functional-mode defect

The functional CU combined completed workgroup IDs from different dispatchers
into a single reply addressed to the last source. Concurrent functional attention
reproduced a dispatcher panic. `emu/computeunit.go` now sends each workgroup's
completion to its own request source, retrying on backpressure. Regression tests
cover distinct destinations and a failed-send retry. This changes functional
completion messaging, not detailed hardware timing or admission policy.

### Audit limits

The probe proves existing concurrency, ordering, output correctness, and repeated
completion. It does not exhaustively prove leak freedom or resource-pressure
behavior. Existing dispatcher/resource tests also pass. Dependency behavior is
only in-order within a queue; the benchmark supplies independent data to different
queues and does not introduce cross-queue dependency handling.

## Attention benchmark (step 3)

The standalone benchmark is independent of full Llama inference:

- One chunked causal prefill request and a batch of independent decode requests.
- Separate prefill/decode kernels and disjoint request caches.
- Query/KV head counts, head dimension, chunk size, and per-request context lengths
  are configurable; grouped-query attention is supported.
- FP32 tiled attention with an eight-token K/V tile and online softmax. Prefill
  shares each K/V tile across four queries; decode uses one query per workgroup.
- No global attention-score matrix and no private-memory spills in the validated
  binaries. FP32 native reciprocal, reciprocal square root, and exponential are
  tolerance-checked against a stable FP64 CPU reference.
- Serial and concurrent modes, plus isolated prefill/decode profiling.
- Functional and detailed execution use the embedded GCN3 binary, with no CPU
  execution fallback. Host verification runs separately from performance runs.

Input layout is `[request][query][query_head][dimension]` for queries and outputs,
and `[request][maximum_context][kv_head][dimension]` for K/V. A length buffer
defines each request's valid context. Ragged padding is poisoned in the input
generator. Prefill queries occupy the final `chunk` positions of the cache, and
the causal mask excludes later positions for each query. Context includes the
current query's K/V. The kernel is attention-only: it neither projects Q/K/V nor
appends them to a growing serving cache.

### Validation and initial profiling

See `VALIDATION.md` and the retained files under `results/` for executed checks,
toolchain information, hashes, residency traces, and isolated-phase profiles.

**The publication-readiness gate is still open.** These are portable initial
kernels, not a reproduction of POD or competitive FlashAttention kernels.
Four-query reuse, scalar softmax work, barriers, padding, and LDS access choices
can dominate their behavior. The initial profiles can establish different SIMD
activity but cannot by themselves establish HBM saturation or the target
compute-bound/memory-bound pairing. Do not implement the proposed scheduler on
the assumption that this pairing has already been validated.

Before scheduler work: expand the shape/context/batch sweep, check enough resident
work per SIMD, characterize cache residency and steady-state memory traffic, and
optimize or revise the benchmark if its bottlenecks are implementation artifacts.
Head dimensions above 64, FP16/BF16, tensor instructions, paged caches, serving
scheduling, and multi-GPU execution remain outside the implemented scope.
