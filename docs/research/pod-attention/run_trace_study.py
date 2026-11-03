"""Replay seeded request mixes using measured MGPUSim attention rounds."""
import argparse
import json
import math
from pathlib import Path
import random
from statistics import mean
import subprocess

SPLITS = (16, 24, 32, 40, 48)


def generate(seed, count, long_fraction, short, long, outputs):
    rng = random.Random(seed)
    requests = []
    for _ in range(count):
        bounds = long if rng.random() < long_fraction else short
        requests.append({"prompt": rng.randint(*bounds), "output": rng.randint(*outputs)})
    return requests


def rounds(requests, chunk_size):
    # One prompt at a time, with all eligible decode requests taking one step.
    active = []
    for index, request in enumerate(requests):
        for prefix in range(0, request["prompt"], chunk_size):
            chunk = min(chunk_size, request["prompt"] - prefix)
            yield index, chunk, prefix + chunk, active
            active = [(i, context + 1, left - 1) for i, context, left in active if left > 1]
        active.append((index, request["prompt"] + 1, request["output"]))
    while active:
        yield None, 0, 0, active
        active = [(i, context + 1, left - 1) for i, context, left in active if left > 1]


def choose_split(policy, chunk, prefill_context, lengths, utilization):
    if not chunk or not lengths or policy == "shared":
        return 0
    if policy.startswith("fixed-"):
        return int(policy.removeprefix("fixed-"))
    prefill_ctas = 8 * math.ceil(chunk / 4)
    decode_ctas = 8 * len(lengths)
    # Favor sharing if the previous round left capacity idle or current work is sparse.
    if utilization < 0.5 or min(prefill_ctas, decode_ctas) < 16:
        return 0
    prefill_work, decode_work = prefill_ctas, decode_ctas
    if policy == "adaptive-context":
        prefill_work *= prefill_context - (chunk - 1) / 2
        decode_work *= mean(lengths)
    target = 64 * prefill_work / (prefill_work + decode_work)
    return min(SPLITS, key=lambda split: abs(split - target))


def measure(binary, chunk, context, lengths, split):
    phase = "both" if chunk and lengths else "prefill" if chunk else "decode"
    command = [
        binary, "-phase", phase, "-mode", "concurrent", "-compact-decode",
        "-chunk", str(chunk or 1), "-prefill-context", str(context or 1),
        "-decode-contexts", ",".join(map(str, lengths or [1])),
        "-prefill-cus", str(split),
    ]
    # Retry the known launch-staging deadlock in a fresh simulator process.
    for attempt in range(3):
        run = subprocess.run(command, capture_output=True, text=True, timeout=900)
        if not run.returncode or "all goroutines are asleep - deadlock!" not in run.stderr:
            break
        if "attention.prepareLaunch" not in run.stderr:
            break
        print(f"Launch staging deadlocked (attempt {attempt + 1}/3): {command}", flush=True)
    if run.returncode:
        raise RuntimeError(f"{command}: {run.stdout}\n{run.stderr}")
    row = json.loads(run.stdout.strip().splitlines()[-1])
    if split and row["mixed_cus"]:
        raise RuntimeError("fixed split produced overlapping CU residency")
    return row


def replay(binary, requests, chunk_size, policy, cache, emit):
    elapsed = 0.0
    utilization = 0.0
    prefill_start = {}
    first_token_latencies = []
    prior_token = {}
    gaps = []
    for index, (prefill, chunk, context, active) in enumerate(rounds(requests, chunk_size)):
        lengths = tuple(length for _, length, _ in active)
        split = choose_split(policy, chunk, context, lengths, utilization)
        key = (chunk, context, lengths, split)
        if key not in cache:
            cache[key] = measure(binary, *key)
        row = cache[key]
        if prefill is not None:
            prefill_start.setdefault(prefill, elapsed)
        if active:
            completed = elapsed + row["decode_completion_us"]
            for request, _, _ in active:
                if request in prior_token:
                    gaps.append(completed - prior_token[request])
                else:
                    first_token_latencies.append(completed - prefill_start[request])
                prior_token[request] = completed
        emit({"kind": "round", "policy": policy, "round": index,
              "prefill_request": prefill, "decode_requests": [i for i, _, _ in active],
              "prior_active_cu_fraction": utilization, **row})
        utilization = row["active_cu_fraction"]
        elapsed += row["makespan_us"]
    tokens = sum(request["output"] for request in requests)
    return {"kind": "summary", "policy": policy, "attention_us": elapsed,
            "decode_tokens": tokens, "decode_tokens_per_attention_second": tokens * 1e6 / elapsed,
            "p95_first_decode_us": percentile95(first_token_latencies),
            "p95_inter_decode_us": percentile95(gaps)}


def percentile95(values):
    return sorted(values)[math.ceil(0.95 * len(values)) - 1] if values else None

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--long-fraction", type=float, default=0.25)
    parser.add_argument("--short", type=int, nargs=2, default=(8, 32))
    parser.add_argument("--long", type=int, nargs=2, default=(64, 128))
    parser.add_argument("--outputs", type=int, nargs=2, default=(4, 12))
    args = parser.parse_args()
    if not 1 <= args.requests <= 128 or args.chunk < 1 or not 0 <= args.long_fraction <= 1:
        parser.error("require 1..128 requests, positive chunk and long fraction in [0, 1]")
    if any(not 1 <= low <= high for low, high in (args.short, args.long, args.outputs)):
        parser.error("require positive, ordered length bounds")
    binary = str(args.binary.resolve())
    policies = ["shared", *(f"fixed-{split}" for split in SPLITS),
                "adaptive-ctas", "adaptive-context"]
    cache = {}
    with args.output.open("x", encoding="utf-8") as output:
        for seed in args.seed:
            def emit(row):
                output.write(json.dumps({"seed": seed, **row}) + "\n")
                output.flush()

            requests = generate(seed, args.requests, args.long_fraction,
                                args.short, args.long, args.outputs)
            emit({"kind": "trace", "requests": requests, "chunk_size": args.chunk,
                  "distribution": {"long_fraction": args.long_fraction, "short": args.short,
                                   "long": args.long, "outputs": args.outputs}})
            for policy in policies:
                summary = replay(binary, requests, args.chunk, policy, cache, emit)
                emit(summary)
                print(json.dumps({"seed": seed, **summary}), flush=True)


if __name__ == "__main__":
    main()
