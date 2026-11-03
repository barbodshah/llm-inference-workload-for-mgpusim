"""Sweep CU placement across prefill and decode shapes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

# Name, head dimension, KV heads, chunk, prefill context, decode batch/context.
WORKLOADS = (
    ("compact-prefill-heavy", 8, 8, 96, 128, 8, 64),
    ("compact-short-mix", 8, 8, 64, 96, 16, 64),
    ("compact-decode-heavy", 8, 8, 16, 64, 24, 192),
    ("gqa32-prefill-heavy", 32, 2, 64, 96, 8, 64),
    ("gqa32-decode-heavy", 32, 2, 32, 96, 16, 96),
    ("gqa64-mix", 64, 2, 64, 96, 16, 96),
)
SPLITS = (0, 16, 24, 32, 40, 48)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--workload", choices=[workload[0] for workload in WORKLOADS])
    parser.add_argument("--split", type=int, choices=SPLITS, action="append")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.jobs < 1 or args.repeats < 1:
        parser.error("jobs and repeats must be positive")

    def run(case):
        workload, split = case
        name, dim, kv, chunk, prefill, batch, context = workload
        command = [str(args.binary.resolve()), "-phase", "both", "-mode", "concurrent"]
        for key, value in (
            ("head-dim", dim), ("heads", 8), ("kv-heads", kv), ("chunk", chunk),
            ("prefill-context", prefill), ("batch", batch), ("decode-context", context),
            ("prefill-cus", split),
        ):
            command.extend(("-" + key, str(value)))
        if dim == 8:
            command.append("-compact-decode")
        result = subprocess.run(command, capture_output=True, text=True, timeout=900)
        if result.returncode:
            raise RuntimeError(f"{case}: {result.stdout}\n{result.stderr}")
        row = json.loads(result.stdout.strip().splitlines()[-1])
        if split and row["mixed_cus"]:
            raise RuntimeError(f"{case}: overlapping CU residency in a fixed split")
        return {"workload": name, **row}

    cases = [
        (workload, split)
        for workload in WORKLOADS
        if args.workload is None or workload[0] == args.workload
        for split in (args.split or SPLITS)
    ] * args.repeats
    with args.output.open("w", encoding="utf-8") as output, ThreadPoolExecutor(args.jobs) as pool:
        for row in pool.map(run, cases):
            output.write(json.dumps(row, sort_keys=True) + "\n")
            output.flush()
            print(row["workload"], row["prefill_cus"], round(row["makespan_us"], 3),
                  round(row["decode_completion_us"], 3), flush=True)


if __name__ == "__main__":
    main()
