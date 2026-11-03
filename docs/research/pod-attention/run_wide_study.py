"""Run wider GQA sensitivity and paired residency measurements."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess


def cases():
    for dim, context in ((32, 96), (64, 544)):
        for phase in ("prefill", "decode"):
            for lanes, columns in ((16, 1), (32, 1), (16, 8)):
                yield dim, context, phase, "serial", lanes, columns
        for mode in ("serial", "concurrent"):
            yield dim, context, "both", mode, 16, 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--original-gap", action="store_true",
                        help="trace the original compact dimension-eight case")
    args = parser.parse_args()

    def run(case):
        dim, context, phase, mode, lanes, columns = case
        kv_heads, chunk = ("8", "64") if args.original_gap else ("2", "32")
        command = [
            str(args.binary.resolve()), "-heads", "8", "-kv-heads", kv_heads, "-head-dim", str(dim),
            "-chunk", chunk, "-prefill-context", "96", "-batch", "16",
            "-decode-context", str(context), "-phase", phase, "-mode", mode,
            "-simd-lanes", str(lanes), "-column-cycles", str(columns),
        ]
        if args.original_gap:
            command.append("-compact-decode")
        result = subprocess.run(command, capture_output=True, text=True, timeout=900)
        if result.returncode:
            raise RuntimeError(f"{case}: {result.stdout}\n{result.stderr}")
        return json.loads(result.stdout.strip().splitlines()[-1])

    matrix = (
        [(8, 320, phase, mode, 16, 1) for phase, mode in
         (("prefill", "serial"), ("decode", "serial"), ("both", "concurrent"))]
        if args.original_gap else cases()
    )
    with args.output.open("w", encoding="utf-8") as output, ThreadPoolExecutor(args.jobs) as pool:
        for row in pool.map(run, matrix):
            output.write(json.dumps(row, sort_keys=True) + "\n")
            output.flush()
            print(row["head_dim"], row["phase"], row["mode"], row["simd_lanes"],
                  row["column_cycles"], round(row["makespan_us"], 3), flush=True)


if __name__ == "__main__":
    main()
