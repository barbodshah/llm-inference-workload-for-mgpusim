"""Run the fixed attention bottleneck matrix and save each completed result."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess


def cases():
    for name, chunk, prefill, batch, context in (
        ("small", 16, 64, 8, 64),
        ("balanced", 64, 96, 16, 64),
        ("large_kv", 64, 96, 16, 320),
    ):
        for phase in ("prefill", "decode"):
            # The two larger cases share the exact prefill configuration.
            if name == "large_kv" and phase == "prefill":
                continue
            for lanes, column in ((16, 1), (8, 1), (32, 1), (16, 8)):
                yield (name, phase, "serial", False, chunk, prefill, batch, context, lanes, column)
                if phase == "decode":
                    yield (name, phase, "serial", True, chunk, prefill, batch, context, lanes, column)
        if name != "small":
            for mode in ("serial", "concurrent"):
                for compact in (False, True):
                    yield (name, "both", mode, compact, chunk, prefill, batch, context, 16, 1)


def stress_cases():
    yield ("balanced", "prefill", "serial", False, 64, 96, 16, 320, 16, 64)
    for compact in (False, True):
        yield ("large_kv", "decode", "serial", compact, 64, 96, 16, 320, 16, 64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--column-stress", action="store_true",
                        help="run only the 64-cycle DRAM spacing controls")
    args = parser.parse_args()

    def run(case):
        name, phase, mode, compact, *numbers = case
        command = [str(args.binary.resolve()), "-phase", phase, "-mode", mode]
        if compact:
            command.append("-compact-decode")
        for key, value in zip(
            ("chunk", "prefill-context", "batch", "decode-context", "simd-lanes", "column-cycles"),
            numbers, strict=True,
        ):
            command.extend(("-" + key, str(value)))
        result = subprocess.run(command, capture_output=True, text=True, timeout=600)
        if result.returncode:
            raise RuntimeError(f"{case}: {result.stdout}\n{result.stderr}")
        row = json.loads(result.stdout.strip().splitlines()[-1])
        return {"case": name, **row}

    with args.output.open("w", encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for row in pool.map(run, stress_cases() if args.column_stress else cases()):
                output.write(json.dumps(row, sort_keys=True) + "\n")
                output.flush()
                print(row["case"], row["phase"], row["mode"], row["simd_lanes"],
                      row["column_cycles"], row["compact_decode"], round(row["makespan_us"], 3), flush=True)


if __name__ == "__main__":
    main()
