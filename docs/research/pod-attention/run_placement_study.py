"""Compare shared dispatch with fixed CU splits and matched solo controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--confirm", action="store_true",
                        help="repeat the shared and 32/32 mixed cases twice")
    args = parser.parse_args()
    cases = [(0, "both", "serial")]
    for split in (0, 16, 32, 48):
        cases += [(split, "prefill", "serial"), (split, "decode", "serial"),
                  (split, "both", "concurrent")]
    if args.confirm:
        cases = [(split, "both", "concurrent") for split in (0, 32)] * 2

    def run(case):
        split, phase, mode = case
        command = [str(args.binary.resolve()), "-compact-decode", "-chunk", "64",
                   "-prefill-context", "96", "-batch", "16", "-decode-context", "320",
                   "-prefill-cus", str(split), "-phase", phase, "-mode", mode]
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=900)
        return json.loads(result.stdout.strip().splitlines()[-1])

    with args.output.open("w", encoding="utf-8") as output, ThreadPoolExecutor(args.jobs) as pool:
        for row in pool.map(run, cases):
            output.write(json.dumps(row, sort_keys=True) + "\n")
            output.flush()
            print(row["prefill_cus"], row["phase"], row["mode"],
                  round(row["makespan_us"], 3), round(row.get("decode_completion_us", 0), 3),
                  flush=True)


if __name__ == "__main__":
    main()
