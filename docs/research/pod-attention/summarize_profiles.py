"""Summarize existing runner metrics without calling them HBM utilization.

Usage: python docs/research/pod-attention/summarize_profiles.py
SIMD busy fraction uses all 256 SIMD units and the CP kernel interval. Pipeline
busy time is not FLOP utilization. DRAM counters include initialization/uploads.
"""
import csv
from pathlib import Path

root = Path(__file__).resolve().parent / "results"
out = []
for phase in ("prefill", "decode"):
    with (root / f"profile-{phase}.csv").open(newline="", encoding="utf-8-sig") as f:
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(f)]
    def values(what, match=lambda _: True):
        return [float(r["value"]) for r in rows if r["what"] == what and match(r["where"])]
    cp = values("kernel_time", lambda w: w.endswith("CommandProcessor"))[0]
    busy = values("busy_time", lambda w: ".SIMD" in w)
    if len(busy) != 256 or cp <= 0:
        raise ValueError("unexpected platform or missing kernel interval")
    out.append({
        "phase": phase,
        "cp_kernel_us": cp * 1e6,
        "mean_simd_pipeline_busy_fraction": sum(busy) / (len(busy) * cp),
        "wave_simd_instructions": sum(values("simd_inst_count")),
        "dram_reads_including_setup": sum(values("read_trans_count", lambda w: ".DRAM[" in w)),
        "dram_writes_including_setup": sum(values("write_trans_count", lambda w: ".DRAM[" in w)),
    })
with (root / "profile-summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=out[0].keys())
    writer.writeheader()
    writer.writerows(out)
for row in out:
    print(row)
print("Pipeline activity is not FLOP utilization; setup-inclusive DRAM counts do not establish bandwidth saturation.")
