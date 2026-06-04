"""M17 precision validation: does bf16 distort the 9B numbers vs fp32?

The committed 9B result (reports/m17_gemma9b.json) runs in bf16 so the model + attribution
backward fit in 48GB. The honest question a reviewer asks: is the topology boundary
(attribution ties the strongest cheap baseline on single-token tasks, wins on the distributed
IOI circuit) a real effect, or an artifact of bf16 rounding? This compares the bf16 run to an
fp32 run of the IDENTICAL ladder (reports/m17_gemma9b_fp32.json, produced by
`run_scale --dtype fp32` on CPU) cell by cell.

The validation passes if (1) the per-cell BOUNDARY VERDICT (attr beats strongest cheap, i.e.
the sign of the attr-minus-strongest CI) agrees between bf16 and fp32 on every cell, and (2)
the point-estimate drift is within the bootstrap noise (CI half-width), i.e. bf16 and fp32 are
statistically indistinguishable. Then bf16 is not distorting the conclusion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _cells(report):
    """Flatten a run_scale report to {(task, layer, k): cell} over gated tasks."""
    out = {}
    for r in report:
        if not r.get("gated"):
            continue
        for layer, cells in r["layers"].items():
            for k, cell in cells.items():
                out[(r["task"], str(layer), str(k))] = cell
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", default="reports/m17_gemma9b.json")
    ap.add_argument("--fp32", default="reports/m17_gemma9b_fp32.json")
    ap.add_argument("--out", default="reports/m17_precision_compare.json")
    args = ap.parse_args()

    bf16 = _cells(json.loads(Path(args.bf16).read_text()))
    fp32 = _cells(json.loads(Path(args.fp32).read_text()))
    keys = sorted(set(bf16) & set(fp32))
    if not keys:
        raise SystemExit("no overlapping cells between the two reports")

    rows = []
    verdict_agree = 0
    within_noise = 0
    max_attr_drift = 0.0
    for key in keys:
        b, f = bf16[key], fp32[key]
        ba, fa = b["attr_minus_strongest"], f["attr_minus_strongest"]
        # attribution sufficiency point estimate drift
        attr_b = b["suff"]["attribution"][0]
        attr_f = f["suff"]["attribution"][0]
        attr_drift = abs(attr_b - attr_f)
        max_attr_drift = max(max_attr_drift, attr_drift)
        # boundary verdict: does attr beat the strongest cheap baseline (CI lower bound > 0)?
        agree = b["attr_beats_strongest"] == f["attr_beats_strongest"]
        verdict_agree += agree
        # is the bf16-vs-fp32 attribution gap inside the bootstrap CI half-width of either?
        b_halfwidth = (b["suff"]["attribution"][2] - b["suff"]["attribution"][1]) / 2
        noise_ok = attr_drift <= b_halfwidth
        within_noise += noise_ok
        rows.append(
            {
                "task": key[0],
                "layer": key[1],
                "k": key[2],
                "bf16_attr_suff": attr_b,
                "fp32_attr_suff": attr_f,
                "attr_suff_drift": attr_drift,
                "bf16_attr_minus_strongest": ba,
                "fp32_attr_minus_strongest": fa,
                "bf16_beats": b["attr_beats_strongest"],
                "fp32_beats": f["attr_beats_strongest"],
                "boundary_verdict_agrees": agree,
                "drift_within_bootstrap_noise": noise_ok,
                "bf16_strongest_cheap": b["strongest_cheap"],
                "fp32_strongest_cheap": f["strongest_cheap"],
            }
        )
        print(
            f"[{key[0]:>16} L{key[1]} k={key[2]:>3}] "
            f"attr bf16={attr_b:+.0%} fp32={attr_f:+.0%} (drift {attr_drift:.1%}) | "
            f"beats bf16={b['attr_beats_strongest']!s:>5} fp32={f['attr_beats_strongest']!s:>5} "
            f"-> {'AGREE' if agree else 'DISAGREE'}{'' if noise_ok else '  [drift>noise]'}",
            flush=True,
        )

    n = len(keys)
    summary = {
        "n_cells": n,
        "boundary_verdict_agree": verdict_agree,
        "drift_within_noise": within_noise,
        "max_attr_suff_drift": max_attr_drift,
        "passed": verdict_agree == n and within_noise == n,
        "cells": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(
        f"\n[precision] {verdict_agree}/{n} cells agree on the boundary verdict; "
        f"{within_noise}/{n} within bootstrap noise; max attribution-sufficiency drift "
        f"{max_attr_drift:.1%}."
    )
    print(
        "[precision] VERDICT: "
        + (
            "bf16 does NOT distort the 9B result — fp32 reproduces the topology boundary and "
            "every cell is within bootstrap noise. The committed bf16 numbers stand."
            if summary["passed"]
            else "see per-cell breakdown — at least one cell disagrees or drifts beyond noise."
        )
    )


if __name__ == "__main__":
    main()
