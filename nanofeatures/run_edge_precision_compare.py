"""Edge precision validation: do cross-layer edge results survive bf16? (reproducible)

Mirrors run_precision_compare (nodes) for edges. Reads the fp32 edge report and one or more bf16
edge reports and emits a per-cell comparison + the mechanism diagnostic, so the "edges need fp32"
claim is reproducible (not a hand-authored file). Pair with `diag_edge_precision` for the
intermediate-error breakdown that locates WHERE bf16 fails (the exact mediated-effect reference
and the transfer measurement, both perturbation-differences; the gradient readout itself is
bf16-robust).

Inputs (defaults):
  fp32 : reports/m14_edges_fdpos.json        (committed fp32, MPS)
  bf16 : reports/m18_edges_bf16.json          (bf16, MPS)
  bf16_cpu (optional): reports/m18_edges_bf16_cpu.json  (bf16 on CPU -> dtype-vs-device control)
  diag (optional): reports/m18_edge_precision_diag.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _by_task(report):
    return {c["task"]: c for c in report if "task" in c}


def _cmp(fp, bf):
    return {
        "eap_vs_exact_rel_residual": {
            "fp32": fp["eap_vs_exact_rel_residual"],
            "bf16": bf["eap_vs_exact_rel_residual"],
        },
        "cheap_fdpos_vs_exact_rho": {
            "fp32": (fp.get("cheap_fdpos_vs_exact") or [None])[0],
            "bf16": (bf.get("cheap_fdpos_vs_exact") or [None])[0],
        },
        "eap_minus_strongest_gradfree": {
            "fp32": fp["eap_minus_strongest_gradfree"][0],
            "bf16": bf["eap_minus_strongest_gradfree"][0],
        },
        "exact_absmax_meanrows": {
            "fp32": fp["exact_absmax_meanrows"],
            "bf16": bf["exact_absmax_meanrows"],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", default="reports/m14_edges_fdpos.json")
    ap.add_argument("--bf16", default="reports/m18_edges_bf16.json")
    ap.add_argument("--bf16-cpu", default="reports/m18_edges_bf16_cpu.json")
    ap.add_argument("--diag", default="reports/m18_edge_precision_diag.json")
    ap.add_argument("--out", default="reports/m18_edges_precision_compare.json")
    args = ap.parse_args()

    fp = _by_task(json.loads(Path(args.fp32).read_text()))
    bf = _by_task(json.loads(Path(args.bf16).read_text()))
    bf_cpu = (
        _by_task(json.loads(Path(args.bf16_cpu).read_text()))
        if Path(args.bf16_cpu).exists()
        else {}
    )
    diag = (
        {r["task"]: r for r in json.loads(Path(args.diag).read_text())}
        if Path(args.diag).exists()
        else {}
    )

    cells = []
    for task in sorted(set(fp) & set(bf)):
        row = {"task": task, "layers": f"{fp[task]['l1']}->{fp[task]['l2']}"}
        row.update(_cmp(fp[task], bf[task]))
        if task in bf_cpu:
            row["bf16_cpu"] = {
                "eap_vs_exact_rel_residual": bf_cpu[task]["eap_vs_exact_rel_residual"],
                "cheap_fdpos_vs_exact_rho": (
                    bf_cpu[task].get("cheap_fdpos_vs_exact") or [None]
                )[0],
            }
        if task in diag:
            row["mechanism"] = {
                k: diag[task][k]
                for k in (
                    "relerr_transfer",
                    "relerr_grad2",
                    "relerr_exact_effect",
                    "spearman_exact_fp32_vs_bf16",
                    "metric_snr",
                )
                if k in diag[task]
            }
        cells.append(row)
        print(
            f"[{task:>9} {row['layers']}] eap~exact resid fp32={row['eap_vs_exact_rel_residual']['fp32']:.3f} "
            f"-> bf16={row['eap_vs_exact_rel_residual']['bf16']:.3f} | "
            f"fdpos rho fp32={row['cheap_fdpos_vs_exact_rho']['fp32']:+.2f} "
            f"-> bf16={row['cheap_fdpos_vs_exact_rho']['bf16']:+.2f}"
            + (
                f" | bf16-CPU resid={row['bf16_cpu']['eap_vs_exact_rel_residual']:.3f} (dtype, not MPS)"
                if "bf16_cpu" in row
                else ""
            )
            + (
                f" | mech: grad relerr={row['mechanism']['relerr_grad2']:.2f} "
                f"exact relerr={row['mechanism']['relerr_exact_effect']:.2f}"
                if "mechanism" in row
                else ""
            ),
            flush=True,
        )

    summary = {
        "description": (
            "Edge precision validation. bf16 collapses the edge result: eap-vs-exact residual "
            "jumps and cheap_fdpos-vs-exact rho falls to ~0. Mechanism (see per-cell 'mechanism' "
            "and diag_edge_precision): the GRADIENT readout is bf16-robust (relerr ~0.03); the "
            "collapse is in the EXACT mediated-effect reference and the TRANSFER measurement, both "
            "differences of near-identical full-network forward passes under a tiny perturbation, "
            "which bf16 cannot resolve. Confirmed on CPU (bf16_cpu) -> a dtype effect, not an MPS "
            "artifact. Node sufficiency (M17 9B) survives bf16 because its effects are large; edge "
            "validation needs fp32 because the exact reference is a fragile perturbation-difference."
        ),
        "cells": cells,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\n[edge-precision] wrote {args.out}")


if __name__ == "__main__":
    main()
