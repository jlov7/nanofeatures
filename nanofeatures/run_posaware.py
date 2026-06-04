"""M19: is the gradient's node-level advantage on distributed circuits just POSITION-RESOLUTION?

The sharpest objection to the topology boundary: across our task set "distributed across
positions" is confounded with sample size n and subject-token-count (every multi-position task is
also small-n), so "attribution wins because the circuit is distributed" might really be
"attribution wins on small-n tasks." This isolates the mechanism WITHIN a single task at fixed n.

On IOI the gradient (attribution) beats the position-blind cheap baseline diff_mag_max. We build a
gradient-FREE but POSITION-RESOLVED cheap score: weight each feature's per-position |delta act| by
the gradient-free per-position causal recovery rec_p (from positional_distributedness:
full-residual patching at each position, no SAE, no ranking, no gradient), then rank. If this
closes the attribution-vs-diff_mag_max gap on IOI, the gradient's only node-level advantage is
position-resolution, recoverable without it - and since the test is WITHIN IOI at fixed n, sample
size cannot be the explanation. This is the NODE analog of the edge cheap_fdpos control: it
unifies nodes and edges into one rule - any score that is causal and position-resolved ties the
gradient; the cheap default fails only because it is position-blind.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .circuit import (
    aligned_pairs,
    feature_rankings,
    ioi_pairs,
    positional_distributedness,
)
from .model import load_gemma, load_sae, pick_device
from .run_m5 import bootstrap_suff, ci, per_row_ld, per_row_patched_ld
from .run_suite import gate
from .tasks import TASK_SUITE

# one distributed circuit (IOI) + single-token controls
DEFAULT_TASKS = ["ioi", "capitals", "antonyms", "past_tense"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--ks", nargs="+", type=int, default=[16, 32, 64])
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="reports/m19_posaware.json")
    args = ap.parse_args()
    device = pick_device(args.device)
    model = load_gemma(device)
    sae, hook = load_sae(args.layer)

    def fmt(c):
        return f"{c[0]:+.0%} [{c[1]:+.0%},{c[2]:+.0%}]"

    methods = ["attribution", "diff_mag_max", "cheap_posaware"]
    results = []
    for task in args.tasks:
        d = (
            ioi_pairs(model)
            if task == "ioi"
            else aligned_pairs(model, *TASK_SUITE[task])
        )
        ok, info = gate(model, d, min_n=8, min_sep=1.0)
        print(
            f"\n[m19/{task}] n={info['n']} gated={ok} ({info.get('reason')})",
            flush=True,
        )
        if not ok:
            results.append({"task": task, "gated": False, "info": info})
            continue

        out = feature_rankings(model, sae, hook, d, exact=False)
        rankings, f_clean, f_corrupt = out[0], out[1], out[2]
        fdiff = (f_clean - f_corrupt).abs()  # [b,pos,d_sae]

        # gradient-free, ranking-free per-position causal recovery -> position weights
        pr, rec = positional_distributedness(model, d, args.layer)
        rec_full = torch.tensor([0.0] + list(rec), device=fdiff.device).clamp(min=0.0)
        # weight each feature's per-position change by how causally relevant that position is
        posaware = (fdiff * rec_full.view(1, -1, 1)).sum(dim=(0, 1))  # [d_sae]
        rankings = {
            **rankings,
            "cheap_posaware": posaware.argsort(descending=True, stable=True).tolist(),
        }

        clean_ld = per_row_ld(model, d, "clean")
        corrupt_ld = per_row_ld(model, d, "corrupt")
        n = d["n"]
        cache = {
            (m, k): per_row_patched_ld(
                model, sae, hook, d, rankings[m][:k], f_clean, f_corrupt
            )
            for m in methods
            for k in args.ks
        }
        g = torch.Generator().manual_seed(args.seed)
        idxs = [torch.randint(0, n, (n,), generator=g) for _ in range(args.boot)]

        task_cells = {}
        for k in args.ks:
            boot = {
                m: [
                    bootstrap_suff(clean_ld, corrupt_ld, cache[(m, k)], ix)
                    for ix in idxs
                ]
                for m in methods
            }
            per = {m: ci(v) for m, v in boot.items()}
            attr_gap = ci(
                [a - c for a, c in zip(boot["attribution"], boot["diff_mag_max"])]
            )
            pos_gap = ci(
                [p - c for p, c in zip(boot["cheap_posaware"], boot["diff_mag_max"])]
            )
            attr_minus_pos = ci(
                [a - p for a, p in zip(boot["attribution"], boot["cheap_posaware"])]
            )
            # what fraction of the gradient's advantage does gradient-free position-weighting recover?
            recovered_frac = (
                pos_gap[0] / attr_gap[0] if abs(attr_gap[0]) > 1e-6 else float("nan")
            )
            cell = {
                "suff": per,
                "attr_minus_diffmagmax": attr_gap,
                "posaware_minus_diffmagmax": pos_gap,
                "attr_minus_posaware": attr_minus_pos,
                "posaware_recovers_frac_of_gradient_gain": recovered_frac,
                "posaware_ties_attribution": not (
                    attr_minus_pos[1] > 0 or attr_minus_pos[2] < 0
                ),
            }
            task_cells[k] = cell
            print(
                f"[m19/{task}] L{args.layer} k={k:>3} | attr-cheap={fmt(attr_gap)} "
                f"posaware-cheap={fmt(pos_gap)} (recovers {recovered_frac:.0%} of gradient gain) "
                f"| attr-posaware={fmt(attr_minus_pos)} "
                f"{'TIE (posaware==gradient)' if cell['posaware_ties_attribution'] else 'gradient still ahead'}",
                flush=True,
            )
        results.append(
            {"task": task, "gated": True, "n": n, "pr": pr, "cells": task_cells}
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n[m19] wrote {args.out}")
    print(
        "[m19] VERDICT: if cheap_posaware (gradient-free, position-resolved) ties attribution on "
        "IOI while diff_mag_max (position-blind) does not, the gradient's node advantage is "
        "POSITION-RESOLUTION - not sample size, not task difficulty - mirroring the edge result."
    )


if __name__ == "__main__":
    main()
