"""M20: equivalence testing (TOST) for the single-token "tie" claim.

The headline says attribution TIES the cheap baseline on single-token node circuits. As committed
that "tie" is just a non-significant gap (95% CI crosses 0) - absence of evidence, not evidence of
equivalence - and it is read off the cheap baseline with the highest in-sample sufficiency (a
winner's-curse selection that biases TOWARD declaring a tie). This replaces that with a proper
two-one-sided-test (TOST) against a PRE-REGISTERED comparator.

Comparator: diff_mag_max (the position-blind peak-change score; it is the selected strongest cheap
baseline in ~62/63 cells anyway, so pre-registering it costs nothing and removes the selection
bias). Margin: delta = 5pp recovered logit-difference, well below the IOI win floor of ~15pp, so
"within 5pp" means "practically equivalent relative to the effects we call real".

Per cell, a four-way verdict on g = suff(attribution) - suff(diff_mag_max):
  attr_win    : 95% CI of g has lower bound > 0          (attribution genuinely beats, signed)
  cheap_win   : 95% CI of g has upper bound < 0          (the cheap baseline genuinely beats)
  equivalent  : 90% CI of g lies entirely within +-delta (TOST: positive equivalence evidence)
  inconclusive: none of the above (n too small to call either way)

This converts the project's own disclosed limitation into a result and, honestly, will turn some
"ties" into "inconclusive" - which is the truthful state at small n.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .circuit import aligned_pairs, feature_rankings
from .model import load_gemma, load_sae, pick_device
from .run_m5 import bootstrap_suff, per_row_ld, per_row_patched_ld
from .run_suite import gate
from .tasks import TASK_SUITE

COMPARATOR = "diff_mag_max"


def _pct(vals, qs):
    t = torch.tensor(vals)
    return [torch.quantile(t, q).item() for q in qs]


def tost_verdict(lo95, hi95, lo90, hi90, delta):
    """Four-way verdict on the gap g = suff(attribution) - suff(comparator):
    attr_win (95% CI lower bound > 0), cheap_win (95% CI upper bound < 0),
    equivalent (90% CI inside +-delta, i.e. two one-sided tests pass), else inconclusive.
    Significance is checked before equivalence so a tiny-but-significant gap is a win, not a tie."""
    if lo95 > 0:
        return "attr_win"
    if hi95 < 0:
        return "cheap_win"
    if lo90 > -delta and hi90 < delta:
        return "equivalent"
    return "inconclusive"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(TASK_SUITE.keys()))
    ap.add_argument("--layers", nargs="+", type=int, default=[5, 7, 9])
    ap.add_argument("--ks", nargs="+", type=int, default=[16, 32, 64])
    ap.add_argument(
        "--boot", type=int, default=20000
    )  # tail decisions need more resamples
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="reports/m20_equivalence.json")
    args = ap.parse_args()
    device = pick_device(args.device)
    model = load_gemma(device)

    methods = ["attribution", COMPARATOR]
    cells, tally = (
        [],
        {"attr_win": 0, "cheap_win": 0, "equivalent": 0, "inconclusive": 0},
    )
    for task in args.tasks:
        d = aligned_pairs(model, *TASK_SUITE[task])
        ok, info = gate(model, d, min_n=8, min_sep=1.0)
        if not ok:
            print(f"[m20/{task}] gated=False ({info.get('reason')})", flush=True)
            continue
        n = d["n"]
        for layer in args.layers:
            sae, hook = load_sae(layer, width="16k")
            out = feature_rankings(model, sae, hook, d, exact=False)
            rankings, f_clean, f_corrupt = out[0], out[1], out[2]
            clean_ld = per_row_ld(model, d, "clean")
            corrupt_ld = per_row_ld(model, d, "corrupt")
            cache = {
                (m, k): per_row_patched_ld(
                    model, sae, hook, d, rankings[m][:k], f_clean, f_corrupt
                )
                for m in methods
                for k in args.ks
            }
            g = torch.Generator().manual_seed(args.seed)
            idxs = [torch.randint(0, n, (n,), generator=g) for _ in range(args.boot)]
            for k in args.ks:
                boot = {
                    m: [
                        bootstrap_suff(clean_ld, corrupt_ld, cache[(m, k)], ix)
                        for ix in idxs
                    ]
                    for m in methods
                }
                gap = [a - c for a, c in zip(boot["attribution"], boot[COMPARATOR])]
                lo95, hi95 = _pct(gap, [0.025, 0.975])
                lo90, hi90 = _pct(gap, [0.05, 0.95])
                med = _pct(gap, [0.5])[0]
                verdict = tost_verdict(lo95, hi95, lo90, hi90, args.delta)
                tally[verdict] += 1
                cells.append(
                    {
                        "task": task,
                        "layer": layer,
                        "k": k,
                        "n": n,
                        "gap_median": med,
                        "gap_ci95": [lo95, hi95],
                        "gap_ci90": [lo90, hi90],
                        "verdict": verdict,
                    }
                )
                print(
                    f"[m20/{task} L{layer} k={k:>3}] gap={med:+.1%} "
                    f"95%[{lo95:+.1%},{hi95:+.1%}] 90%[{lo90:+.1%},{hi90:+.1%}] -> {verdict}",
                    flush=True,
                )

    out = {
        "delta": args.delta,
        "comparator": COMPARATOR,
        "boot": args.boot,
        "tally": tally,
        "n_cells": len(cells),
        "cells": cells,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[m20] wrote {args.out}")
    print(
        f"[m20] of {len(cells)} single-token cells (TOST delta={args.delta:.0%}, pre-registered "
        f"vs {COMPARATOR}): {tally['attr_win']} attribution-win, {tally['equivalent']} EQUIVALENT, "
        f"{tally['cheap_win']} cheap-win, {tally['inconclusive']} inconclusive."
    )


if __name__ == "__main__":
    main()
