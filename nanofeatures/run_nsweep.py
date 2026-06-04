"""M21: is the topology boundary driven by sample size n, or by task topology?

The deepest confound in the "distributed circuits need the gradient" story: across our task set,
every distributed (multi-position) task is also a small-n task (Pearson(n, gap) = -0.52). So "the
gradient wins on IOI" could be "the gradient wins when n is small." This decouples the two WITHIN
a task by subsampling: recompute the attribution-vs-diff_mag_max gap at a range of n, holding the
TASK (hence topology) fixed.

If topology drives the gap, it is flat in n: a single-token task (capitals) stays a tie at every
n down to 8, and a distributed task (IOI) stays a win at every n. If sample size drove it, the
single-token gap would grow as n shrinks toward the small-n regime where the multi-position tasks
live. No new forward passes: we cache per-row patched logit-diffs once and resample subsets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .circuit import aligned_pairs, feature_rankings, ioi_pairs
from .model import load_gemma, load_sae, pick_device
from .run_m5 import bootstrap_suff, ci, per_row_ld, per_row_patched_ld
from .run_suite import gate
from .tasks import TASK_SUITE

COMPARATOR = "diff_mag_max"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["ioi", "capitals", "past_tense"])
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--ns", nargs="+", type=int, default=[8, 12, 16, 20])
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="reports/m21_nsweep.json")
    args = ap.parse_args()
    device = pick_device(args.device)
    model = load_gemma(device)
    sae, hook = load_sae(args.layer)

    results = []
    for task in args.tasks:
        d = (
            ioi_pairs(model)
            if task == "ioi"
            else aligned_pairs(model, *TASK_SUITE[task])
        )
        ok, info = gate(model, d, min_n=8, min_sep=1.0)
        if not ok:
            print(f"[m21/{task}] gated=False ({info.get('reason')})", flush=True)
            continue
        N = d["n"]
        out = feature_rankings(model, sae, hook, d, exact=False)
        rankings, f_clean, f_corrupt = out[0], out[1], out[2]
        clean_ld = per_row_ld(model, d, "clean")
        corrupt_ld = per_row_ld(model, d, "corrupt")
        patched = {
            m: per_row_patched_ld(
                model, sae, hook, d, rankings[m][: args.k], f_clean, f_corrupt
            )
            for m in ("attribution", COMPARATOR)
        }
        # a fixed row order so nested n's are subsets (clean nesting)
        gperm = torch.Generator().manual_seed(args.seed + 777)
        order = torch.randperm(N, generator=gperm)

        sweep = []
        for n in args.ns:
            if n > N:
                continue
            rows = order[:n]
            cl, co = clean_ld[rows], corrupt_ld[rows]
            pa = {m: patched[m][rows] for m in patched}
            g = torch.Generator().manual_seed(args.seed)
            idxs = [torch.randint(0, n, (n,), generator=g) for _ in range(args.boot)]
            boot = {
                m: [bootstrap_suff(cl, co, pa[m], ix) for ix in idxs] for m in patched
            }
            gap = ci([a - c for a, c in zip(boot["attribution"], boot[COMPARATOR])])
            sweep.append({"n": n, "gap": gap})
            print(
                f"[m21/{task:>9} L{args.layer} k={args.k}] n={n:>2} "
                f"attr-cheap gap={gap[0]:+.1%} [{gap[1]:+.1%},{gap[2]:+.1%}]",
                flush=True,
            )
        results.append({"task": task, "N_full": N, "sweep": sweep})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n[m21] wrote {args.out}")
    print(
        "[m21] VERDICT: if each task's attr-vs-cheap gap is flat across n (single-token stays a "
        "tie at n=8, IOI stays a win at n=8), the boundary is set by TOPOLOGY, not sample size."
    )


if __name__ == "__main__":
    main()
