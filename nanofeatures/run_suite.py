"""M7 (breadth): is the causal-vs-cheap-baseline finding TASK-GENERAL, or a capitals
anecdote? We run the full ladder on every task in tasks.TASK_SUITE that (a) survives the
single-token alignment filter with enough pairs and (b) passes a behavior gate (the model
actually does the task: clean logit-diff positive and clean-vs-corrupt separated).

For each surviving task we evaluate at a FIXED set of layers (no per-task layer
selection: full-residual recovery saturates at ~100% everywhere for these single-token
contrastive tasks, so it cannot discriminate layers, and any ranking-dependent selector
would leak into the method comparison). Reporting every task x layer x k cell is strictly
more transparent than picking one layer -- it answers 'does it hold across tasks AND
layers' at once, the way M3 did for capitals alone.

Per cell we paired-bootstrap the sufficiency of attribution vs every cheap baseline, and
(if --exact) exact causal vs attribution, reporting gaps with 95% CIs. The headline is
whatever the AGGREGATE says -- we do not pre-commit it. Two questions decide it: does
attribution beat the STRONGEST cheap baseline (CI>0) on most cells, and is exact causal
worth its cost over attribution (gap CI excludes 0)?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .circuit import aligned_pairs, feature_rankings
from .model import load_gemma, load_neuron_basis, load_sae, pick_device
from .run_m5 import bootstrap_suff, ci, per_row_ld, per_row_patched_ld
from .tasks import TASK_SUITE

# cheap, gradient-free competitors to attribution (same set as M6)
BASELINES = [
    "diff_mag",
    "diff_mag_subjpos",
    "diff_mag_lastpos",
    "diff_mag_max",
    "diff_mag_max_wdec",
    "magnitude",
]


def gate(model, d, min_n, min_sep):
    """Behavior gate: enough aligned pairs, model prefers the correct answer on the
    clean run, and clean is separated from corrupt (there is a behavior to explain)."""
    n = d["n"]
    if n < min_n:
        return False, {"reason": f"only {n} single-token pairs (<{min_n})", "n": n}
    clean_ld = per_row_ld(model, d, "clean")
    corrupt_ld = per_row_ld(model, d, "corrupt")
    mean_clean = clean_ld.mean().item()
    sep = (clean_ld - corrupt_ld).mean().item()
    ok = mean_clean > 0.0 and sep > min_sep
    info = {
        "n": n,
        "mean_clean_logit_diff": mean_clean,
        "clean_minus_corrupt_sep": sep,
        "reason": None
        if ok
        else f"behavior gate failed (mean_clean={mean_clean:+.2f}, sep={sep:+.2f})",
    }
    return ok, info


def eval_cell(model, sae, hook, d, ks, B, seed, exact, clean_ld, corrupt_ld):
    """Bootstrap the sufficiency ladder at one (task, layer). Returns {k: cell}."""
    out = feature_rankings(model, sae, hook, d, exact=exact)
    rankings, f_clean, f_corrupt, active = out[0], out[1], out[2], out[3]
    n = d["n"]

    # a seeded RANDOM ranking over active features = the ladder floor (self-contained, so
    # the "magnitude ~= random ~= 0" claim does not have to be imported from earlier
    # milestones). Excluded from BASELINES, so it never competes to be "strongest cheap".
    rg = torch.Generator().manual_seed(seed + 9973)
    rand_rank = active[torch.randperm(len(active), generator=rg)].tolist()
    rankings = {**rankings, "random": rand_rank}

    methods = (["causal"] if exact else []) + ["attribution", *BASELINES, "random"]
    cache = {
        (rname, k): per_row_patched_ld(
            model, sae, hook, d, rankings[rname][:k], f_clean, f_corrupt
        )
        for rname in methods
        for k in ks
    }

    g = torch.Generator().manual_seed(seed)
    idxs = [torch.randint(0, n, (n,), generator=g) for _ in range(B)]

    cells = {}
    for k in ks:
        boot = {
            rname: [
                bootstrap_suff(clean_ld, corrupt_ld, cache[(rname, k)], ix)
                for ix in idxs
            ]
            for rname in methods
        }
        per = {rname: ci(vals) for rname, vals in boot.items()}
        gaps = {
            b: ci([a - bb for a, bb in zip(boot["attribution"], boot[b])])
            for b in BASELINES
        }
        strongest = max(BASELINES, key=lambda b: per[b][0])
        attr_vs_strong = gaps[strongest]
        cell = {
            "suff": per,
            "gaps_attr_minus": gaps,
            "strongest_cheap": strongest,
            "attr_minus_strongest": attr_vs_strong,
            "attr_beats_strongest": attr_vs_strong[1] > 0,
        }
        if exact:
            cell["causal_minus_attr"] = ci(
                [c - a for c, a in zip(boot["causal"], boot["attribution"])]
            )
            cell["causal_beats_attr"] = cell["causal_minus_attr"][1] > 0
        cells[k] = cell
    return cells


def run_task(
    model,
    device,
    name,
    pairs,
    template,
    layers,
    ks,
    B,
    seed,
    exact,
    neuron=False,
    width="16k",
):
    d = aligned_pairs(model, pairs=pairs, template=template)
    ok, info = gate(model, d, min_n=8, min_sep=1.0)
    print(
        f"\n[suite/{name}] candidates={len(pairs)} aligned_n={info['n']} "
        f"mean_clean={info.get('mean_clean_logit_diff', float('nan')):+.2f} "
        f"sep={info.get('clean_minus_corrupt_sep', float('nan')):+.2f}"
    )
    if not ok:
        print(f"[suite/{name}] SKIP: {info['reason']}")
        return {"task": name, "gated": False, "info": info}

    clean_ld = per_row_ld(model, d, "clean")
    corrupt_ld = per_row_ld(model, d, "corrupt")

    def fmt(c):
        return f"{c[0]:+.0%} [{c[1]:+.0%},{c[2]:+.0%}]"

    layers_out = {}
    for layer in layers:
        sae, hook = (
            load_neuron_basis(model, layer, device)
            if neuron
            else load_sae(layer, device=device, width=width)
        )
        cells = eval_cell(model, sae, hook, d, ks, B, seed, exact, clean_ld, corrupt_ld)
        layers_out[layer] = cells
        for k in ks:
            cell = cells[k]
            per = cell["suff"]
            line = (
                f"[suite/{name}] L{layer:>2} k={k:>3}  attr={fmt(per['attribution'])}"
            )
            if exact:
                line += f"  causal={fmt(per['causal'])}"
            print(line)
            avs = cell["attr_minus_strongest"]
            sig = "SIG" if avs[1] > 0 else "n.s."
            print(
                f"            strongest cheap = {cell['strongest_cheap']} "
                f"({fmt(per[cell['strongest_cheap']])}); attr - it = {fmt(avs)} -> {sig}"
            )
            if exact:
                cg = cell["causal_minus_attr"]
                csig = (
                    "exact WORTH it"
                    if cg[1] > 0
                    else ("attr WORSE" if cg[2] < 0 else "exact ~= attr")
                )
                print(f"            causal - attr = {fmt(cg)} -> {csig}")
    return {
        "task": name,
        "gated": True,
        "n": d["n"],
        "B": B,
        "info": info,
        "layers": layers_out,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="subset of TASK_SUITE keys (default all)",
    )
    ap.add_argument("--layers", nargs="+", type=int, default=[5, 7, 9])
    ap.add_argument("--ks", nargs="+", type=int, default=[16, 32, 64])
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--exact",
        action="store_true",
        help="also compute exact causal effects (expensive) for causal-vs-attr",
    )
    ap.add_argument(
        "--neuron",
        action="store_true",
        help="rank/patch RAW residual neurons instead of SAE features (basis control)",
    )
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--width",
        default="16k",
        help="Gemma Scope SAE width (16k default, 65k for the sweep)",
    )
    ap.add_argument("--out", default="reports/m7_suite.json")
    args = ap.parse_args()
    device = pick_device(args.device)
    model = load_gemma(device)

    names = args.tasks or list(TASK_SUITE.keys())
    basis = "NEURON" if args.neuron else f"SAE/{args.width}"
    print(
        f"[suite] tasks={names} layers={args.layers} exact={args.exact} basis={basis}"
    )

    results = []
    for name in names:
        pairs, template = TASK_SUITE[name]
        results.append(
            run_task(
                model,
                device,
                name,
                pairs,
                template,
                args.layers,
                args.ks,
                args.boot,
                args.seed,
                args.exact,
                args.neuron,
                args.width,
            )
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[suite] wrote {out}")

    # --- aggregate verdict over GATED task x layer x k cells ---
    gated = [r for r in results if r.get("gated")]
    print(
        f"\n[suite] gated {len(gated)}/{len(results)} tasks: {[r['task'] for r in gated]}"
    )
    cells = wins = 0
    causal_cells = causal_wins = 0
    for r in gated:
        for layer_cells in r["layers"].values():
            for cell in layer_cells.values():
                cells += 1
                if cell["attr_beats_strongest"]:
                    wins += 1
                if "causal_beats_attr" in cell:
                    causal_cells += 1
                    if cell["causal_beats_attr"]:
                        causal_wins += 1
    print(
        f"[suite] attribution beats the STRONGEST cheap baseline (CI>0) at "
        f"{wins}/{cells} task x layer x k cells"
    )
    if causal_cells:
        print(
            f"[suite] exact causal beats attribution (CI>0) at "
            f"{causal_wins}/{causal_cells} cells -> "
            + (
                "exact buys faithfulness"
                if causal_wins > causal_cells / 2
                else "exact ~= attribution (cheap AtP suffices)"
            )
        )
    if cells:
        frac = wins / cells
        print(
            "[suite] VERDICT: "
            + (
                "attribution's causal advantage is TASK-GENERAL."
                if frac >= 0.8
                else (
                    "MIXED -- attribution wins some cells, ties the strong cheap baseline on "
                    "others; the honest headline is the per-task breakdown, not a blanket claim."
                    if frac >= 0.4
                    else "attribution does NOT reliably beat the strong cheap baseline; the "
                    "peak-change baseline (diff_mag_max) is a competitive, gradient-free "
                    "alternative -- THIS is the calibration finding."
                )
            )
        )


if __name__ == "__main__":
    main()
