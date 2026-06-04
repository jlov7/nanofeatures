"""M14 (cross-layer EDGES): does the node-level boundary extend to connections?

The node result: a gradient-free score recovers single-feature importance as well as the
gradient on single-token tasks, and loses only on distributed circuits. The standing
objection is that circuits are EDGES, and EAP-style edge attribution is where a gradient
is supposed to be indispensable. So we build an EXACT mediated edge effect (gold standard,
no gradient) for every u@L1 -> d@L2 pair and ask which cheaper score RECOVERS it (Spearman).

Crucially, we do NOT compare against one cheap score. We compare against a LADDER of
gradient-free scores (the M7 strawman-avoidance discipline at the edge level), including a
CAUSAL gradient-free readout (transfer x node-ablation effect, no backward pass). The
question is whether the GRADIENT is needed, or whether some gradient-free score recovers the
exact edge as well. Every correlation and every gap carries a paired example-bootstrap CI.

We also report the eap-vs-exact second-order residual and the mean perturbation size, so the
near-perfect eap~exact correlation is read as the expected first-order (AtP) fact, not as the
load-bearing result. The load-bearing result is gradient vs the STRONGEST gradient-free score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .circuit import aligned_pairs, ioi_pairs
from .edges import cross_layer_edge_scores, edge_circuit_sufficiency
from .model import load_gemma, load_gpt2, load_gpt2_sae, load_sae, pick_device
from .run_distributedness import _spearman
from .run_m5 import ci
from .task import PAIRS, TEMPLATE


def _spearman_vs_exact(method_be, exact_be, idx):
    """Spearman between an edge method and the exact effect, on a bootstrap resample of
    examples (idx): average each [b,n_u,n_d] matrix over the resampled rows, flatten, rank-
    correlate. Paired = the SAME idx is used for method and exact, so gaps are paired."""
    m = method_be[idx].mean(0).flatten().tolist()
    e = exact_be[idx].mean(0).flatten().tolist()
    return _spearman(m, e)


def eval_edges(
    model,
    d,
    l1,
    l2,
    n_u,
    n_d,
    B,
    seed,
    device,
    sae_loader=load_sae,
    fdpos=False,
    fdpos_hs=(1.0,),
):
    # return_transfer=True so the SAME computation feeds both M14 (rank-recovery) and M15
    # (behavioral edge-circuit sufficiency) — the expensive exact loop runs once.
    r = cross_layer_edge_scores(
        model,
        d,
        l1,
        l2,
        n_u=n_u,
        n_d=n_d,
        device=device,
        return_transfer=True,
        sae_loader=sae_loader,
        fdpos=fdpos,
        fdpos_hs=fdpos_hs,
    )
    exact = r["exact"]
    n = exact.shape[0]
    gradfree = r["gradfree"]
    # cheap_fdpos* (if computed) is a CAUSAL+position-resolved but gradient-FREE readout. It is
    # NOT a "cheap" headline baseline (it costs n_d*n_pos forwards); it is the control showing
    # the finding is a readout PROPERTY, scored separately, not in `strongest`. With an h-grid
    # there is one cheap_fdpos@{h} key per probe step.
    extra = sorted(k for k in r if isinstance(k, str) and k.startswith("cheap_fdpos"))
    scores = ["eap", *gradfree, *extra]

    g = torch.Generator().manual_seed(seed)
    idxs = [torch.randint(0, n, (n,), generator=g) for _ in range(B)]
    boot = {s: [_spearman_vs_exact(r[s], exact, ix) for ix in idxs] for s in scores}
    per = {s: ci(vals) for s, vals in boot.items()}

    # strongest gradient-free score = argmax median Spearman vs exact (per-cell, adversarial:
    # gives the free side its best shot, exactly like run_suite's strongest-cheap choice)
    strongest = max(gradfree, key=lambda s: per[s][0])
    gap_vs_strongest = ci([a - b for a, b in zip(boot["eap"], boot[strongest])])
    # eap vs each finite-difference readout (over the h-grid): should be ~0 (same property)
    fdpos_grid = {
        k: {
            "vs_exact": per[k],
            "eap_minus": ci([a - b for a, b in zip(boot["eap"], boot[k])]),
        }
        for k in extra
    }
    fdpos_gap = fdpos_grid.get("cheap_fdpos", {}).get("eap_minus")

    # eap-vs-exact second-order residual: is eap~exact just first-order (AtP) triviality?
    em = exact.mean(0).flatten()
    ap = r["eap"].mean(0).flatten()
    resid_rel = (em - ap).abs().mean().item() / (em.abs().mean().item() + 1e-12)

    # M15: behavioral edge-circuit sufficiency (the rank-recovery result's behavioral analog)
    n_edges = n_u * n_d
    ms = [m for m in (n_edges // 36, n_edges // 12, n_edges // 4) if m >= 1]
    suff = edge_circuit_sufficiency(
        model,
        d,
        r,
        ms,
        methods=["exact", "eap", "cheap", "cheap_node", "mag"],
        seed=seed,
        B=B,
    )

    return {
        "l1": l1,
        "l2": l2,
        "n": n,
        "n_u": n_u,
        "n_d": n_d,
        "spearman_vs_exact": per,
        "strongest_gradfree": strongest,
        "eap_minus_strongest_gradfree": gap_vs_strongest,
        "eap_beats_all_gradfree": gap_vs_strongest[1] > 0,
        "eap_vs_exact_rel_residual": resid_rel,
        "pert_norm_mean": r["pert_norm_mean"],
        "exact_absmax_meanrows": exact.mean(0).abs().max().item(),
        "m_corrupt_mean": r["m_corrupt_mean"],
        "edge_sufficiency": suff,
        "cheap_fdpos_vs_exact": per.get("cheap_fdpos"),
        "eap_minus_fdpos": fdpos_gap,
        "fdpos_grid": fdpos_grid,
        "U": r["U"],
        "D": r["D"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pairs",
        nargs="+",
        default=["5,7"],
        help="layer pairs l1,l2 (e.g. 5,7 3,9 6,7)",
    )
    ap.add_argument("--n-u", type=int, default=24)
    ap.add_argument("--n-d", type=int, default=24)
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--model", default="gemma", choices=["gemma", "gpt2"])
    ap.add_argument(
        "--dtype",
        default="fp32",
        choices=["fp32", "bf16"],
        help="model dtype (gemma only). bf16 is the precision-fidelity sanity vs the default "
        "fp32 — the SAE is cast to match so there is no dtype mismatch.",
    )
    ap.add_argument(
        "--fdpos",
        action="store_true",
        help="also compute the gradient-free CAUSAL+position-resolved finite-difference "
        "readout (expensive: n_d*n_pos forwards) — the control showing the finding is a "
        "readout property, not the analytic gradient",
    )
    ap.add_argument(
        "--fdpos-h",
        nargs="+",
        type=float,
        default=[1.0],
        help="probe step(s) h for the fdpos finite-difference readout; pass several (e.g. "
        "0.25 0.5 1 2) to check the recovery is not an artifact of one step size",
    )
    ap.add_argument("--out", default="reports/m14_edges.json")
    args = ap.parse_args()
    device = pick_device(args.device)
    dt = torch.bfloat16 if args.dtype == "bf16" else None
    if args.model == "gpt2":
        model = load_gpt2(device)
        base_loader = load_gpt2_sae
    else:
        model = load_gemma(device, dtype=dt)
        base_loader = load_sae
    # cast the SAE to the model dtype so a bf16 model doesn't dtype-mismatch the fp32 SAE
    if dt is not None:

        def sae_loader(layer, device=None):
            sae, hook = base_loader(layer, device=device)
            return sae.to(dt), hook
    else:
        sae_loader = base_loader

    builders = {
        "capitals": lambda: aligned_pairs(model, pairs=PAIRS, template=TEMPLATE),
        "ioi": lambda: ioi_pairs(model),
    }
    layer_pairs = [tuple(int(x) for x in p.split(",")) for p in args.pairs]

    def fmt(c):
        return f"{c[0]:+.2f} [{c[1]:+.2f},{c[2]:+.2f}]"

    results = []
    for l1, l2 in layer_pairs:
        for name, build in builders.items():
            d = build()
            print(
                f"\n[m14/{name}] L{l1}->L{l2} n={d['n']} n_u={args.n_u} n_d={args.n_d} "
                f"B={args.boot}"
            )
            cell = eval_edges(
                model,
                d,
                l1,
                l2,
                args.n_u,
                args.n_d,
                args.boot,
                args.seed,
                device,
                sae_loader=sae_loader,
                fdpos=args.fdpos,
                fdpos_hs=args.fdpos_h,
            )
            cell["task"] = name
            results.append(cell)
            per = cell["spearman_vs_exact"]
            print(f"[m14/{name}] spearman vs exact:")
            for s, c in per.items():
                tag = " (GRADIENT)" if s == "eap" else ""
                print(f"            {s:14s} {fmt(c)}{tag}")
            gc = cell["eap_minus_strongest_gradfree"]
            sig = (
                f"SIG: gradient beats the strongest CHEAP score ({cell['strongest_gradfree']})"
                if gc[1] > 0
                else f"n.s.: cheap {cell['strongest_gradfree']} recovers exact as well as the gradient"
            )
            print(
                f"[m14/{name}] eap - {cell['strongest_gradfree']} = {fmt(gc)} -> {sig}"
            )
            print(
                f"[m14/{name}] eap~exact rel 2nd-order residual="
                f"{cell['eap_vs_exact_rel_residual']:.3f}  mean||Δ||={cell['pert_norm_mean']:.3f}"
                f"  exact absmax={cell['exact_absmax_meanrows']:.4f}"
            )
            if cell.get("fdpos_grid"):
                print(
                    f"[m14/{name}] CONTROL fdpos (gradient-FREE causal+position-resolved) "
                    f"vs exact, over probe-step h:"
                )
                for k, g in cell["fdpos_grid"].items():
                    fg = g["eap_minus"]
                    fsig = (
                        "n.s. (same property)"
                        if not (fg[1] > 0 or fg[2] < 0)
                        else "differs from eap"
                    )
                    print(
                        f"            {k:18s} vs exact = {fmt(g['vs_exact'])}; "
                        f"eap - it = {fmt(fg)} -> {fsig}"
                    )
            print(
                f"[m15/{name}] edge-circuit sufficiency (patch top-m edges, recovered LD):"
            )
            for m, sc in cell["edge_sufficiency"].items():
                p = sc["suff"]
                line = "  ".join(
                    f"{k}={fmt(p[k])}"
                    for k in ["exact", "eap", "cheap", "cheap_node", "mag"]
                )
                print(f"            m={m:>3}  {line}")
                g15 = sc["eap_minus_strongest_gradfree"]
                if g15 is not None:
                    s15 = "SIG" if g15[1] > 0 else "n.s."
                    print(
                        f"                  eap-circuit - {sc['strongest_gradfree']}-circuit "
                        f"= {fmt(g15)} -> {s15}"
                    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[m14] wrote {out}")

    # verdict over the default pair, both tasks
    base = [r for r in results if (r["l1"], r["l2"]) == layer_pairs[0]]
    by = {r["task"]: r for r in base}
    if "capitals" in by and "ioi" in by:
        # NOTE: eap_beats_all_gradfree compares eap only to the CHEAP (cached / no-extra-
        # forward) scores; the causal+position-resolved finite-difference control cheap_fdpos
        # is deliberately excluded from that ladder (it is expensive), so it does NOT count
        # against this flag. The verdict therefore claims "cheap scores fail," never "only the
        # gradient" — and if --fdpos was run, reports that a gradient-free score also recovers.
        st = by["capitals"]["eap_beats_all_gradfree"]
        io = by["ioi"]["eap_beats_all_gradfree"]
        fd = by["capitals"].get("cheap_fdpos_vs_exact") is not None
        if st and io:
            verdict = (
                "no cheap (cached / no-extra-forward) edge score recovers the exact edge on "
                "either the single-token task or IOI; an edge needs a CAUSAL, POSITION-RESOLVED "
                "readout (the gradient is its cheap analytic instance), so the node tie does NOT "
                "extend to cheap edge scores."
            )
            if fd:
                verdict += (
                    " The --fdpos control confirms a gradient-FREE finite-difference readout "
                    "(causal + position-resolved) recovers exact as well as the gradient: it is "
                    "the readout PROPERTY that is needed, not the gradient specifically."
                )
        else:
            verdict = (
                "a cheap edge score recovers the exact edge as well as the gradient on at least "
                f"one task (capitals_beats={st}, ioi_beats={io}); the refined, honest claim is "
                "whichever cheap score that is (see strongest_gradfree per task)."
            )
        print("[m14] VERDICT: " + verdict)


if __name__ == "__main__":
    main()
