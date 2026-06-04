"""M11 (generality): does the distributedness boundary replicate on a SECOND model? Same
ladder, same metric, same baselines, but GPT-2-small + Joseph Bloom's residual SAEs instead
of Gemma-2-2B + Gemma Scope. IOI is GPT-2-small's home turf (Wang et al.), so the
distributed side should be strong; single-token factual recall is hard for GPT-2-small, so
only the tasks that pass the behavior gate are tested on the tie side.

Prediction if the boundary is a property of circuit topology (not of Gemma): single-token
tasks that gate -> attribution ties the cheap baseline; IOI -> attribution wins.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .circuit import aligned_pairs, ioi_pairs
from .model import load_gpt2, load_gpt2_sae, pick_device
from .run_m5 import per_row_ld
from .run_suite import eval_cell, gate
from .tasks import TASK_SUITE


def _fmt(c):
    return f"{c[0]:+.0%} [{c[1]:+.0%},{c[2]:+.0%}]"


def _run(model, device, name, d, layers, ks, B, seed):
    ok, info = gate(model, d, min_n=8, min_sep=1.0)
    if not ok:
        print(f"[m11/{name}] SKIP ({info.get('reason')})")
        return None
    clean_ld = per_row_ld(model, d, "clean")
    corrupt_ld = per_row_ld(model, d, "corrupt")
    layers_out = {}
    for layer in layers:
        sae, hook = load_gpt2_sae(layer, device=device)
        cells = eval_cell(model, sae, hook, d, ks, B, seed, False, clean_ld, corrupt_ld)
        layers_out[layer] = cells
        for k in ks:
            cell = cells[k]
            avs = cell["attr_minus_strongest"]
            print(
                f"[m11/{name}] L{layer:>2} k={k:>3} attr={_fmt(cell['suff']['attribution'])} "
                f"| attr-{cell['strongest_cheap']}={_fmt(avs)} -> "
                + ("SIG" if avs[1] > 0 else "n.s.")
            )
    return {"task": name, "n": d["n"], "info": info, "layers": layers_out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+", type=int, default=[5, 7, 9])
    ap.add_argument("--ks", nargs="+", type=int, default=[16, 32, 64])
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="reports/m11_gpt2.json")
    args = ap.parse_args()
    device = pick_device(args.device)
    model = load_gpt2(device)
    print(f"[m11] GPT-2-small | layers={args.layers} ks={args.ks}")

    results = []
    for name in TASK_SUITE:
        pairs, template = TASK_SUITE[name]
        r = _run(
            model,
            device,
            name,
            aligned_pairs(model, pairs=pairs, template=template),
            args.layers,
            args.ks,
            args.boot,
            args.seed,
        )
        if r:
            r["kind"] = "single"
            results.append(r)
    r = _run(
        model,
        device,
        "ioi",
        ioi_pairs(model),
        args.layers,
        args.ks,
        args.boot,
        args.seed,
    )
    if r:
        r["kind"] = "distributed"
        results.append(r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[m11] wrote {out}")

    # boundary summary: among gated tasks, do single-token tasks tie while IOI wins?
    def wins(r):
        return sum(
            cell["attr_beats_strongest"]
            for ld in r["layers"].values()
            for cell in ld.values()
        )

    def cells(r):
        return sum(len(ld) for ld in r["layers"].values())

    single = [r for r in results if r["kind"] == "single"]
    dist = [r for r in results if r["kind"] == "distributed"]
    sw, sc = sum(wins(r) for r in single), sum(cells(r) for r in single)
    print(
        f"[m11] gated single-token tasks: {[r['task'] for r in single]} -> attribution "
        f"beats cheap at {sw}/{sc} cells"
    )
    for r in dist:
        print(
            f"[m11] {r['task']} (distributed) -> attribution beats cheap at {wins(r)}/{cells(r)} cells"
        )
    if dist and single is not None:
        dw = sum(wins(r) for r in dist)
        dc = sum(cells(r) for r in dist)
        print(
            "[m11] VERDICT: "
            + (
                "boundary REPLICATES on GPT-2-small (single-token tie, IOI wins)."
                if (sc and sw / sc < 0.5 and dc and dw / dc > 0.5)
                else "boundary does NOT cleanly replicate — report the per-task breakdown."
            )
        )
    elif not single:
        print(
            "[m11] NOTE: no single-token task gated on GPT-2-small; only the distributed side is testable here."
        )


if __name__ == "__main__":
    main()
