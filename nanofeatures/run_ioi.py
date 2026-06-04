"""M8 (the decisive test of the M7 negative result's scope): does attribution ≈ cheap
peak-change baseline STILL hold on a genuinely DISTRIBUTED, multi-position circuit?

The M7 sweep used single-token contrastive tasks where clean and corrupt differ at ONE
position, so the changed-feature set is tiny and concentrated and ANY contrastive selector
recovers it — an adversarial reviewer correctly noted that diff_mag_max ≈ attribution is
near-tautological there, and that gradient attribution / AtP* exist precisely for
distributed circuits with path cancellation (IOI: duplicate-token, S-inhibition,
name-mover heads). This runs the IDENTICAL ladder on IOI (Gemma-2-2B, SAE features), where
clean vs corrupt differ at THREE name-token positions. Two outcomes, both honest:

  - attribution STILL ties diff_mag_max  -> the negative result generalizes beyond the easy
    regime: cheap peak-change suffices even for a distributed circuit (strong claim).
  - attribution PULLS AHEAD here          -> we have found the boundary: cheap suffices for
    single-token recall/morphology, but the gradient earns its cost on distributed
    circuits (an equally honest, sharper claim).

Same bootstrap machinery, same metric, same baselines as run_suite -> directly comparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .circuit import ioi_pairs
from .model import load_gemma, load_neuron_basis, load_sae, pick_device
from .run_m5 import per_row_ld
from .run_suite import eval_cell, gate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+", type=int, default=[5, 7, 9, 11])
    ap.add_argument("--ks", nargs="+", type=int, default=[16, 32, 64])
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--no-exact", action="store_true", help="skip exact causal (faster)"
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
    ap.add_argument("--out", default="reports/m8_ioi.json")
    args = ap.parse_args()
    exact = not args.no_exact
    device = pick_device(args.device)
    model = load_gemma(device)

    d = ioi_pairs(model)
    ok, info = gate(model, d, min_n=8, min_sep=1.0)
    print(
        f"[ioi] n={info['n']} mean_clean_logit_diff(IO-S)="
        f"{info.get('mean_clean_logit_diff', float('nan')):+.2f} "
        f"sep={info.get('clean_minus_corrupt_sep', float('nan')):+.2f} gated={ok}"
    )
    if not ok:
        print(f"[ioi] SKIP: {info['reason']}")
        return

    clean_ld = per_row_ld(model, d, "clean")
    corrupt_ld = per_row_ld(model, d, "corrupt")

    def fmt(c):
        return f"{c[0]:+.0%} [{c[1]:+.0%},{c[2]:+.0%}]"

    layers_out = {}
    for layer in args.layers:
        sae, hook = (
            load_neuron_basis(model, layer, device)
            if args.neuron
            else load_sae(layer, device=device, width=args.width)
        )
        cells = eval_cell(
            model,
            sae,
            hook,
            d,
            args.ks,
            args.boot,
            args.seed,
            exact,
            clean_ld,
            corrupt_ld,
        )
        layers_out[layer] = cells
        for k in args.ks:
            cell = cells[k]
            per = cell["suff"]
            line = f"[ioi] L{layer:>2} k={k:>3}  attr={fmt(per['attribution'])}"
            if exact:
                line += f"  causal={fmt(per['causal'])}"
            print(line)
            avs = cell["attr_minus_strongest"]
            print(
                f"          strongest cheap = {cell['strongest_cheap']} "
                f"({fmt(per[cell['strongest_cheap']])}); attr - it = {fmt(avs)} -> "
                + ("SIG" if avs[1] > 0 else "n.s.")
            )
            if exact:
                cg = cell["causal_minus_attr"]
                print(
                    f"          causal - attr = {fmt(cg)} -> "
                    + ("exact WORTH it" if cg[1] > 0 else "exact ~= attr")
                )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "task": "ioi",
                "n": d["n"],
                "B": args.boot,
                "info": info,
                "layers": layers_out,
            },
            indent=2,
        )
    )
    print(f"\n[ioi] wrote {out}")

    cells = wins = causal_cells = causal_wins = 0
    for layer_cells in layers_out.values():
        for cell in layer_cells.values():
            cells += 1
            wins += cell["attr_beats_strongest"]
            if "causal_beats_attr" in cell:
                causal_cells += 1
                causal_wins += cell["causal_beats_attr"]
    print(
        f"[ioi] attribution beats strongest cheap baseline (CI>0) at {wins}/{cells} cells"
    )
    if causal_cells:
        print(
            f"[ioi] exact causal beats attribution (CI>0) at {causal_wins}/{causal_cells} cells"
        )
    if wins >= max(1, cells - 1):
        print(
            "[ioi] VERDICT: on a DISTRIBUTED circuit attribution DOES beat the cheap "
            "baseline -> the M7 negative result is SCOPED to single-token tasks; the "
            "gradient earns its cost where the circuit is distributed."
        )
    elif wins == 0:
        print(
            "[ioi] VERDICT: even on IOI, cheap peak-change ties attribution -> the negative "
            "result GENERALIZES beyond single-token tasks (the stronger claim)."
        )
    else:
        print(
            f"[ioi] VERDICT: MIXED ({wins}/{cells}) -> attribution's distributed-circuit "
            "advantage is partial/k-dependent; report the per-cell breakdown."
        )


if __name__ == "__main__":
    main()
