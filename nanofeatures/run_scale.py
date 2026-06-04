"""M17 (scale generality): does the topology boundary hold on Gemma-2-9B?

The boundary (attribution ties the strongest cheap baseline on single-token tasks, wins on a
distributed circuit) is shown on Gemma-2-2B and GPT-2-small (M11). This runs the IDENTICAL
ladder on Gemma-2-9B + Gemma Scope 9B SAEs, the frontier-scale point, loaded in bf16 so it
fits in 48GB. Same eval_cell, gate, metric, baselines as run_suite/run_ioi -> directly
comparable. No --exact (the per-feature loop is too slow at 9B and AtP=exact is already
established); the question here is purely attribution vs the strongest cheap baseline, single-
token vs distributed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .circuit import aligned_pairs, ioi_pairs
from .model import load_gemma9b, load_sae9b, pick_device
from .run_m5 import per_row_ld
from .run_suite import eval_cell, gate
from .tasks import TASK_SUITE

# a representative single-token subset (keep 9B runtime modest) + IOI as the distributed circuit
SINGLE = ["capitals", "antonyms", "country_language", "past_tense"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+", type=int, default=[20])
    ap.add_argument("--ks", nargs="+", type=int, default=[16, 32, 64])
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", default="16k")
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--dtype",
        choices=["bf16", "fp32"],
        default="bf16",
        help="9B precision. bf16 (default, committed result) or fp32 precision-validation run. "
        "fp32 forces --device cpu: 36GB fits 48GB only off-GPU, and CPU cannot trip the MPS "
        "watchdog a swap-thrashing fp32 model on the GPU would.",
    )
    ap.add_argument("--out", default="reports/m17_gemma9b.json")
    args = ap.parse_args()
    dt = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    if args.dtype == "fp32" and args.device is None:
        args.device = "cpu"
    # never let an fp32 run silently overwrite the committed bf16 report
    if args.dtype == "fp32" and args.out == "reports/m17_gemma9b.json":
        args.out = "reports/m17_gemma9b_fp32.json"
    device = pick_device(args.device)
    model = load_gemma9b(device, dtype=dt)

    def fmt(c):
        return f"{c[0]:+.0%} [{c[1]:+.0%},{c[2]:+.0%}]"

    builders = {
        name: lambda n=name: aligned_pairs(model, *TASK_SUITE[n]) for name in SINGLE
    }
    builders["ioi"] = lambda: ioi_pairs(model)

    results = []
    single_cells = single_wins = ioi_cells = ioi_wins = 0
    for name, build in builders.items():
        d = build()
        ok, info = gate(model, d, min_n=8, min_sep=1.0)
        print(
            f"\n[m17/{name}] n={info['n']} gated={ok} ({info.get('reason')})",
            flush=True,
        )
        if not ok:
            results.append({"task": name, "gated": False, "info": info})
            continue
        clean_ld = per_row_ld(model, d, "clean")
        corrupt_ld = per_row_ld(model, d, "corrupt")
        layers_out = {}
        for layer in args.layers:
            sae, hook = load_sae9b(layer, device=device, width=args.width, dtype=dt)
            cells = eval_cell(
                model,
                sae,
                hook,
                d,
                args.ks,
                args.boot,
                args.seed,
                False,
                clean_ld,
                corrupt_ld,
            )
            layers_out[layer] = cells
            for k in args.ks:
                cell = cells[k]
                avs = cell["attr_minus_strongest"]
                sig = "SIG" if avs[1] > 0 else "n.s."
                print(
                    f"[m17/{name}] L{layer} k={k:>3} attr={fmt(cell['suff']['attribution'])} "
                    f"strongest({cell['strongest_cheap']})={fmt(cell['suff'][cell['strongest_cheap']])} "
                    f"attr-it={fmt(avs)} -> {sig}",
                    flush=True,
                )
                if name == "ioi":
                    ioi_cells += 1
                    ioi_wins += cell["attr_beats_strongest"]
                else:
                    single_cells += 1
                    single_wins += cell["attr_beats_strongest"]
        results.append({"task": name, "gated": True, "n": d["n"], "layers": layers_out})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(
        f"\n[m17] Gemma-2-9B: attribution beats strongest cheap baseline at "
        f"{single_wins}/{single_cells} single-token cells and {ioi_wins}/{ioi_cells} IOI cells."
    )
    print(
        "[m17] VERDICT: "
        + (
            "the topology boundary REPLICATES at 9B scale (single-token tie, IOI win)."
            if (
                ioi_cells
                and ioi_wins >= max(1, ioi_cells - 1)
                and single_wins <= single_cells // 2
            )
            else "see per-cell breakdown (does not cleanly match the 2B/GPT-2 pattern)."
        )
    )


if __name__ == "__main__":
    main()
