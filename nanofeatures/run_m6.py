"""M6: did we beat a STRAWMAN? An independent adversarial review noted that the M2-M5
diff-magnitude baseline sums |f_clean-f_corrupt| over ALL positions, diluting the
contrastive signal with token-identical positions. The honest, stronger cheap baseline
is diff-magnitude restricted to the position that actually DIFFERS (the subject token),
to the answer position, or by position-max. If attribution still beats the BEST of these
with a paired-bootstrap CI excluding zero, Claim 1 survives its own discipline. If not,
Claim 1 must be rescoped.

No exact patching here (M2/M5 already settled AtP~=exact); attribution is the working
method, compared against every cheap baseline. One model load, both tasks, fast.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .circuit import aligned_pairs, feature_rankings
from .model import load_gemma, load_sae, pick_device
from .run_m5 import bootstrap_suff, ci, per_row_ld, per_row_patched_ld
from .task import PAIRS, TEMPLATE
from .tasks import ANTONYM_TASK

BASELINES = [
    "diff_mag",
    "diff_mag_subjpos",
    "diff_mag_lastpos",
    "diff_mag_max",
    "diff_mag_max_wdec",
    "magnitude",
]


def run_task(model, device, name, pairs, template, layer, ks, B, seed):
    d = aligned_pairs(model, pairs=pairs, template=template)
    sae, hook = load_sae(layer, device=device)
    # exact=False: skip the 7205-pass loop; we only need attribution + cheap baselines
    rankings, f_clean, f_corrupt = feature_rankings(model, sae, hook, d, exact=False)[
        :3
    ]

    clean_ld = per_row_ld(model, d, "clean")
    corrupt_ld = per_row_ld(model, d, "corrupt")
    n = d["n"]
    print(f"\n[m6/{name}] n={n} layer={layer} B={B}")

    cache = {}
    for rname in ["attribution", *BASELINES]:
        for k in ks:
            cache[(rname, k)] = per_row_patched_ld(
                model, sae, hook, d, rankings[rname][:k], f_clean, f_corrupt
            )

    g = torch.Generator().manual_seed(seed)
    idxs = [torch.randint(0, n, (n,), generator=g) for _ in range(B)]
    result = {"task": name, "layer": layer, "n": n, "B": B, "k": {}}

    def fmt(c):
        return f"{c[0]:+.0%} [{c[1]:+.0%},{c[2]:+.0%}]"

    for k in ks:
        attr_vals = [
            bootstrap_suff(clean_ld, corrupt_ld, cache[("attribution", k)], ix)
            for ix in idxs
        ]
        per = {"attribution": ci(attr_vals)}
        gaps = {}
        for b in BASELINES:
            base_vals = [
                bootstrap_suff(clean_ld, corrupt_ld, cache[(b, k)], ix) for ix in idxs
            ]
            per[b] = ci(base_vals)
            gv = [a - bb for a, bb in zip(attr_vals, base_vals)]
            gaps[b] = ci(gv)
        result["k"][k] = {"suff": per, "gaps_attr_minus": gaps}

        print(
            f"[m6/{name}] k={k:>3}  attr={fmt(per['attribution'])}  "
            + "  ".join(
                f"{b.replace('diff_mag', 'dm')}={fmt(per[b])}" for b in BASELINES
            )
        )
        # the decisive line: attribution vs the STRONGEST (highest-suff) cheap baseline
        strongest = max(BASELINES, key=lambda b: per[b][0])
        gc = gaps[strongest]
        sig = "SIGNIFICANT" if gc[1] > 0 else "n.s. (CI spans 0)"
        print(
            f"            strongest cheap baseline = {strongest} ({fmt(per[strongest])}); "
            f"attr - it = {fmt(gc)} -> {sig}"
        )
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--ks", nargs="+", type=int, default=[16, 32, 64])
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="reports/m6_strong_baselines.json")
    args = ap.parse_args()
    device = pick_device(args.device)
    model = load_gemma(device)

    results = [
        run_task(
            model,
            device,
            "capitals",
            PAIRS,
            TEMPLATE,
            args.layer,
            args.ks,
            args.boot,
            args.seed,
        ),
        run_task(
            model,
            device,
            "antonyms",
            *ANTONYM_TASK,
            args.layer,
            args.ks,
            args.boot,
            args.seed,
        ),
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[m6] wrote {out}")

    # overall: does attribution beat the strongest cheap baseline (CI>0) at every cell?
    cells, wins = 0, 0
    for r in results:
        for k, kd in r["k"].items():
            per = kd["suff"]
            strongest = max(BASELINES, key=lambda b: per[b][0])
            cells += 1
            if kd["gaps_attr_minus"][strongest][1] > 0:
                wins += 1
    print(
        f"[m6] attribution beats the STRONGEST cheap baseline (CI excludes 0) at {wins}/{cells} cells"
    )
    print(
        "[m6] VERDICT: "
        + (
            "Claim 1 SURVIVES the stronger baseline."
            if wins >= cells - 1
            else "Claim 1 must be RESCOPED — a position-restricted diff-mag is competitive."
        )
    )


if __name__ == "__main__":
    main()
