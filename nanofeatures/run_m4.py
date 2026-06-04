"""M4: is the finding capitals-specific, or task-general? Re-run the full comparison
(exact + attribution + diff-mag + magnitude + random, sufficiency AND necessity) on a
DIFFERENT relation — lexical antonyms ("The opposite of X is ...") — which exercises a
different mechanism than factual geography recall.

Gated on behavior: if Gemma-2-2B doesn't do the antonym task cleanly (clean metric not
strongly positive), the attribution analysis is meaningless and we say so.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .circuit import _metric, aligned_pairs
from .model import load_gemma, pick_device
from .run_m3 import eval_layer
from .tasks import ANTONYM_TASK


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="reports/m4_antonyms.json")
    args = ap.parse_args()
    device = pick_device(args.device)
    model = load_gemma(device)

    pairs, template = ANTONYM_TASK
    d = aligned_pairs(model, pairs=pairs, template=template)
    with torch.no_grad():
        m_clean = _metric(model(d["clean"]), d).item()
        m_corrupt = _metric(model(d["corrupt"]), d).item()
    print(f"[m4] antonym task: n={d['n']} -> {d['countries']}")
    print(f"[m4] template: {template!r}")
    print(f"[m4] m_clean={m_clean:+.2f} m_corrupt={m_corrupt:+.2f}")

    gate_ok = m_clean > 2.0 and m_clean - m_corrupt > 4.0 and d["n"] >= 8
    print(
        f"[m4] BEHAVIOR GATE: {'PASS' if gate_ok else 'WEAK'} "
        f"(need m_clean>2, clean-corrupt gap>4, n>=8 for attribution to be meaningful)"
    )
    if not gate_ok:
        print(
            "[m4] task too weak/small for a clean attribution result — reporting the "
            "gate failure honestly rather than analysing noise."
        )

    r = eval_layer(model, args.layer, d, args.ks, args.seeds, device, exact=True)
    print(
        f"[m4] layer {args.layer}: {r['n_active']} active | "
        f"causal~attr top64 overlap={r['causal_attr_top64_overlap']:.0%}"
    )
    for row in r["curve"]:
        print(
            f"     k={row['k']:>3} SUFF causal={row['suff_causal']:+.0%} "
            f"attr={row['suff_attribution']:+.0%} diff={row['suff_diff_mag']:+.0%} "
            f"mag={row['suff_magnitude']:+.0%} rnd={row['suff_random']:+.0%}  ||  "
            f"NEC causal={row['nec_causal']:+.0%} attr={row['nec_attribution']:+.0%} "
            f"diff={row['nec_diff_mag']:+.0%} rnd={row['nec_random']:+.0%}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "task": "antonyms",
                "template": template,
                "n_pairs": d["n"],
                "behavior_gate_pass": gate_ok,
                "m_clean": m_clean,
                "m_corrupt": m_corrupt,
                "layer": args.layer,
                **r,
            },
            indent=2,
        )
    )
    print(f"[m4] wrote {out}")

    row64 = next((row for row in r["curve"] if row["k"] == 64), r["curve"][-1])
    attr_beats_diff = row64["suff_attribution"] - row64["suff_diff_mag"]
    exact_minus_attr = row64["suff_causal"] - row64["suff_attribution"]
    print(
        f"[m4] at k={row64['k']}: attribution beats diff-mag by {attr_beats_diff:+.0%}; "
        f"exact beats attribution by {exact_minus_attr:+.0%}"
    )
    if gate_ok and attr_beats_diff > 0.05 and abs(exact_minus_attr) < 0.05:
        print(
            "[m4] VERDICT: the capitals finding REPLICATES on antonyms — "
            "attribution >> diff-mag, and exact ~= attribution. Task-general."
        )
    elif gate_ok:
        print(
            "[m4] VERDICT: pattern differs on antonyms — report the difference; "
            "the finding is task-dependent."
        )
    else:
        print("[m4] VERDICT: inconclusive — behavior gate failed; do not over-read.")


if __name__ == "__main__":
    main()
