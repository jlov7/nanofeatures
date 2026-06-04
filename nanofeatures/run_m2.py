"""M2: the HONEST baseline test. M1 showed causal beats raw activation-magnitude,
but that baseline is a strawman — it picks BOS/positional features that are identical
between clean and corrupt (f_clean-f_corrupt ~= 0), so they recover nothing by
construction. The real question is whether EXACT per-feature patching (7205 forward
passes) buys a more faithful circuit than the CHEAP methods people actually use:

  - attribution patching : gradient linear approximation, ONE backward pass
  - diff-magnitude       : rank by |f_clean - f_corrupt|, zero model calls

If attribution patching nearly matches exact, the expensive method is unjustified and
that is the real (and useful) finding. If exact substantially wins, that justifies it.
Either way we report the truth, per the discipline carried over from nanocircuits.
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path

import torch

from .circuit import _metric, aligned_pairs, feature_rankings
from .model import load_gemma, load_sae, pick_device


def sufficiency(model, sae, hook, d, selected, f_clean, f_corrupt, m_clean, m_corrupt):
    """Patch ONLY the selected features (clean values) into the corrupt run; return
    recovered metric fraction."""
    if len(selected) == 0:
        return 0.0
    mask = torch.zeros(sae.cfg.d_sae, device=model.cfg.device)
    mask[torch.tensor(selected, device=model.cfg.device)] = 1.0
    fdiff = (f_clean - f_corrupt) * mask
    delta = fdiff @ sae.W_dec

    def patch(act, hook, _d=delta):
        return act + _d

    with torch.no_grad():
        m = _metric(
            model.run_with_hooks(d["corrupt"], fwd_hooks=[(hook, patch)]), d
        ).item()
    return (m - m_corrupt) / (m_clean - m_corrupt + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="reports/m2_baselines.json")
    args = ap.parse_args()
    device = pick_device(args.device)
    model = load_gemma(device)
    d = aligned_pairs(model)
    with torch.no_grad():
        m_clean = _metric(model(d["clean"]), d).item()
    print(f"[m2] n={d['n']} layer={args.layer} m_clean={m_clean:+.2f}")

    sae, hook = load_sae(args.layer, device=device)
    rankings, f_clean, f_corrupt, active, m_corrupt, effects, attribution = (
        feature_rankings(model, sae, hook, d)
    )
    print(f"[m2] m_corrupt={m_corrupt:+.2f}  active features={len(active)}")

    # rank agreement between the cheap approximation and the exact method
    causal_top = set(rankings["causal"][:64])
    attr_top = set(rankings["attribution"][:64])
    overlap = len(causal_top & attr_top) / len(causal_top)
    print(f"[m2] causal vs attribution top-64 overlap: {overlap:.0%}")

    def suff(sel):
        return sufficiency(
            model, sae, hook, d, sel, f_clean, f_corrupt, m_clean, m_corrupt
        )

    names = ["causal", "attribution", "diff_mag", "magnitude"]
    rows = []
    for k in args.ks:
        vals = {name: suff(rankings[name][:k]) for name in names}
        rnd = []
        for s in args.seeds:
            g = torch.Generator().manual_seed(s)
            sel = active[torch.randperm(len(active), generator=g)[:k]].tolist()
            rnd.append(suff(sel))
        vals["random"] = stats.mean(rnd)
        vals["k"] = k
        vals["exact_minus_attribution"] = vals["causal"] - vals["attribution"]
        vals["exact_minus_diffmag"] = vals["causal"] - vals["diff_mag"]
        rows.append(vals)
        print(
            f"[m2] k={k:>3}: causal={vals['causal']:+.1%} | attribution={vals['attribution']:+.1%} "
            f"| diff_mag={vals['diff_mag']:+.1%} | magnitude={vals['magnitude']:+.1%} "
            f"| random={vals['random']:+.1%}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "layer": args.layer,
                "n_pairs": d["n"],
                "m_clean": m_clean,
                "m_corrupt": m_corrupt,
                "n_active": len(active),
                "causal_vs_attribution_top64_overlap": overlap,
                "curve": rows,
            },
            indent=2,
        )
    )
    print(f"[m2] wrote {out}")

    # verdict: does EXACT patching beat the best CHEAP method by a meaningful margin?
    beats_attr = sum(1 for r in rows if r["exact_minus_attribution"] > 0.05)
    beats_diff = sum(1 for r in rows if r["exact_minus_diffmag"] > 0.05)
    print(
        f"[m2] exact beats attribution (>5pp) at {beats_attr}/{len(rows)} sizes; "
        f"exact beats diff-magnitude at {beats_diff}/{len(rows)} sizes"
    )
    if beats_attr >= len(rows) // 2:
        print(
            "[m2] VERDICT: exact per-feature patching is justified — the cheap "
            "gradient approximation leaves real faithfulness on the table."
        )
    else:
        print(
            "[m2] VERDICT: attribution patching is competitive — exact patching is "
            "NOT worth 7205x the compute here. (Honest negative; a useful one.)"
        )


if __name__ == "__main__":
    main()
