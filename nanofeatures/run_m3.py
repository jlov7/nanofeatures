"""M3: is the M2 finding ROBUST, or an n=1 (one layer, one direction) anecdote?

M2 showed at layer 7: attribution patching ~= exact, and both >> diff-magnitude >>
magnitude ~= random, on SUFFICIENCY. A frontier-grade claim needs more:

  1. Layer generalization: does the ladder (random < magnitude < diff_mag <
     attribution) hold across the network, or only at layer 7?
  2. Necessity as a corroboration of sufficiency: patching the corrupt features into the
     clean run should DESTROY the behavior. NOTE: for a symmetric additive patch around
     two fixed endpoints, necessity is near-mirror to sufficiency (Pearson ~0.999 in
     practice), so it confirms rather than adds independent evidence.
  3. Equivalence spot-check: confirm attribution ~= exact at a second layer, so the
     "don't bother with exact" conclusion isn't layer-7-specific.

We use ATTRIBUTION as the working method across layers (M2 earned that: it's ~free and
~= exact) and run EXACT only as a spot-check at one extra layer.
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path

import torch

from .circuit import _metric, aligned_pairs, feature_rankings
from .model import load_gemma, load_sae, pick_device


def _patched_metric(model, sae, hook, d, selected, f_src, f_dst, base_tokens):
    """Run `base_tokens` but overwrite the selected features' values from f_dst->f_src
    at the hook, and return the resulting metric. delta = (f_src - f_dst)*mask @ W_dec."""
    if len(selected) == 0:
        with torch.no_grad():
            return _metric(model(base_tokens), d).item()
    mask = torch.zeros(sae.cfg.d_sae, device=model.cfg.device)
    mask[torch.tensor(selected, device=model.cfg.device)] = 1.0
    delta = ((f_src - f_dst) * mask) @ sae.W_dec

    def patch(act, hook, _d=delta):
        return act + _d

    with torch.no_grad():
        return _metric(
            model.run_with_hooks(base_tokens, fwd_hooks=[(hook, patch)]), d
        ).item()


def sufficiency(model, sae, hook, d, sel, f_clean, f_corrupt, m_clean, m_corrupt):
    """Add the selected features' CLEAN values into the CORRUPT run; fraction recovered."""
    m = _patched_metric(model, sae, hook, d, sel, f_clean, f_corrupt, d["corrupt"])
    return (m - m_corrupt) / (m_clean - m_corrupt + 1e-9)


def necessity(model, sae, hook, d, sel, f_clean, f_corrupt, m_clean, m_corrupt):
    """Force the selected features' CORRUPT values into the CLEAN run; fraction of the
    behavior destroyed. High = removing these features breaks the task = necessary."""
    m = _patched_metric(model, sae, hook, d, sel, f_corrupt, f_clean, d["clean"])
    return (m_clean - m) / (m_clean - m_corrupt + 1e-9)


def eval_layer(model, layer, d, ks, seeds, device, exact):
    sae, hook = load_sae(layer, device=device)
    out = feature_rankings(model, sae, hook, d, exact=exact)
    rankings, f_clean, f_corrupt, active, m_corrupt = (
        out[0],
        out[1],
        out[2],
        out[3],
        out[4],
    )
    with torch.no_grad():
        m_clean = _metric(model(d["clean"]), d).item()

    def suff(sel):
        return sufficiency(
            model, sae, hook, d, sel, f_clean, f_corrupt, m_clean, m_corrupt
        )

    def nec(sel):
        return necessity(
            model, sae, hook, d, sel, f_clean, f_corrupt, m_clean, m_corrupt
        )

    rows = []
    for k in ks:
        row = {"k": k}
        for name, rank in rankings.items():
            row[f"suff_{name}"] = suff(rank[:k])
            row[f"nec_{name}"] = nec(rank[:k])
        rnd_s, rnd_n = [], []
        for s in seeds:
            g = torch.Generator().manual_seed(s)
            sel = active[torch.randperm(len(active), generator=g)[:k]].tolist()
            rnd_s.append(suff(sel))
            rnd_n.append(nec(sel))
        row["suff_random"] = stats.mean(rnd_s)
        row["nec_random"] = stats.mean(rnd_n)
        rows.append(row)

    overlap = None
    if exact:
        c, a = set(rankings["causal"][:64]), set(rankings["attribution"][:64])
        overlap = len(c & a) / len(c)
    return {
        "layer": layer,
        "n_active": len(active),
        "m_clean": m_clean,
        "m_corrupt": m_corrupt,
        "causal_attr_top64_overlap": overlap,
        "curve": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--layers", nargs="+", type=int, default=[2, 4, 6, 8, 10, 12, 16, 20]
    )
    ap.add_argument(
        "--exact-layers",
        nargs="+",
        type=int,
        default=[11],
        help="layers to ALSO run exact patching on (equivalence spot-check)",
    )
    ap.add_argument("--ks", nargs="+", type=int, default=[8, 16, 32, 64])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="reports/m3_robustness.json")
    args = ap.parse_args()
    device = pick_device(args.device)
    model = load_gemma(device)
    d = aligned_pairs(model)
    print(f"[m3] n={d['n']} layers={args.layers} exact-spot-check={args.exact_layers}")

    results = []
    for layer in sorted(set(args.layers) | set(args.exact_layers)):
        exact = layer in args.exact_layers
        r = eval_layer(model, layer, d, args.ks, args.seeds, device, exact)
        results.append(r)
        tag = " (EXACT spot-check)" if exact else ""
        ov = (
            f" | causal~attr top64 overlap={r['causal_attr_top64_overlap']:.0%}"
            if exact
            else ""
        )
        print(f"[m3] layer {layer:>2}{tag}: {r['n_active']} active{ov}")
        for row in r["curve"]:
            line = (
                f"     k={row['k']:>3} SUFF attr={row['suff_attribution']:+.0%} "
                f"diff={row['suff_diff_mag']:+.0%} mag={row['suff_magnitude']:+.0%} "
                f"rnd={row['suff_random']:+.0%}  ||  NEC attr={row['nec_attribution']:+.0%} "
                f"diff={row['nec_diff_mag']:+.0%} rnd={row['nec_random']:+.0%}"
            )
            if exact:
                line += f"  [exact suff={row['suff_causal']:+.0%} nec={row['nec_causal']:+.0%}]"
            print(line)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_pairs": d["n"], "layers": results}, indent=2))
    print(f"[m3] wrote {out}")

    # verdict at k=64 vs BOTH baselines: the weak summed diff_mag AND the strong
    # diff_mag_max. The gap vs the strong baseline is the honest one (see README).
    big = [r for r in results if any(row["k"] == 64 for row in r["curve"])]
    wins_weak = wins_strong = 0
    for r in big:
        row = next(row for row in r["curve"] if row["k"] == 64)
        if row["suff_attribution"] - row["suff_diff_mag"] > 0.05:
            wins_weak += 1
        if row["suff_attribution"] - row.get("suff_diff_mag_max", 0.0) > 0.05:
            wins_strong += 1
    print(
        f"[m3] attribution beats diff_mag (weak) at {wins_weak}/{len(big)} layers; "
        f"beats diff_mag_max (STRONG) at {wins_strong}/{len(big)} layers (suff, k=64)"
    )
    if wins_strong >= max(1, len(big) - 1):
        print(
            "[m3] VERDICT: causal advantage ROBUST across layers vs the strong baseline."
        )
    else:
        print(
            "[m3] VERDICT: vs the STRONG baseline the per-layer advantage is small/"
            "inconsistent — the large mid-network gap is mostly an artifact of the weak "
            "summed baseline (the localization of *signal*, not of a *causal win*)."
        )


if __name__ == "__main__":
    main()
