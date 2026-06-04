"""M1: does a CAUSALLY-attributed SAE feature circuit beat trivial baselines
(activation magnitude, random) on FAITHFULNESS (sufficiency)? On a real model with
no ground truth, this is the honest question — and it can't be a structural artifact
because faithfulness is a behavioral causal measure.
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path

import torch

from .circuit import _metric, aligned_pairs, feature_effects, layer_scan
from .model import load_gemma, load_sae, pick_device


def sufficiency(model, sae, hook, d, selected, f_clean, f_corrupt, m_clean, m_corrupt):
    """Patch ONLY the selected features (clean values) into the corrupt run; return
    recovered metric fraction in [0,1]-ish."""
    if len(selected) == 0:
        return 0.0
    mask = torch.zeros(sae.cfg.d_sae, device=model.cfg.device)
    mask[torch.tensor(selected, device=model.cfg.device)] = 1.0
    fdiff = (f_clean - f_corrupt) * mask  # [b,pos,d_sae]
    delta = fdiff @ sae.W_dec  # [b,pos,d_model]

    def patch(act, hook, _d=delta):
        return act + _d

    with torch.no_grad():
        m = _metric(
            model.run_with_hooks(d["corrupt"], fwd_hooks=[(hook, patch)]), d
        ).item()
    return (m - m_corrupt) / (m_clean - m_corrupt + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scan-layers", nargs="+", type=int, default=[3, 5, 7, 9, 11, 13, 15, 18, 20]
    )
    ap.add_argument(
        "--layer",
        type=int,
        default=7,
        help="layer for feature attribution (a parameter, not argmax of "
        "the scan: full-resid recovery saturates and can't select it)",
    )
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="reports/m1_feature_circuit.json")
    args = ap.parse_args()
    device = pick_device(args.device)
    model = load_gemma(device)
    d = aligned_pairs(model)
    print(
        f"[m1] aligned pairs (single-token country+capital): n={d['n']} -> {d['countries']}"
    )

    scan, m_clean, m_corrupt = layer_scan(model, d, args.scan_layers)
    print(f"[m1] metric clean={m_clean:+.2f} corrupt={m_corrupt:+.2f}")
    print(
        "[m1] layer scan (full-resid patch recovery; DESCRIPTIVE only — saturates "
        "downstream, so NOT used to pick the attribution layer):"
    )
    for L in sorted(scan):
        print(f"     layer {L:>2}: recovered {scan[L]:+.2%}")
    layer = args.layer
    print(f"[m1] feature attribution at layer {layer} (parameter)")

    sae, hook = load_sae(layer, device=device)
    with torch.no_grad():
        _, cc = model.run_with_cache(d["clean"], names_filter=hook)
        _, ccor = model.run_with_cache(d["corrupt"], names_filter=hook)
        f_clean, f_corrupt = sae.encode(cc[hook]), sae.encode(ccor[hook])
    effects, mag, active = feature_effects(model, sae, hook, d)
    print(f"[m1] layer {layer}: {len(active)} active candidate features")

    causal_rank = effects.argsort(descending=True).tolist()
    mag_rank = mag.argsort(descending=True).tolist()

    def suff(sel):
        return sufficiency(
            model, sae, hook, d, sel, f_clean, f_corrupt, m_clean, m_corrupt
        )

    rows = []
    for k in args.ks:
        causal_k = suff(causal_rank[:k])
        magnitude_k = suff(mag_rank[:k])
        rnd = []
        for s in args.seeds:
            g = torch.Generator().manual_seed(s)
            sel = active[torch.randperm(len(active), generator=g)[:k]].tolist()
            rnd.append(suff(sel))
        random_k = stats.mean(rnd)
        rows.append(
            {
                "k": k,
                "causal": causal_k,
                "magnitude": magnitude_k,
                "random": random_k,
                "causal_lift_over_magnitude": causal_k - magnitude_k,
            }
        )
        print(
            f"[m1] k={k:>3}: causal suff={causal_k:+.2%} | magnitude={magnitude_k:+.2%} "
            f"| random={random_k:+.2%} | causal lift over magnitude={causal_k - magnitude_k:+.2%}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "layer": layer,
                "n_pairs": d["n"],
                "m_clean": m_clean,
                "m_corrupt": m_corrupt,
                "n_active": len(active),
                "curve": rows,
            },
            indent=2,
        )
    )
    beats = [r for r in rows if r["causal_lift_over_magnitude"] > 0.05]
    print(f"[m1] wrote {out}")
    print(
        f"[m1] causal beats magnitude (lift>5pp) at {len(beats)}/{len(rows)} circuit sizes "
        f"-> {'causal attribution adds real signal' if len(beats) >= len(rows) // 2 else 'NEGATIVE: magnitude is competitive'}"
    )


if __name__ == "__main__":
    main()
