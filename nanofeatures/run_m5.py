"""M5: confidence intervals. n is small (20-24 single-token pairs), so every gap quoted
elsewhere needs error bars before it can be believed. We use a PAIRED bootstrap over
pairs: cache each pair's clean / corrupt / patched logit-difference ONCE (no extra
forward passes), then resample pairs with replacement and recompute the sufficiency
ratio. Paired = the same resample is used for competing rankings, so the CI is on the
GAP directly (does it exclude zero?).

This is the test that turns 'attribution beats diff-mag by 21pp' and 'exact beats
attribution by 7pp on antonyms' into claims with stated uncertainty.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .circuit import aligned_pairs, feature_rankings
from .model import load_gemma, load_sae, pick_device
from .task import PAIRS, TEMPLATE, logit_diff
from .tasks import ANTONYM_TASK


def per_row_ld(model, d, key):
    """Per-pair logit-difference on the clean or corrupt run (no patch)."""
    with torch.no_grad():
        logits = model(d[key])
    return logit_diff(logits, d["ans_clean"], d["ans_corrupt"], d["end"])


def per_row_patched_ld(model, sae, hook, d, selected, f_clean, f_corrupt):
    """Per-pair logit-difference on the corrupt run with `selected` features set to
    their clean values (sufficiency patch)."""
    if len(selected) == 0:
        return per_row_ld(model, d, "corrupt")
    # match the activation dtype (bf16 for the 9B model) so the mask*acts product and the
    # @ W_dec matmul don't dtype-mismatch under reduced precision
    mask = torch.zeros(sae.cfg.d_sae, device=model.cfg.device, dtype=f_clean.dtype)
    mask[torch.tensor(selected, device=model.cfg.device)] = 1.0
    delta = ((f_clean - f_corrupt) * mask) @ sae.W_dec

    def patch(act, hook, _d=delta):
        return act + _d

    with torch.no_grad():
        logits = model.run_with_hooks(d["corrupt"], fwd_hooks=[(hook, patch)])
    return logit_diff(logits, d["ans_clean"], d["ans_corrupt"], d["end"])


def bootstrap_suff(clean_ld, corrupt_ld, patched_ld, idx):
    """Sufficiency ratio for one resample index tensor (rows)."""
    c = clean_ld[idx].mean()
    z = corrupt_ld[idx].mean()
    p = patched_ld[idx].mean()
    return ((p - z) / (c - z + 1e-9)).item()


def ci(vals, lo=2.5, hi=97.5):
    t = torch.tensor(vals)
    return (
        t.median().item(),
        torch.quantile(t, lo / 100).item(),
        torch.quantile(t, hi / 100).item(),
    )


def run_task(model, device, name, pairs, template, layer, ks, B, seed):
    d = aligned_pairs(model, pairs=pairs, template=template)
    sae, hook = load_sae(layer, device=device)
    out = feature_rankings(model, sae, hook, d, exact=True)
    rankings, f_clean, f_corrupt = out[0], out[1], out[2]

    clean_ld = per_row_ld(model, d, "clean")
    corrupt_ld = per_row_ld(model, d, "corrupt")
    n = d["n"]
    print(f"\n[m5/{name}] n={n} layer={layer} B={B}")

    # cache per-row patched LD for every ranking x k
    cache = {}
    for rname, rank in rankings.items():
        for k in ks:
            cache[(rname, k)] = per_row_patched_ld(
                model, sae, hook, d, rank[:k], f_clean, f_corrupt
            )

    # one set of B resample-index tensors, REUSED across rankings -> paired bootstrap
    g = torch.Generator().manual_seed(seed)
    idxs = [torch.randint(0, n, (n,), generator=g) for _ in range(B)]

    result = {"task": name, "layer": layer, "n": n, "B": B, "k": {}}
    for k in ks:
        per = {}
        for rname in rankings:
            vals = [
                bootstrap_suff(clean_ld, corrupt_ld, cache[(rname, k)], ix)
                for ix in idxs
            ]
            per[rname] = ci(vals)
        # paired gaps with CIs
        gaps = {}
        for a, b in [("attribution", "diff_mag"), ("causal", "attribution")]:
            if a in rankings and b in rankings:
                gv = [
                    bootstrap_suff(clean_ld, corrupt_ld, cache[(a, k)], ix)
                    - bootstrap_suff(clean_ld, corrupt_ld, cache[(b, k)], ix)
                    for ix in idxs
                ]
                gaps[f"{a}_minus_{b}"] = ci(gv)
        result["k"][k] = {"suff": per, "gaps": gaps}

        def fmt(c):
            return f"{c[0]:+.0%} [{c[1]:+.0%},{c[2]:+.0%}]"

        line = f"[m5/{name}] k={k:>3}  " + "  ".join(
            f"{r}={fmt(per[r])}"
            for r in ["causal", "attribution", "diff_mag"]
            if r in per
        )
        print(line)
        for gname, gc in gaps.items():
            sig = "SIGNIFICANT" if (gc[1] > 0 or gc[2] < 0) else "n.s. (CI spans 0)"
            print(f"            gap {gname}: {fmt(gc)}  -> {sig}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--ks", nargs="+", type=int, default=[16, 32, 64])
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="reports/m5_bootstrap_ci.json")
    args = ap.parse_args()
    device = pick_device(args.device)
    model = load_gemma(device)

    results = []
    results.append(
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
        )
    )
    pairs, template = ANTONYM_TASK
    results.append(
        run_task(
            model,
            device,
            "antonyms",
            pairs,
            template,
            args.layer,
            args.ks,
            args.boot,
            args.seed,
        )
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[m5] wrote {out}")


if __name__ == "__main__":
    main()
