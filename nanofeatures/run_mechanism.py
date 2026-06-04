"""M12 (mechanism): WHY does the cheap peak-change baseline fail on IOI? The claim is that
diff_mag_max is POSITION-BLIND — it ranks a feature by its largest per-position change at
ANY position, so on a distributed circuit it picks features whose biggest change sits at a
position the readout does not use (e.g. the duplicated-subject token), while gradient
attribution weights features by their effect on the END-position logit and so picks features
at causally-relevant positions.

We test this directly. For each token position we already have its causal relevance r_p =
how much patching the full residual at p (clean->corrupt) recovers the metric
(positional_distributedness). For each feature we find its PEAK change position. Then for a
method's top-k features we compute the mean causal relevance of those peak positions. If
attribution's selected features sit at higher-r_p positions than diff_mag_max's, the
position-blindness explanation is confirmed — not asserted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


from .circuit import feature_rankings, ioi_pairs, positional_distributedness
from .model import load_gemma, load_sae, pick_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="reports/m12_mechanism.json")
    args = ap.parse_args()
    device = pick_device(args.device)
    model = load_gemma(device)
    sae, hook = load_sae(args.layer, device=device)

    d = ioi_pairs(model)
    out = feature_rankings(model, sae, hook, d, exact=False)
    rankings, f_clean, f_corrupt = out[0], out[1], out[2]

    # per-feature peak change position (mean over batch of |f_clean - f_corrupt|)
    fdiff = (f_clean - f_corrupt).abs().mean(0)  # [pos, d_sae]
    peak_pos = fdiff.argmax(
        dim=0
    )  # [d_sae] -> position index of each feature's peak change

    # causal relevance per position: r_p from full-residual per-position patching
    pr, rec = positional_distributedness(model, d, args.layer)
    seqlen = d["clean"].shape[1]

    def relevance_of_pos(p):
        # rec[i] is recovery of position i+1 (BOS at 0 excluded); clamp negatives to 0
        if p <= 0 or p - 1 >= len(rec):
            return 0.0
        return max(0.0, rec[p - 1])

    # decode the IOI token positions for human-readable context
    toks = model.to_str_tokens(d["clean"][0])

    def summarize(method):
        feats = rankings[method][: args.k]
        positions = [int(peak_pos[i]) for i in feats]
        rels = [relevance_of_pos(p) for p in positions]
        # histogram of how many of the top-k peak at each position
        hist = {}
        for p in positions:
            hist[p] = hist.get(p, 0) + 1
        return {
            "mean_position_relevance": sum(rels) / len(rels),
            "frac_at_high_relevance": sum(r > 0.1 for r in rels) / len(rels),
            "position_histogram": dict(sorted(hist.items())),
        }

    attr = summarize("attribution")
    dmm = summarize("diff_mag_max")

    # the most causally relevant positions (for context)
    top_positions = sorted(
        range(1, seqlen), key=lambda p: relevance_of_pos(p), reverse=True
    )[:4]
    pos_relevance = {p: round(relevance_of_pos(p), 3) for p in range(1, seqlen)}

    result = {
        "layer": args.layer,
        "k": args.k,
        "n": d["n"],
        "distributedness_PR": pr,
        "str_tokens": toks,
        "position_causal_relevance": pos_relevance,
        "most_relevant_positions": top_positions,
        "attribution": attr,
        "diff_mag_max": dmm,
        "attr_minus_dmm_position_relevance": attr["mean_position_relevance"]
        - dmm["mean_position_relevance"],
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(result, indent=2))

    print(f"[m12] IOI layer {args.layer} k={args.k}  (PR={pr:.2f})")
    print(f"[m12] tokens: {toks}")
    print(
        f"[m12] most causally-relevant positions (by full-residual recovery): {top_positions}"
    )
    print(
        f"[m12] attribution top-{args.k}: mean position-relevance="
        f"{attr['mean_position_relevance']:.3f}, "
        f"{attr['frac_at_high_relevance']:.0%} at high-relevance positions"
    )
    print(
        f"[m12] diff_mag_max top-{args.k}: mean position-relevance="
        f"{dmm['mean_position_relevance']:.3f}, "
        f"{dmm['frac_at_high_relevance']:.0%} at high-relevance positions"
    )
    print(f"[m12] attribution peak-position histogram: {attr['position_histogram']}")
    print(f"[m12] diff_mag_max peak-position histogram: {dmm['position_histogram']}")
    delta = result["attr_minus_dmm_position_relevance"]
    print(f"[m12] wrote {outp}")
    if delta > 0.05:
        print(
            "[m12] VERDICT: attribution selects features at MORE causally-relevant positions "
            f"than diff_mag_max (+{delta:.2f}). Position-blindness of the cheap baseline is "
            "the mechanism — confirmed, not asserted."
        )
    else:
        print(
            f"[m12] VERDICT: position-relevance gap is small ({delta:+.2f}); the "
            "position-blindness story is not clearly supported — investigate further."
        )


if __name__ == "__main__":
    main()
