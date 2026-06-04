"""M13 (the deepest open critique): does the cheap≈attribution boundary survive
INTERACTION-AWARE selection? Every other milestone ranks features independently and takes
the top-k. An adversarial reviewer's strongest objection: that greedy single-feature ranking
is exactly where a gradient's estimate of feature INTERACTIONS cannot help, so the
single-token tie may be an artifact of the selection rule, not of the gradient.

So we add the interaction-aware gold standard: a GREEDY-EXACT circuit that, at each step,
adds the feature most increasing JOINT sufficiency (one real forward pass per candidate).
This accounts for how features combine. We compare its sufficiency curve against top-k
attribution and top-k diff_mag_max, with paired-bootstrap CIs.

Reading the outcomes (all honest):
  - greedy ≈ attribution ≈ cheap        -> features are ~additive here; interactions don't
                                            matter; the boundary holds for joint selection
                                            too (the critique is answered).
  - greedy >> both, attribution closer  -> attribution's value shows up under interactions;
                                            the single-token headline is refined.
  - greedy >> both, attribution ≈ cheap -> interactions matter but the gradient doesn't help
                                            capture them; an even stronger negative.

Candidate pool = top-64 attribution UNION top-64 diff_mag_max, so neither method is
disadvantaged. Bounded cost: |pool| x k forward passes per task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .circuit import aligned_pairs, feature_rankings, ioi_pairs
from .model import load_gemma, load_sae, pick_device
from .run_m5 import bootstrap_suff, ci, per_row_ld, per_row_patched_ld
from .tasks import TASK_SUITE


def _mean_suff(clean_ld, corrupt_ld, patched_ld):
    c, z, p = clean_ld.mean(), corrupt_ld.mean(), patched_ld.mean()
    return ((p - z) / (c - z + 1e-9)).item()


def greedy_exact(
    model, sae, hook, d, f_clean, f_corrupt, candidates, ks, clean_ld, corrupt_ld
):
    """Greedily add the candidate that most increases JOINT mean sufficiency. Returns the
    selected order and a dict k -> per-row patched LD for the first-k selected set."""
    remaining = list(candidates)
    selected: list[int] = []
    per_row_at_k = {}
    kmax = max(ks)
    for step in range(kmax):
        best_c, best_suff, best_rows = None, float("-inf"), None
        for c in remaining:
            rows = per_row_patched_ld(
                model, sae, hook, d, selected + [c], f_clean, f_corrupt
            )
            s = _mean_suff(clean_ld, corrupt_ld, rows)
            if s > best_suff:
                best_c, best_suff, best_rows = c, s, rows
        selected.append(best_c)
        remaining.remove(best_c)
        if (step + 1) in ks:
            per_row_at_k[step + 1] = best_rows
    return selected, per_row_at_k


def run_task(model, device, name, d, layer, ks, B, seed, pool_each):
    sae, hook = load_sae(layer, device=device)
    out = feature_rankings(model, sae, hook, d, exact=False)
    rankings, f_clean, f_corrupt = out[0], out[1], out[2]
    clean_ld = per_row_ld(model, d, "clean")
    corrupt_ld = per_row_ld(model, d, "corrupt")
    n = d["n"]

    # neutral candidate pool: top-`pool_each` of attribution UNION top-`pool_each` of cheap
    pool = list(
        dict.fromkeys(
            rankings["attribution"][:pool_each] + rankings["diff_mag_max"][:pool_each]
        )
    )
    print(
        f"\n[m13/{name}] n={n} layer={layer} |pool|={len(pool)} (greedy is {len(pool)}x{max(ks)} fwd)"
    )

    _selected, greedy_rows = greedy_exact(
        model, sae, hook, d, f_clean, f_corrupt, pool, ks, clean_ld, corrupt_ld
    )

    # cache per-row patched LD for the top-k sets of attribution and cheap
    cache = {("greedy", k): greedy_rows[k] for k in ks}
    for meth in ("attribution", "diff_mag_max"):
        for k in ks:
            cache[(meth, k)] = per_row_patched_ld(
                model, sae, hook, d, rankings[meth][:k], f_clean, f_corrupt
            )

    g = torch.Generator().manual_seed(seed)
    idxs = [torch.randint(0, n, (n,), generator=g) for _ in range(B)]

    def boot(meth, k):
        return [
            bootstrap_suff(clean_ld, corrupt_ld, cache[(meth, k)], ix) for ix in idxs
        ]

    result = {"task": name, "layer": layer, "n": n, "pool": len(pool), "k": {}}
    for k in ks:
        gb, ab, cb = boot("greedy", k), boot("attribution", k), boot("diff_mag_max", k)
        per = {"greedy": ci(gb), "attribution": ci(ab), "diff_mag_max": ci(cb)}
        gaps = {
            "greedy_minus_attr": ci([x - y for x, y in zip(gb, ab)]),
            "greedy_minus_cheap": ci([x - y for x, y in zip(gb, cb)]),
            "attr_minus_cheap": ci([x - y for x, y in zip(ab, cb)]),
        }
        result["k"][k] = {"suff": per, "gaps": gaps}

        def f(c):
            return f"{c[0]:+.0%}[{c[1]:+.0%},{c[2]:+.0%}]"

        print(
            f"[m13/{name}] k={k:>2} greedy={f(per['greedy'])} attr={f(per['attribution'])} "
            f"cheap={f(per['diff_mag_max'])}"
        )
        gma, gmc = gaps["greedy_minus_attr"], gaps["greedy_minus_cheap"]
        print(
            f"          greedy-attr={f(gma)} ({'SIG' if gma[1] > 0 else 'n.s.'})  "
            f"greedy-cheap={f(gmc)} ({'SIG' if gmc[1] > 0 else 'n.s.'})  "
            f"attr-cheap={f(gaps['attr_minus_cheap'])}"
        )
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tasks",
        nargs="+",
        default=["capitals", "antonyms", "ioi"],
        help="any TASK_SUITE key, or 'ioi'",
    )
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--ks", nargs="+", type=int, default=[8, 16, 32])
    ap.add_argument("--pool-each", type=int, default=64)
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="reports/m13_interactions.json")
    args = ap.parse_args()
    device = pick_device(args.device)
    model = load_gemma(device)

    results = []
    for name in args.tasks:
        d = (
            ioi_pairs(model)
            if name == "ioi"
            else aligned_pairs(model, *TASK_SUITE[name])
        )
        results.append(
            run_task(
                model,
                device,
                name,
                d,
                args.layer,
                args.ks,
                args.boot,
                args.seed,
                args.pool_each,
            )
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[m13] wrote {out}")

    # verdict: across single-token tasks, does greedy beat the top-k methods, and does
    # attribution track greedy better than cheap does?
    single = [r for r in results if r["task"] != "ioi"]
    g_beats_attr = g_beats_cheap = cells = 0
    for r in single:
        for kd in r["k"].values():
            cells += 1
            if kd["gaps"]["greedy_minus_attr"][1] > 0:
                g_beats_attr += 1
            if kd["gaps"]["greedy_minus_cheap"][1] > 0:
                g_beats_cheap += 1
    print(
        f"[m13] single-token: interaction-aware greedy beats top-k attribution at "
        f"{g_beats_attr}/{cells} cells, beats top-k cheap at {g_beats_cheap}/{cells} cells"
    )
    if cells and g_beats_attr <= cells // 4 and g_beats_cheap <= cells // 4:
        print(
            "[m13] VERDICT: interactions barely matter on single-token tasks (features ~additive); "
            "the cheap≈attribution boundary HOLDS for joint selection, not just top-k. C2 answered."
        )
    else:
        print(
            "[m13] VERDICT: interaction-aware selection changes the picture; see per-cell "
            "greedy-attr vs greedy-cheap gaps for whether the gradient captures the interactions."
        )


if __name__ == "__main__":
    main()
