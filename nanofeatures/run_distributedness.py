"""M10: is the attribution-over-cheap-baseline advantage a PREDICTABLE function of how
DISTRIBUTED the task's causal signal is? M7/M8 gave two points (single-token tasks: tie;
IOI: attribution wins). This turns the binary boundary into a quantitative law: for every
task, measure distributedness (participation ratio of per-position causal recovery, a
ranking-free property of the task) and the attribution - strongest-cheap-baseline
sufficiency gap, then correlate them (Spearman rho, task-bootstrap CI).

If gap rises monotonically with measured distributedness, you can PREDICT in advance, from
a cheap property of the task, whether the expensive gradient is worth it. Tasks span the
axis: 7 single-token tasks (signal at ~1 position) < capitals_2tok (2-token subjects,
~2 positions) < IOI (~3 positions, with path cancellation).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .circuit import (
    aligned_pairs,
    aligned_pairs_multi,
    ioi_pairs,
    positional_distributedness,
)
from .model import load_gemma, load_sae, pick_device
from .run_m5 import per_row_ld
from .run_suite import eval_cell, gate
from .tasks import (
    CAPITALS_2TOK,
    CAPITALS_2TOK_TEMPLATE,
    CITY_COUNTRY_2TOK,
    CITY_COUNTRY_2TOK_TEMPLATE,
    PERSON_COUNTRY_2TOK,
    PERSON_COUNTRY_2TOK_TEMPLATE,
    TASK_SUITE,
)

SINGLE = [
    "capitals",
    "antonyms",
    "country_language",
    "past_tense",
    "comparative",
    "plural",
    "successor",
]


def _builders(model):
    """name -> d dict, spanning the distributedness axis."""
    out = {}
    for name in SINGLE:
        pairs, template = TASK_SUITE[name]
        out[name] = aligned_pairs(model, pairs=pairs, template=template)
    out["capitals_2tok"] = aligned_pairs_multi(
        model, CAPITALS_2TOK, CAPITALS_2TOK_TEMPLATE
    )
    out["city_country_2tok"] = aligned_pairs_multi(
        model, CITY_COUNTRY_2TOK, CITY_COUNTRY_2TOK_TEMPLATE
    )
    out["person_country_2tok"] = aligned_pairs_multi(
        model, PERSON_COUNTRY_2TOK, PERSON_COUNTRY_2TOK_TEMPLATE
    )
    out["ioi"] = ioi_pairs(model)
    return out


def _rankdata(x):
    """average-rank of a list (ties share the mean rank), for Spearman."""
    n = len(x)
    order = sorted(range(n), key=lambda i: x[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a, b):
    ta, tb = torch.tensor(a), torch.tensor(b)
    ta, tb = ta - ta.mean(), tb - tb.mean()
    d = (ta.norm() * tb.norm()).item()
    return (ta @ tb).item() / d if d > 0 else 0.0


def _spearman(x, y):
    return _pearson(_rankdata(x), _rankdata(y))


def _mannwhitney_exact(single, multi):
    """Exact one-sided Mann-Whitney: P(multi-group ranks at least this high | null), by
    enumerating all C(N, n_multi) rank assignments (N small). Tests 'multi tasks have
    larger gaps than single-token tasks' without assuming a smooth PR relationship."""
    from itertools import combinations

    vals = single + multi
    ranks = _rankdata(vals)
    obs = sum(ranks[len(single) + j] for j in range(len(multi)))
    n, nb = len(vals), len(multi)
    cnt = tot = 0
    for comb in combinations(range(n), nb):
        tot += 1
        if sum(ranks[i] for i in comb) >= obs - 1e-9:
            cnt += 1
    return obs, cnt / tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--rho-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="reports/m10_distributedness.json")
    args = ap.parse_args()
    device = pick_device(args.device)
    model = load_gemma(device)
    sae, hook = load_sae(args.layer, device=device)
    L = args.layer

    rows = []
    for name, d in _builders(model).items():
        # min_n=6 here (not 8): the multi-token-subject tasks are small by construction;
        # their small-n gap CIs are wide and disclosed, and the correlation weights them
        # the same as the larger tasks (no special pleading).
        ok, info = gate(model, d, min_n=6, min_sep=1.0)
        if not ok:
            print(f"[m10] {name}: SKIP ({info.get('reason')})")
            continue
        pr, _rec = positional_distributedness(model, d, L)
        clean_ld = per_row_ld(model, d, "clean")
        corrupt_ld = per_row_ld(model, d, "corrupt")
        cells = eval_cell(
            model,
            sae,
            hook,
            d,
            [args.k],
            args.boot,
            args.seed,
            False,
            clean_ld,
            corrupt_ld,
        )
        cell = cells[args.k]
        gap = cell["attr_minus_strongest"]
        rows.append(
            {
                "task": name,
                "n": d["n"],
                "distributedness_PR": pr,
                "gap_med": gap[0],
                "gap_lo": gap[1],
                "gap_hi": gap[2],
                "strongest_cheap": cell["strongest_cheap"],
                "attr_suff": cell["suff"]["attribution"][0],
            }
        )
        print(
            f"[m10] {name:18s} n={d['n']:>2} PR={pr:4.2f}  "
            f"gap={gap[0]:+.0%} [{gap[1]:+.0%},{gap[2]:+.0%}]  (vs {cell['strongest_cheap']})"
        )

    prs = [r["distributedness_PR"] for r in rows]
    gaps = [r["gap_med"] for r in rows]
    rho = _spearman(prs, gaps)
    pear = _pearson(prs, gaps)

    # task-bootstrap CI on Spearman rho (resample tasks with replacement)
    g = torch.Generator().manual_seed(args.seed)
    m = len(rows)
    boot_rhos = []
    for _ in range(args.rho_boot):
        idx = torch.randint(0, m, (m,), generator=g).tolist()
        xs = [prs[i] for i in idx]
        ys = [gaps[i] for i in idx]
        if len(set(xs)) > 1 and len(set(ys)) > 1:
            boot_rhos.append(_spearman(xs, ys))
    bt = torch.tensor(boot_rhos)
    rho_lo = torch.quantile(bt, 0.025).item()
    rho_hi = torch.quantile(bt, 0.975).item()

    # The cleaner, primary statistic: do MULTI-position tasks (signal spans >1 token:
    # 2-token subjects + IOI) have larger gaps than SINGLE-position tasks? This needs no
    # smooth PR relationship — just the structural single-vs-multi split known a priori.
    single_gaps = [r["gap_med"] for r in rows if r["task"] in SINGLE]
    multi_gaps = [r["gap_med"] for r in rows if r["task"] not in SINGLE]
    mw_u, mw_p = _mannwhitney_exact(single_gaps, multi_gaps)
    perfect = max(single_gaps) < min(multi_gaps)

    result = {
        "layer": L,
        "k": args.k,
        "rows": rows,
        "single_vs_multi_mannwhitney_u": mw_u,
        "single_vs_multi_p_onesided": mw_p,
        "perfect_separation": perfect,
        "spearman_rho": rho,
        "spearman_rho_ci": [rho_lo, rho_hi],
        "pearson_r": pear,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(
        f"\n[m10] single-position (n={len(single_gaps)}) gaps "
        f"[{min(single_gaps):+.0%},{max(single_gaps):+.0%}]  vs  multi-position "
        f"(n={len(multi_gaps)}) gaps [{min(multi_gaps):+.0%},{max(multi_gaps):+.0%}]"
    )
    print(
        f"[m10] group separation: Mann-Whitney one-sided p={mw_p:.4f}"
        + ("  (PERFECT: every multi > every single)" if perfect else "")
    )
    print(
        f"[m10] corroborating: Spearman rho(PR, gap) = {rho:+.2f} "
        f"[{rho_lo:+.2f}, {rho_hi:+.2f}]  (Pearson {pear:+.2f}); wrote {out}"
    )
    if mw_p < 0.05:
        print(
            "[m10] VERDICT: attribution's advantage is governed by whether the contrastive "
            "signal is SINGLE- or MULTI-position. Single-position tasks tie the cheap "
            "baseline; multi-position tasks (incl. IOI) show attribution winning. The "
            "continuous distributedness score corroborates but is the weaker statement."
        )
    else:
        print("[m10] VERDICT: no significant single-vs-multi separation.")


if __name__ == "__main__":
    main()
