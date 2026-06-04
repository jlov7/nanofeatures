"""M22: does the edge readout law hold for a genuine MULTI-LAYER (2-hop) feature path?

The single-hop result (M14) showed: recovering the exact mediated effect of an edge u@L1->d@L2
needs a readout that is both causal and position-resolved (gradient eap, or gradient-free
cheap_fdpos; magnitude and node-ablation fail). That is one layer pair. Real feature circuits are
multi-hop paths. This extends the exact-mediated-effect machinery to a connected 2-hop chain
u@L1 -> m@L2 -> d@L3 and asks two things:

  1. Does the readout 2x2 still hold when the transfer is itself a 2-hop composition? (generality)
  2. Does eap ~= exact (AtP at the edge level) survive composition, or does the linearization
     degrade with path length? (a "depth" axis: compare the 2-hop second-order residual to 1-hop)

Exact 2-hop path effect (gradient-free gold standard, sequential patching): patch u->clean, read
the induced change in m; patch m by exactly that change, read the induced change in d; move d by
exactly that change, measure the metric. Path score = <2-hop-composed transfer, readout_d>; all
scores share the SAME exact composed transfer and differ only in readout_d, so this isolates the
readout exactly as the single-hop ladder did. fp32 only (M18: edge effects are fp32-fragile).

Self-consistency smoke check baked in: eap (the linearization) must recover exact at high rho and
magnitude must fail; if the composition were wired wrong, eap would NOT track exact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .circuit import aligned_pairs, ioi_pairs
from .edges import (
    _encode_at,
    _encode_with_hook,
    _node_effects,
    _node_effects_per_pos,
    _row_metric,
)
from .model import load_gemma, load_sae, pick_device
from .tasks import TASK_SUITE


def _rankdata(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def _spearman(a, b):
    ra, rb = torch.tensor(_rankdata(a)), torch.tensor(_rankdata(b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = (ra.norm() * rb.norm()).item()
    return (ra @ rb).item() / den if den > 0 else 0.0


def _ci(vals, lo=2.5, hi=97.5):
    t = torch.tensor(vals)
    return (
        t.median().item(),
        torch.quantile(t, lo / 100).item(),
        torch.quantile(t, hi / 100).item(),
    )


def two_hop(model, d, l1, l2, l3, n_u, n_m, n_d, device):
    sae1, h1 = load_sae(l1, device=device)
    sae2, h2 = load_sae(l2, device=device)
    sae3, h3 = load_sae(l3, device=device)
    for s in (sae1, sae2, sae3):
        s.to(torch.float32)
    fc1, fk1 = (
        _encode_at(model, sae1, h1, d["clean"]),
        _encode_at(model, sae1, h1, d["corrupt"]),
    )
    fc2, fk2 = (
        _encode_at(model, sae2, h2, d["clean"]),
        _encode_at(model, sae2, h2, d["corrupt"]),
    )
    fc3, fk3 = (
        _encode_at(model, sae3, h3, d["clean"]),
        _encode_at(model, sae3, h3, d["corrupt"]),
    )
    U = (fc1 - fk1).abs().sum(dim=(0, 1)).topk(n_u).indices
    M = (fc2 - fk2).abs().sum(dim=(0, 1)).topk(n_m).indices
    D = (fc3 - fk3).abs().sum(dim=(0, 1)).topk(n_d).indices

    with torch.no_grad():
        m_corrupt = _row_metric(model(d["corrupt"]), d)  # [b]
    b = m_corrupt.shape[0]

    # readout factors at the FINAL node d (L3): same 2x2 axes as single-hop
    grabbed = {}

    def grab(act, hook):
        if not act.requires_grad:
            act.requires_grad_(True)
        act.retain_grad()
        grabbed["act"] = act
        return act

    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        logits = model.run_with_hooks(d["corrupt"], fwd_hooks=[(h3, grab)])
        _row_metric(logits, d).sum().backward()
    grad3 = grabbed["act"].grad.detach()
    dec3_D = sae3.W_dec[D]
    gw_D = (
        grad3 @ dec3_D.T
    ).cpu()  # [b,pos,n_d]  causal + position-resolved (gradient)
    dchange_D = (fc3 - fk3)[:, :, D].cpu()  # position-resolved, NOT causal (magnitude)
    node_eff = _node_effects(
        model, sae3, h3, d, D, fc3, fk3, m_corrupt
    )  # [b,n_d] causal not pos
    slope_D = _node_effects_per_pos(
        model, sae3, h3, d, D, m_corrupt
    )  # [b,n_d,pos] causal+pos free

    names = ["eap", "cheap_fdpos", "cheap_mag", "cheap_node"]
    scores = {nm: torch.zeros(b, n_u, n_m, n_d) for nm in names}
    exact = torch.zeros(b, n_u, n_m, n_d)
    pert_sq = 0.0
    pert_cnt = 0

    for ui, u in enumerate(U.tolist()):
        delta1 = (fc1[:, :, u] - fk1[:, :, u]).unsqueeze(-1) * sae1.W_dec[u]

        def patch1(act, hook, _d=delta1):
            return act + _d

        f2_after = _encode_with_hook(model, sae2, h2, d["corrupt"], h1, patch1)
        dM = f2_after[:, :, M] - fk2[:, :, M]  # [b,pos,n_m] hop-1 transfer u->each m

        for mi, mfeat in enumerate(M.tolist()):
            delta2 = (
                dM[:, :, mi].unsqueeze(-1) * sae2.W_dec[mfeat]
            )  # move m by u-induced amount

            def patch2(act, hook, _d=delta2):
                return act + _d

            f3_after = _encode_with_hook(model, sae3, h3, d["corrupt"], h2, patch2)
            T2 = (
                f3_after[:, :, D] - fk3[:, :, D]
            ).cpu()  # [b,pos,n_d] composed 2-hop transfer
            scores["eap"][:, ui, mi, :] = (T2 * gw_D).sum(1)
            scores["cheap_mag"][:, ui, mi, :] = (T2 * dchange_D).sum(1)
            scores["cheap_fdpos"][:, ui, mi, :] = (T2 * slope_D.permute(0, 2, 1)).sum(1)
            scores["cheap_node"][:, ui, mi, :] = T2.sum(1) * node_eff

            T2_dev = T2.to(dec3_D.device)
            for di, dfeat in enumerate(D.tolist()):
                delta3 = T2_dev[:, :, di].unsqueeze(-1) * sae3.W_dec[dfeat]
                pert_sq += delta3.norm(dim=(1, 2)).mean().item()
                pert_cnt += 1

                def patch3(act, hook, _d=delta3):
                    return act + _d

                with torch.no_grad():
                    m_p = _row_metric(
                        model.run_with_hooks(d["corrupt"], fwd_hooks=[(h3, patch3)]), d
                    )
                exact[:, ui, mi, di] = (m_p - m_corrupt).cpu()

    return {
        "scores": scores,
        "exact": exact,
        "names": names,
        "n": b,
        "pert_norm_mean": pert_sq / max(pert_cnt, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", nargs=3, type=int, default=[5, 7, 9])
    ap.add_argument("--tasks", nargs="+", default=["capitals", "ioi"])
    ap.add_argument("--n-u", type=int, default=6)
    ap.add_argument("--n-m", type=int, default=6)
    ap.add_argument("--n-d", type=int, default=6)
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="reports/m22_multihop.json")
    args = ap.parse_args()
    device = pick_device(args.device)
    model = load_gemma(
        device, dtype=torch.float32
    )  # fp32: edges are precision-fragile (M18)
    l1, l2, l3 = args.chain

    results = []
    for task in args.tasks:
        d = (
            ioi_pairs(model)
            if task == "ioi"
            else aligned_pairs(model, *TASK_SUITE[task])
        )
        r = two_hop(model, d, l1, l2, l3, args.n_u, args.n_m, args.n_d, device)
        n = r["n"]
        ex = r["exact"]  # [b,nu,nm,nd]
        g = torch.Generator().manual_seed(args.seed)
        idxs = [torch.randint(0, n, (n,), generator=g) for _ in range(args.boot)]

        per = {}
        for nm in r["names"]:
            sc = r["scores"][nm]
            vals = []
            for ix in idxs:
                a = sc[ix].mean(0).flatten().tolist()
                e = ex[ix].mean(0).flatten().tolist()
                vals.append(_spearman(a, e))
            per[nm] = _ci(vals)
        # eap-vs-exact second-order residual (the AtP-at-2-hop check / depth degradation signal)
        em = ex.mean(0).flatten()
        am = r["scores"]["eap"].mean(0).flatten()
        # best linear fit eap->exact; residual fraction = 1 - R^2-like
        denom = (em - em.mean()).pow(2).sum().item()
        resid = (em - am * (em @ am / (am @ am + 1e-9))).pow(2).sum().item()
        rel_resid = resid / (denom + 1e-9)

        def fmt(c):
            return f"{c[0]:+.2f} [{c[1]:+.2f},{c[2]:+.2f}]"

        print(
            f"\n[m22/{task}] chain {l1}->{l2}->{l3}  n={n}  "
            f"top {args.n_u}x{args.n_m}x{args.n_d} paths",
            flush=True,
        )
        for nm in r["names"]:
            print(
                f"            {nm:12s} rho vs exact 2-hop = {fmt(per[nm])}", flush=True
            )
        print(
            f"            eap~exact rel 2nd-order residual = {rel_resid:.3f}  "
            f"mean||Δ|| = {r['pert_norm_mean']:.3f}",
            flush=True,
        )
        results.append(
            {
                "task": task,
                "chain": [l1, l2, l3],
                "n": n,
                "rho_vs_exact": {nm: per[nm] for nm in r["names"]},
                "eap_vs_exact_rel_residual": rel_resid,
                "pert_norm_mean": r["pert_norm_mean"],
            }
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n[m22] wrote {args.out}")
    print(
        "[m22] readout law holds at 2-hop if eap & cheap_fdpos recover exact (high rho) while "
        "cheap_mag (~0) and cheap_node (<=0) fail - same 2x2, now over a multi-layer chain."
    )


if __name__ == "__main__":
    main()
