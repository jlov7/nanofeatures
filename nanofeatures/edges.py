"""Cross-layer feature EDGES — extending the boundary question from nodes to connections.

The node result (M7-M13): a cheap gradient-free score (peak |Δactivation|) ranks single
features as well as gradient attribution on single-token tasks, and loses to it only on
distributed circuits. The obvious objection to stopping there is that circuits are EDGES,
not just nodes, and EAP-style edge attribution is exactly where a gradient is meant to be
indispensable. M14 tests that.

An edge u@L1 -> d@L2 (L1<L2) factorizes into TRANSFER (how much patching u moves d) x
READOUT (how much moving d moves the metric). We define an EXACT mediated edge effect
(no gradient), then ask which cheaper score recovers it.

EXACT edge effect (gold standard, no gradient):
  edge_exact(u->d) = metric(corrupt, with d@L2 moved by exactly the per-position amount that
                    patching u@L1->clean induces in d) - metric(corrupt).
  The indirect effect of u on the metric mediated by d. Zero if u doesn't move d, or if
  moving d doesn't move the metric (both checked by the smoke).

transfer_u,d[b,p] = a_d(corrupt with u@L1 patched to clean)[b,p] - a_d(corrupt)[b,p].

Every candidate score = transfer x (some READOUT proxy). They differ ONLY in the readout
proxy, so the comparison isolates what kind of readout estimate an edge needs (the same
strawman-avoidance discipline as M7's summed-vs-peak node baselines):

  eap          : transfer . (grad . dec_d)      gradient readout (one backward)         [GRADIENT]
  cheap        : transfer . Δa_d                magnitude readout (position-resolved)   [free]
  cheap_abs    : |transfer| . |Δa_d|            sign-agnostic magnitude readout         [free]
  cheap_wdec   : (transfer . Δa_d) * ||dec_d||  decoder-norm-weighted magnitude         [free]
  cheap_node   : (Σ transfer) * node_effect_d   CAUSAL readout via node ablation        [free, no backward]
  transfer_only: Σ |transfer|                   connection strength, NO readout          [free]
  mag          : |Δa_u| * |Δa_d|                no transfer measurement at all           [free]

cheap_node is the strongest gradient-free competitor: node_effect_d is the exact behavioral
effect of moving d alone (clean->corrupt), an ablation-based readout that needs no gradient.
If it (or any free score) recovers the exact edge as well as the gradient, "edges need the
gradient" is false; if only eap and cheap_node track exact, the honest claim refines to
"edges need a CAUSAL readout estimate (gradient OR ablation); activation magnitude can't
proxy it." Either outcome is reported as-is.

All matrices are returned per-example ([b, n_u, n_d]) so the runner can paired-bootstrap
over examples, the same discipline as every other milestone.
"""

from __future__ import annotations

import torch

from .model import load_sae
from .task import logit_diff


def _row_metric(logits, d):
    """Per-example logit-difference at each row's end position (no batch mean)."""
    return logit_diff(logits, d["ans_clean"], d["ans_corrupt"], d["end"])


def _encode_at(model, sae, hook, tokens):
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook)
    return sae.encode(cache[hook])


def _encode_with_hook(model, sae, hook_read, tokens, hook_patch, patch_fn):
    """Run with `patch_fn` applied at hook_patch and grab the (downstream) activation at
    hook_read. Requires hook_patch to fire before hook_read in the forward pass (l1<l2).
    run_with_cache forwards **kwargs to the forward, not the hook machinery, so we cannot
    pass fwd_hooks to it — use run_with_hooks with an explicit grab hook instead."""
    grabbed = {}

    def grab(act, hook):
        grabbed["act"] = act
        return act

    with torch.no_grad():
        model.run_with_hooks(
            tokens, fwd_hooks=[(hook_patch, patch_fn), (hook_read, grab)]
        )
    return sae.encode(grabbed["act"])


def _node_effects(model, sae2, hook2, d, D, f_clean2, f_corr2, m_corrupt):
    """Per-example behavioral effect of moving each downstream feature d in D alone,
    clean->corrupt (the gradient-free CAUSAL readout used by cheap_node). [b, n_d]."""
    b = m_corrupt.shape[0]
    node = torch.zeros(b, len(D))
    for di, dfeat in enumerate(D.tolist()):
        delta = (f_clean2[:, :, dfeat] - f_corr2[:, :, dfeat]).unsqueeze(
            -1
        ) * sae2.W_dec[dfeat]

        def patch(act, hook, _dl=delta):
            return act + _dl

        with torch.no_grad():
            m = _row_metric(
                model.run_with_hooks(d["corrupt"], fwd_hooks=[(hook2, patch)]), d
            )
        node[:, di] = (m - m_corrupt).cpu()
    return node


def _node_effects_per_pos(model, sae2, hook2, d, D, m_corrupt, h=1.0):
    """Finite-difference directional readout of the metric w.r.t. each downstream feature d,
    PER POSITION: add a step h*W_dec[d] at position p only, measure the metric slope. This is
    the gradient-free, CAUSAL, POSITION-RESOLVED readout (the secant analog of grad.dec_d at
    each position). Cost n_d*n_pos forwards. Used by the cheap_fdpos control to show the M14
    finding is about a readout PROPERTY (causal + position-resolved), recoverable without the
    analytic gradient, not about the gradient algorithm itself. [b, n_d, n_pos]."""
    b = m_corrupt.shape[0]
    n_pos = d["corrupt"].shape[1]
    slope = torch.zeros(b, len(D), n_pos)
    for di, dfeat in enumerate(D.tolist()):
        step = sae2.W_dec[dfeat] * h
        for p in range(n_pos):

            def patch(act, hook, _p=p, _s=step):
                act = act.clone()
                act[:, _p, :] = act[:, _p, :] + _s
                return act

            with torch.no_grad():
                m = _row_metric(
                    model.run_with_hooks(d["corrupt"], fwd_hooks=[(hook2, patch)]), d
                )
            slope[:, di, p] = ((m - m_corrupt) / h).cpu()
    return slope  # [b,n_d,n_pos]


def cross_layer_edge_scores(
    model,
    d,
    l1,
    l2,
    n_u=24,
    n_d=24,
    device=None,
    return_transfer=False,
    sae_loader=load_sae,
    fdpos=False,
    fdpos_hs=(1.0,),
):
    """Per-example edge-score matrices [b, n_u, n_d] for the top-n_u upstream (L1) and
    top-n_d downstream (L2) features by |Δactivation|, all sharing the same exactly-measured
    transfer so they differ only in the readout proxy. Cost: ~n_u (transfer) + n_d (node
    effects) + n_u*n_d (exact) forward passes + 1 backward.

    `sae_loader(layer, device) -> (sae, hook_name)` defaults to Gemma Scope; pass
    `load_gpt2_sae` for the second-model edge replication.

    With return_transfer=True, also returns the full per-edge transfer tensor T[n_u,b,pos,n_d],
    the downstream decoder rows, the L2 hook, and per-row clean/corrupt metrics — everything
    M15 (`edge_circuit_sufficiency`) needs to patch a selected EDGE SET and read recovered
    behavior."""
    sae1, hook1 = sae_loader(l1, device=device)
    sae2, hook2 = sae_loader(l2, device=device)

    f_clean1 = _encode_at(model, sae1, hook1, d["clean"])
    f_corr1 = _encode_at(model, sae1, hook1, d["corrupt"])
    f_clean2 = _encode_at(model, sae2, hook2, d["clean"])
    f_corr2 = _encode_at(model, sae2, hook2, d["corrupt"])

    U = (f_clean1 - f_corr1).abs().sum(dim=(0, 1)).topk(n_u).indices
    D = (f_clean2 - f_corr2).abs().sum(dim=(0, 1)).topk(n_d).indices

    with torch.no_grad():
        m_corrupt = _row_metric(model(d["corrupt"]), d)  # [b]
        m_clean = _row_metric(model(d["clean"]), d)  # [b]
    b = m_corrupt.shape[0]
    n_pos = d["corrupt"].shape[1]
    T = torch.zeros(n_u, b, n_pos, n_d) if return_transfer else None

    # gradient of the metric w.r.t. the L2 residual (corrupt run), one backward pass
    grabbed = {}

    def grab(act, hook):
        if not act.requires_grad:
            act.requires_grad_(True)
        act.retain_grad()
        grabbed["act"] = act
        return act

    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        logits = model.run_with_hooks(d["corrupt"], fwd_hooks=[(hook2, grab)])
        _row_metric(logits, d).sum().backward()  # sum -> per-example grads intact
    grad2 = grabbed["act"].grad.detach()  # [b,pos,d_model]

    dec2_D = sae2.W_dec[D]  # [n_d, d_model]
    wnorm_D = dec2_D.norm(dim=-1).cpu()  # [n_d] decoder norms
    gw_D = grad2 @ dec2_D.T  # [b,pos,n_d]  = grad . dec_d
    dchange_D = (f_clean2 - f_corr2)[:, :, D]  # [b,pos,n_d] raw downstream change

    dmag_u = (f_clean1 - f_corr1)[:, :, U].abs().sum(dim=1)  # [b,n_u]
    dmag_d = (f_clean2 - f_corr2)[:, :, D].abs().sum(dim=1)  # [b,n_d]

    # gradient-free CAUSAL readout (node ablation), independent of u
    node_eff = _node_effects(
        model, sae2, hook2, d, D, f_clean2, f_corr2, m_corrupt
    )  # [b,n_d]
    # optional gradient-free CAUSAL + POSITION-RESOLVED readout (finite difference), one slope
    # tensor per probe step h (the h-grid checks the +0.99 recovery isn't an artifact of h=1).
    fd_specs = []  # (score_name, slope_pos[b,n_d,n_pos])
    if fdpos:
        hs = list(fdpos_hs)
        for h in hs:
            nm = "cheap_fdpos" if (len(hs) == 1 and h == 1.0) else f"cheap_fdpos@{h:g}"
            fd_specs.append(
                (nm, _node_effects_per_pos(model, sae2, hook2, d, D, m_corrupt, h=h))
            )

    names = [
        "eap",
        "cheap",
        "cheap_abs",
        "cheap_wdec",
        "cheap_node",
        "transfer_only",
        "mag",
    ]
    names.extend(nm for nm, _ in fd_specs)
    out = {nm: torch.zeros(b, n_u, n_d) for nm in names}
    out["exact"] = torch.zeros(b, n_u, n_d)
    pert_sq = 0.0  # accumulate mean per-example ||delta2|| for the eap-vs-exact tautology check
    pert_cnt = 0

    for ui, u in enumerate(U.tolist()):
        # patch u@L1 -> clean in the corrupt run; read the induced change in every d@L2
        delta1 = (f_clean1[:, :, u] - f_corr1[:, :, u]).unsqueeze(-1) * sae1.W_dec[u]

        def patch1(act, hook, _dl=delta1):
            return act + _dl

        f2_after = _encode_with_hook(model, sae2, hook2, d["corrupt"], hook1, patch1)
        transfer = (f2_after[:, :, D] - f_corr2[:, :, D]).cpu()  # [b,pos,n_d]
        if T is not None:
            T[ui] = transfer
        t_sum = transfer.sum(dim=1)  # [b,n_d] signed
        t_abs = transfer.abs().sum(dim=1)  # [b,n_d]

        out["eap"][:, ui, :] = (transfer * gw_D.cpu()).sum(dim=1)
        out["cheap"][:, ui, :] = (transfer * dchange_D.cpu()).sum(dim=1)
        out["cheap_abs"][:, ui, :] = (transfer.abs() * dchange_D.abs().cpu()).sum(dim=1)
        out["cheap_wdec"][:, ui, :] = (transfer * dchange_D.cpu()).sum(dim=1) * wnorm_D[
            None, :
        ]
        out["cheap_node"][:, ui, :] = t_sum * node_eff
        out["transfer_only"][:, ui, :] = t_abs
        for nm, slope_pos in fd_specs:
            # transfer [b,pos,n_d] . slope_pos [b,n_d,pos] -> sum over pos
            out[nm][:, ui, :] = (transfer * slope_pos.permute(0, 2, 1)).sum(dim=1)

        # exact: move each d by exactly its per-position transfer, measure metric change
        transfer_dev = transfer.to(dec2_D.device)
        for di, dfeat in enumerate(D.tolist()):
            delta2 = transfer_dev[:, :, di].unsqueeze(-1) * sae2.W_dec[dfeat]
            pert_sq += delta2.norm(dim=(1, 2)).mean().item()
            pert_cnt += 1

            def patch2(act, hook, _dl=delta2):
                return act + _dl

            with torch.no_grad():
                m = _row_metric(
                    model.run_with_hooks(d["corrupt"], fwd_hooks=[(hook2, patch2)]), d
                )
            out["exact"][:, ui, di] = (m - m_corrupt).cpu()

    out["mag"] = (dmag_u[:, :, None] * dmag_d[:, None, :]).cpu()
    res = {
        **{k: v for k, v in out.items()},
        "U": U.tolist(),
        "D": D.tolist(),
        "m_corrupt_mean": m_corrupt.mean().item(),
        "pert_norm_mean": pert_sq / max(pert_cnt, 1),
        "gradfree": [
            "cheap",
            "cheap_abs",
            "cheap_wdec",
            "cheap_node",
            "transfer_only",
            "mag",
        ],
    }
    if return_transfer:
        res["transfer"] = T  # [n_u,b,pos,n_d]
        res["dec2_D"] = dec2_D.detach().cpu()  # [n_d,d_model]
        res["hook2"] = hook2
        res["m_clean_row"] = m_clean.detach().cpu()
        res["m_corrupt_row"] = m_corrupt.detach().cpu()
    return res


def edge_circuit_sufficiency(model, d, r, ms, methods, seed=0, B=5000):
    """M15: BEHAVIORAL faithfulness of an EDGE circuit. Rank all n_u*n_d edges by each score
    (mean over examples), take the top-m, and patch ONLY those edges: for each downstream d,
    move it by the SUM of the measured transfers from its selected upstream parents, apply all
    such moves at L2 at once, and read the recovered logit-difference. This is the edge analog
    of node sufficiency, and the behavioral check the rank-correlation result (M14) is a proxy
    for. Caveat (disclosed): transfers are measured one upstream feature at a time, so patching
    several at once ignores higher-order interactions among parents — a sufficiency proxy.

    `r` is the dict from cross_layer_edge_scores(..., return_transfer=True). Returns, per m and
    method, the paired-bootstrap sufficiency CI over examples. Includes a `random` floor."""
    T = r["transfer"].to(model.cfg.device)  # [n_u,b,pos,n_d]
    dec2 = r["dec2_D"].to(model.cfg.device)  # [n_d,d_model]
    hook2 = r["hook2"]
    n_u, b, _, n_d = T.shape
    clean_ld = r["m_clean_row"]
    corrupt_ld = r["m_corrupt_row"]

    def suff_rows(edge_list):
        # combined L2 delta from the selected edges, then one patched forward
        move = torch.zeros(b, T.shape[2], dec2.shape[1], device=model.cfg.device)
        for ui, di in edge_list:
            move += T[ui, :, :, di].unsqueeze(-1) * dec2[di]

        def patch(act, hook, _m=move):
            return act + _m

        with torch.no_grad():
            logits = model.run_with_hooks(d["corrupt"], fwd_hooks=[(hook2, patch)])
        return _row_metric(logits, d).cpu()

    rg = torch.Generator().manual_seed(seed + 11)
    order_rand = torch.randperm(n_u * n_d, generator=rg).tolist()

    ranked = {}
    for name in methods:
        flat = r[name].mean(0).flatten()  # [n_u*n_d]
        ranked[name] = flat.argsort(descending=True, stable=True).tolist()
    ranked["random"] = order_rand

    g = torch.Generator().manual_seed(seed)
    idxs = [torch.randint(0, b, (b,), generator=g) for _ in range(B)]

    def boot_suff(patched_ld):
        out = []
        for ix in idxs:
            c = clean_ld[ix].mean()
            z = corrupt_ld[ix].mean()
            p = patched_ld[ix].mean()
            out.append(((p - z) / (c - z + 1e-9)).item())
        return out

    cells = {}
    for m in ms:
        per = {}
        boots = {}
        for name in [*methods, "random"]:
            edges = [(e // n_d, e % n_d) for e in ranked[name][:m]]
            pl = suff_rows(edges)
            boots[name] = boot_suff(pl)
            t = torch.tensor(boots[name])
            per[name] = (
                t.median().item(),
                torch.quantile(t, 0.025).item(),
                torch.quantile(t, 0.975).item(),
            )
        # paired gap: gradient-selected vs the strongest gradient-free-selected circuit
        gf = [x for x in methods if x not in ("eap", "exact")]
        strongest_gf = max(gf, key=lambda s: per[s][0]) if gf else None
        gap = None
        if strongest_gf is not None:
            gv = [a - b_ for a, b_ in zip(boots["eap"], boots[strongest_gf])]
            gt = torch.tensor(gv)
            gap = (
                gt.median().item(),
                torch.quantile(gt, 0.025).item(),
                torch.quantile(gt, 0.975).item(),
            )
        cells[m] = {
            "suff": per,
            "strongest_gradfree": strongest_gf,
            "eap_minus_strongest_gradfree": gap,
            "eap_beats_strongest_gradfree": (gap[1] > 0) if gap else None,
        }
    return cells
