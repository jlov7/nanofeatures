"""Diagnostic: WHERE does bf16 destroy the cross-layer edge signal?

The committed claim said edges collapse in bf16 via "catastrophic cancellation because the edge
effects are tiny, differenced from O(1-10) logit-diffs." An adversarial review pointed out this
cannot explain IOI: IOI's metric baseline is ~0.6, so its bf16 ulp is ~0.002 and the edge effect
(~0.05) sits ~19x above it, yet IOI collapses as hard as capitals (whose baseline ~15 does put
the effect at the bf16 noise floor). So the cancellation must live somewhere OTHER than the final
metric difference.

This recomputes the three load-bearing intermediates of cross_layer_edge_scores in fp32 and bf16
on the SAME device (CPU, the reliable backend -- this also de-confounds the MPS-vs-dtype question:
if bf16 collapses on CPU too, it is a dtype effect, not an MPS kernel bug) using the SAME
fp32-selected feature sets U,D, and reports the relative error of each:

  transfer  : f2_after[D] - f_corr2[D]  (per-edge transfer, a difference of two ~equal SAE acts)
  grad2     : d metric / d resid at L2   (the bf16 backward)
  exact     : the per-edge exact mediated effect (the gold standard the scores recover)

plus the metric-scale / ulp / effect-size SNR for each task. The intermediate with the largest
relative error is the true cancellation site. Not committed-as-result; produces the evidence that
the docs' mechanism sentence is then rewritten to match.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .circuit import aligned_pairs, ioi_pairs
from .edges import _encode_at, _encode_with_hook, _row_metric
from .model import load_gemma, load_sae
from .tasks import TASK_SUITE

PAIRS = TASK_SUITE["capitals"][0] if "capitals" in TASK_SUITE else None


def _relerr(a_fp32, b_bf16):
    """Relative L2 error ||bf16 - fp32|| / ||fp32|| with both in fp32 for the comparison."""
    a = a_fp32.float().flatten()
    b = b_bf16.float().flatten()
    denom = a.norm().item()
    return (b - a).norm().item() / denom if denom > 0 else float("nan")


def _intermediates(model, sae1, hook1, sae2, hook2, d, U, D):
    """Compute (transfer for u=U[0], grad2, exact-effect-vector for u=U[0]) at the given U,D."""
    f_corr2 = _encode_at(model, sae2, hook2, d["corrupt"])
    with torch.no_grad():
        m_corrupt = _row_metric(model(d["corrupt"]), d)
    f_clean1 = _encode_at(model, sae1, hook1, d["clean"])
    f_corr1 = _encode_at(model, sae1, hook1, d["corrupt"])

    # gradient w.r.t. L2 residual (corrupt run)
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
        _row_metric(logits, d).sum().backward()
    grad2 = grabbed["act"].grad.detach()  # [b,pos,d_model]

    u = U[0].item()
    delta1 = (f_clean1[:, :, u] - f_corr1[:, :, u]).unsqueeze(-1) * sae1.W_dec[u]

    def patch1(act, hook, _dl=delta1):
        return act + _dl

    f2_after = _encode_with_hook(model, sae2, hook2, d["corrupt"], hook1, patch1)
    transfer = (f2_after[:, :, D] - f_corr2[:, :, D]).cpu()  # [b,pos,n_d]

    # exact per-edge effect for u=U[0]: move each d by its per-position transfer, measure metric
    transfer_dev = transfer.to(sae2.W_dec.device)
    exact = torch.zeros(len(D))
    for di, dfeat in enumerate(D.tolist()):
        delta2 = transfer_dev[:, :, di].unsqueeze(-1) * sae2.W_dec[dfeat]

        def patch2(act, hook, _d=delta2):
            return act + _d

        with torch.no_grad():
            m_patched = _row_metric(
                model.run_with_hooks(d["corrupt"], fwd_hooks=[(hook2, patch2)]), d
            )
        exact[di] = (m_patched - m_corrupt).mean().cpu()
    return {
        "transfer": transfer,
        "grad2": grad2.cpu(),
        "exact": exact,
        "m_corrupt_absmean": m_corrupt.abs().mean().item(),
        "exact_absmax": exact.abs().max().item(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1", type=int, default=5)
    ap.add_argument("--l2", type=int, default=7)
    ap.add_argument("--n-u", type=int, default=24)
    ap.add_argument("--n-d", type=int, default=24)
    ap.add_argument("--tasks", nargs="+", default=["capitals", "ioi"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="reports/m18_edge_precision_diag.json")
    args = ap.parse_args()

    results = []
    for task in args.tasks:
        # fp32 reference model (also used to fix the token set + U,D selection)
        m32 = load_gemma(args.device, dtype=torch.float32)
        s1_32, h1 = load_sae(args.l1, device=args.device)
        s2_32, h2 = load_sae(args.l2, device=args.device)
        s1_32, s2_32 = s1_32.to(torch.float32), s2_32.to(torch.float32)
        d = ioi_pairs(m32) if task == "ioi" else aligned_pairs(m32, *TASK_SUITE[task])

        # select U,D in fp32 (shared across both dtypes so we isolate numeric error, not selection)
        fc1 = _encode_at(m32, s1_32, h1, d["clean"])
        fk1 = _encode_at(m32, s1_32, h1, d["corrupt"])
        fc2 = _encode_at(m32, s2_32, h2, d["clean"])
        fk2 = _encode_at(m32, s2_32, h2, d["corrupt"])
        U = (fc1 - fk1).abs().sum(dim=(0, 1)).topk(args.n_u).indices
        D = (fc2 - fk2).abs().sum(dim=(0, 1)).topk(args.n_d).indices

        ref = _intermediates(m32, s1_32, h1, s2_32, h2, d, U, D)
        del m32, s1_32, s2_32

        # bf16 on the SAME device
        m16 = load_gemma(args.device, dtype=torch.bfloat16)
        s1_16, _ = load_sae(args.l1, device=args.device)
        s2_16, _ = load_sae(args.l2, device=args.device)
        s1_16, s2_16 = s1_16.to(torch.bfloat16), s2_16.to(torch.bfloat16)
        test = _intermediates(m16, s1_16, h1, s2_16, h2, d, U, D)
        del m16, s1_16, s2_16

        # spearman of the exact-effect vector (the readout target) fp32 vs bf16
        def spearman(a, b):
            ar = a.argsort().argsort().float()
            br = b.argsort().argsort().float()
            ar = ar - ar.mean()
            br = br - br.mean()
            return (ar @ br / (ar.norm() * br.norm() + 1e-9)).item()

        ulp = ref["m_corrupt_absmean"] * 2**-8
        row = {
            "task": task,
            "l1": args.l1,
            "l2": args.l2,
            "metric_absmean": ref["m_corrupt_absmean"],
            "bf16_ulp_at_metric": ulp,
            "exact_effect_absmax": ref["exact_absmax"],
            "metric_snr": ref["exact_absmax"] / ulp if ulp > 0 else float("nan"),
            "relerr_transfer": _relerr(ref["transfer"], test["transfer"]),
            "relerr_grad2": _relerr(ref["grad2"], test["grad2"]),
            "relerr_exact_effect": _relerr(ref["exact"], test["exact"]),
            "spearman_exact_fp32_vs_bf16": spearman(ref["exact"], test["exact"]),
        }
        results.append(row)
        print(
            f"[{task:>9} L{args.l1}->{args.l2}] metric|mean|={row['metric_absmean']:.3f} "
            f"ulp={ulp:.4f} effect_absmax={row['exact_effect_absmax']:.4f} "
            f"SNR={row['metric_snr']:.1f}\n"
            f"            relerr: transfer={row['relerr_transfer']:.3f} "
            f"grad2={row['relerr_grad2']:.3f} exact_effect={row['relerr_exact_effect']:.3f} "
            f"| spearman(exact fp32 vs bf16)={row['spearman_exact_fp32_vs_bf16']:+.2f}",
            flush=True,
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n[diag] wrote {args.out}")
    print(
        "[diag] the intermediate with the largest relerr is the true cancellation site; "
        "if relerr is large on CPU bf16, the collapse is a DTYPE effect (not an MPS artifact)."
    )


if __name__ == "__main__":
    main()
