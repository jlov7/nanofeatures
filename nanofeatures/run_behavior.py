"""Behavior gate: does Gemma-2-2B actually do the factual-recall task cleanly?
Everything downstream (feature attribution) is meaningless if it doesn't. Gate:
clean logit-diff strongly positive, corrupt negative, high top-1 accuracy.
"""

from __future__ import annotations

import torch

from .model import load_gemma, pick_device
from .task import logit_diff, make_pairs


def main() -> None:
    device = pick_device()
    model = load_gemma(device)
    d = make_pairs(model)
    n = d["clean"].shape[0]
    with torch.no_grad():
        clean_logits = model(d["clean"])
        corrupt_logits = model(d["corrupt"])

    ld_clean = logit_diff(
        clean_logits, d["ans_clean"], d["ans_corrupt"], d["end_clean"]
    )
    ld_corrupt = logit_diff(
        corrupt_logits, d["ans_clean"], d["ans_corrupt"], d["end_corrupt"]
    )

    # top-1 accuracy: is the clean answer the argmax at the end position?
    b = torch.arange(n, device=clean_logits.device)
    pred = clean_logits[b, d["end_clean"], :].argmax(-1)
    acc = (pred == d["ans_clean"]).float().mean().item()

    print(f"[behavior] n_pairs={n} (single-token capitals)")
    print(
        f"[behavior] clean logit-diff  mean={ld_clean.mean():+.3f}  (frac>0: {(ld_clean > 0).float().mean():.2f})"
    )
    print(f"[behavior] corrupt logit-diff mean={ld_corrupt.mean():+.3f}")
    print(f"[behavior] top-1 accuracy (clean answer = argmax): {acc:.2f}")
    ok = ld_clean.mean() > 2.0 and acc >= 0.7
    print(
        f"[behavior] GATE: {'PASS' if ok else 'WEAK - reconsider task/pairs'} "
        f"(need clean LD>2 and acc>=0.7 for attribution to be meaningful)"
    )


if __name__ == "__main__":
    main()
