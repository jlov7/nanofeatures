"""Feature-level circuit discovery on Gemma-2-2B via Gemma Scope SAEs.

Method (exact, no gradient approximation for the first result):
- Aligned clean/corrupt factual pairs (single-token country AND capital => identical
  prompt length => position-aligned patching, well-defined).
- For a layer, feature-patch each active SAE feature from clean->corrupt and measure
  how much it recovers the clean metric. This is the EXACT causal effect of that
  feature (denoising). Circuit = top-k by effect.
- Validate by FAITHFULNESS (sufficiency): patch the top-k features together; measure
  recovered metric fraction. Compare causal selection vs activation-magnitude vs
  random baselines at matched k. The honest question: does causal attribution give a
  more faithful circuit per feature than a trivial baseline?
"""

from __future__ import annotations

import torch

from .model import load_sae
from .task import PAIRS, TEMPLATE, logit_diff

# single-token first names (verified in Gemma's vocab) for the IOI distributed task
IOI_NAMES = [
    "John",
    "Mary",
    "Tom",
    "James",
    "Robert",
    "Michael",
    "David",
    "Susan",
    "Karen",
    "Linda",
    "Paul",
    "Mark",
    "Anna",
    "Kevin",
    "Brian",
    "Steven",
    "Sarah",
    "Laura",
    "Peter",
    "Alice",
    "George",
    "Helen",
    "Frank",
    "Grace",
    "Henry",
    "Julia",
    "Oscar",
    "Emma",
    "Jack",
    "Lucy",
    "Sam",
    "Kate",
    "Carl",
    "Nina",
    "Ben",
    "Rose",
]
IOI_TEMPLATE = "When {a} and {b} went to the store, {b} gave a drink to"


def ioi_pairs(model, names=IOI_NAMES, seed: int = 0):
    """Indirect-Object-Identification pairs — a DISTRIBUTED, multi-position circuit (the
    regime single-token contrastive tasks cannot probe). Clean prompt: '{IO} and {S} ...,
    {S} gave a drink to' -> answer = IO (the name mentioned once). The subject S appears
    TWICE, so clean and corrupt differ at THREE name-token positions and the known IOI
    circuit (duplicate-token, S-inhibition, name-mover heads) involves path cancellation.
    metric = logit(IO) - logit(S). Corrupt = a disjoint name pair (rotate by 1), so the
    IO/S tokens are absent and the IO-preference signal is removed (denoising baseline).
    Returns the same dict shape as aligned_pairs so feature_rankings/bootstrap reuse it."""
    # keep only names that are a single token IN THIS MODEL's tokenizer (so prompts stay
    # uniform-length); this makes IOI model-agnostic (all 36 are single-token for Gemma).
    names = [
        s for s in names if model.to_tokens(" " + s, prepend_bos=False).shape[1] == 1
    ]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(names), generator=g).tolist()
    names = [names[i] for i in perm]
    pairs = [(names[2 * i], names[2 * i + 1]) for i in range(len(names) // 2)]
    n = len(pairs)

    def tok(s):
        return int(model.to_tokens(" " + s, prepend_bos=False)[0, 0])

    clean_p = [IOI_TEMPLATE.format(a=a, b=b) for a, b in pairs]
    # corrupt = the NEXT pair's prompt (disjoint names) -> IO/S tokens absent
    corrupt_p = [
        IOI_TEMPLATE.format(a=pairs[(i + 1) % n][0], b=pairs[(i + 1) % n][1])
        for i in range(n)
    ]
    clean = model.to_tokens(clean_p)
    corrupt = model.to_tokens(corrupt_p)
    lens = {model.to_tokens(p).shape[1] for p in clean_p + corrupt_p}
    assert len(lens) == 1, (
        f"variable IOI prompt lengths {sorted(lens)} -> unsafe uniform end"
    )
    assert clean.shape == corrupt.shape
    ans_clean = torch.tensor([tok(a) for a, _ in pairs], device=clean.device)  # IO
    ans_corrupt = torch.tensor([tok(b) for _, b in pairs], device=clean.device)  # S
    end = torch.full((n,), clean.shape[1] - 1, device=clean.device)
    return {
        "clean": clean,
        "corrupt": corrupt,
        "ans_clean": ans_clean,
        "ans_corrupt": ans_corrupt,
        "end": end,
        "n": n,
        "countries": [a for a, _ in pairs],
    }


def aligned_pairs(model, pairs=PAIRS, template=TEMPLATE, seed: int = 0):
    """Pairs where BOTH subject and answer are single tokens -> identical prompt
    length -> aligned positions for patching. Returns batches + answer ids + metric fn.
    Defaults to the capitals task; pass (pairs, template) for a different relation."""
    ok = []
    for country, capital in pairs:
        ct = model.to_tokens(" " + capital, prepend_bos=False)
        # subject sits mid-prompt (no leading space after 'of '); check single-token
        qt = model.to_tokens(" " + country, prepend_bos=False)
        if ct.shape[1] == 1 and qt.shape[1] == 1:
            ok.append((country, capital, int(ct[0, 0])))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ok), generator=g).tolist()
    ok = [ok[i] for i in perm]
    n = len(ok)
    # pass both keys so a template may use {country} (capitals) or {subject} (generic)
    clean_p = [template.format(country=c, subject=c) for c, _, _ in ok]
    corrupt_p = [
        template.format(country=ok[(i + 1) % n][0], subject=ok[(i + 1) % n][0])
        for i in range(n)
    ]
    clean = model.to_tokens(clean_p)
    corrupt = model.to_tokens(corrupt_p)
    assert clean.shape == corrupt.shape, (
        f"prompts not aligned: {clean.shape} vs {corrupt.shape}"
    )
    # `end` below is a single uniform index, so EVERY row must be the same real length
    # (no right-padding) or logits[:, -1] would read PAD for short rows — the exact
    # nanocircuits IOI bug. Single-token subject+answer should guarantee this; assert it.
    lens = {model.to_tokens(p).shape[1] for p in clean_p + corrupt_p}
    assert len(lens) == 1, (
        f"variable prompt lengths {sorted(lens)} -> uniform `end` is unsafe; use per-row "
        f"end positions (see task.make_pairs) before reusing aligned_pairs for this task"
    )
    ans_clean = torch.tensor([t for _, _, t in ok], device=clean.device)
    ans_corrupt = torch.tensor(
        [ok[(i + 1) % n][2] for i in range(n)], device=clean.device
    )
    end = torch.full((n,), clean.shape[1] - 1, device=clean.device)
    return {
        "clean": clean,
        "corrupt": corrupt,
        "ans_clean": ans_clean,
        "ans_corrupt": ans_corrupt,
        "end": end,
        "n": n,
        "countries": [c for c, _, _ in ok],
    }


def aligned_pairs_multi(model, pairs, template, seed: int = 0):
    """Like aligned_pairs but for MULTI-token subjects: keep pairs whose answer is a single
    token and whose subject tokenizes to the modal length L (so the contrastive signal
    spans L positions and all prompts stay uniform-length). L=1 reduces to aligned_pairs.
    Used to place tasks at intermediate points on the distributedness axis."""
    cand = []
    for subj, ans in pairs:
        at = model.to_tokens(" " + ans, prepend_bos=False)
        st = model.to_tokens(" " + subj, prepend_bos=False)
        if at.shape[1] == 1:
            cand.append((subj, ans, int(at[0, 0]), st.shape[1]))
    # modal subject token-length, then keep only that length (uniform prompts)
    lens = [c[3] for c in cand]
    target = max(set(lens), key=lens.count)
    ok = [(s, a, t) for s, a, t, ln in cand if ln == target]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ok), generator=g).tolist()
    ok = [ok[i] for i in perm]
    n = len(ok)
    clean_p = [template.format(subject=s) for s, _, _ in ok]
    corrupt_p = [template.format(subject=ok[(i + 1) % n][0]) for i in range(n)]
    clean = model.to_tokens(clean_p)
    corrupt = model.to_tokens(corrupt_p)
    plens = {model.to_tokens(p).shape[1] for p in clean_p + corrupt_p}
    assert len(plens) == 1, (
        f"non-uniform prompt lengths {sorted(plens)} (subj_len={target})"
    )
    assert clean.shape == corrupt.shape
    ans_clean = torch.tensor([t for _, _, t in ok], device=clean.device)
    ans_corrupt = torch.tensor(
        [ok[(i + 1) % n][2] for i in range(n)], device=clean.device
    )
    end = torch.full((n,), clean.shape[1] - 1, device=clean.device)
    return {
        "clean": clean,
        "corrupt": corrupt,
        "ans_clean": ans_clean,
        "ans_corrupt": ans_corrupt,
        "end": end,
        "n": n,
        "subj_len": target,
        "countries": [s for s, _, _ in ok],
    }


def _metric(logits, d):
    return logit_diff(logits, d["ans_clean"], d["ans_corrupt"], d["end"]).mean()


def positional_distributedness(model, d, layer):
    """How DISTRIBUTED across token positions is the task's causal signal? Method-neutral
    (no SAE, no ranking): for each non-BOS position p, patch the FULL residual at p from
    clean->corrupt at this layer and measure metric recovery r_p. Distributedness = the
    participation ratio of the positive recoveries, PR = (sum r_p)^2 / sum r_p^2 = the
    effective number of positions carrying the signal (1 for single-token recall, ~3 for
    IOI). This is the x-axis against which we test the attribution-vs-cheap gap; being
    ranking-free it cannot be circular with that gap. Returns (PR, recovery_per_position)."""
    hook = f"blocks.{layer}.hook_resid_post"
    with torch.no_grad():
        m_clean = _metric(model(d["clean"]), d).item()
        m_corrupt = _metric(model(d["corrupt"]), d).item()
        _, cache = model.run_with_cache(d["clean"], names_filter=hook)
        clean_resid = cache[hook]
    denom = (m_clean - m_corrupt) + 1e-9
    rec = []
    for p in range(1, d["clean"].shape[1]):  # skip BOS at position 0

        def patch(act, hook, _p=p, _r=clean_resid):
            act = act.clone()
            act[:, _p, :] = _r[:, _p, :]
            return act

        with torch.no_grad():
            m_p = _metric(
                model.run_with_hooks(d["corrupt"], fwd_hooks=[(hook, patch)]), d
            ).item()
        rec.append((m_p - m_corrupt) / denom)
    r = torch.tensor(rec).clamp(
        min=0.0
    )  # only positions that help recover the behavior
    pr = (r.sum() ** 2 / (r.pow(2).sum() + 1e-12)).item() if r.sum() > 0 else 0.0
    return pr, rec


def layer_scan(model, d, layers):
    """For each layer, patch the FULL clean residual into the corrupt run at that
    layer's SAE hook and measure metric recovery. Picks where the signal lives."""
    with torch.no_grad():
        m_clean = _metric(model(d["clean"]), d).item()
        m_corrupt = _metric(model(d["corrupt"]), d).item()
    out = {}
    for L in layers:
        _sae, hook = load_sae(L, device=model.cfg.device)
        with torch.no_grad():
            _, cache = model.run_with_cache(d["clean"], names_filter=hook)
            clean_resid = cache[hook]

            def patch(act, hook, _r=clean_resid):
                return _r

            m_patched = _metric(
                model.run_with_hooks(d["corrupt"], fwd_hooks=[(hook, patch)]), d
            ).item()
        recovered = (m_patched - m_corrupt) / (m_clean - m_corrupt + 1e-9)
        out[L] = recovered
    return out, m_clean, m_corrupt


def feature_effects(model, sae, hook, d):
    """Exact causal effect of patching each active SAE feature clean->corrupt.
    Returns (effects[d_sae], f_clean_mag[d_sae], active_idx)."""
    with torch.no_grad():
        _, cc = model.run_with_cache(d["clean"], names_filter=hook)
        _, ccor = model.run_with_cache(d["corrupt"], names_filter=hook)
        f_clean = sae.encode(cc[hook])  # [b,pos,d_sae]
        f_corrupt = sae.encode(ccor[hook])
        m_corrupt = _metric(model(d["corrupt"]), d).item()

    # candidate features: active on the clean run anywhere
    active = (f_clean.abs().sum(dim=(0, 1)) > 0).nonzero().flatten()
    W_dec = sae.W_dec  # [d_sae, d_model]
    effects = torch.zeros(sae.cfg.d_sae, device=model.cfg.device)
    # delta to add to corrupt resid to set feature i to clean: (f_clean-f_corrupt)*W_dec[i]
    fdiff = f_clean - f_corrupt  # [b,pos,d_sae]
    for i in active.tolist():
        delta = fdiff[:, :, i].unsqueeze(-1) * W_dec[i]  # [b,pos,d_model]

        def patch(act, hook, _d=delta):
            return act + _d

        with torch.no_grad():
            m_patched = _metric(
                model.run_with_hooks(d["corrupt"], fwd_hooks=[(hook, patch)]), d
            ).item()
            effects[i] = m_patched - m_corrupt
    f_clean_mag = f_clean.abs().sum(dim=(0, 1))
    return effects, f_clean_mag, active


def feature_rankings(model, sae, hook, d, exact: bool = True):
    """Compute four candidate feature rankings on the SAME activations so the
    sufficiency comparison is apples-to-apples:

      causal      : EXACT effect of patching each active feature clean->corrupt
                    (7205 forward passes — the expensive ground-truth method)
      attribution : gradient linear approx of that effect, ONE backward pass
                    attr_i = sum_{b,pos} (f_clean-f_corrupt)_i * (W_dec[i] . d metric/d resid)
      diff_mag    : |f_clean - f_corrupt| summed (cheap 'features that change most')
      magnitude   : |f_clean| summed (the strawman; picks BOS/positional features)

    Returns dict of ranking-name -> descending index list, plus (f_clean, f_corrupt,
    active, m_corrupt). The honest question M2 answers: does EXACT patching buy a
    more faithful circuit than the cheap ATTRIBUTION approximation people actually use?
    """
    with torch.no_grad():
        _, cc = model.run_with_cache(d["clean"], names_filter=hook)
        _, ccor = model.run_with_cache(d["corrupt"], names_filter=hook)
        f_clean = sae.encode(cc[hook])
        f_corrupt = sae.encode(ccor[hook])
        m_corrupt = _metric(model(d["corrupt"]), d).item()

    active = (f_clean.abs().sum(dim=(0, 1)) > 0).nonzero().flatten()
    W_dec = sae.W_dec  # [d_sae, d_model]
    fdiff = f_clean - f_corrupt  # [b,pos,d_sae]

    # --- gradient of the metric w.r.t. the residual at the hook, on the corrupt run ---
    grabbed = {}

    def grab(act, hook):
        # act is a non-leaf requiring grad (Gemma params require grad), so DON'T call
        # requires_grad_ on it (that errors on non-leaves) — retain_grad reads its grad.
        if not act.requires_grad:
            act.requires_grad_(True)
        act.retain_grad()
        grabbed["act"] = act
        return act

    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        logits = model.run_with_hooks(d["corrupt"], fwd_hooks=[(hook, grab)])
        _metric(logits, d).backward()
    grad = grabbed["act"].grad.detach()  # [b,pos,d_model]

    # per-feature attribution: fdiff_i * (grad . W_dec[i]), summed over batch/pos
    g_w = grad @ W_dec.T  # [b,pos,d_sae]  element i = grad . W_dec[i]
    attribution = (fdiff.detach() * g_w).sum(dim=(0, 1))  # [d_sae]

    # --- exact causal effect (expensive: one forward pass PER active feature) ---
    fdiff_d = fdiff.detach()
    effects = torch.zeros(sae.cfg.d_sae, device=model.cfg.device)
    if exact:
        for i in active.tolist():
            delta = fdiff_d[:, :, i].unsqueeze(-1) * W_dec[i]

            def patch(act, hook, _d=delta):
                return act + _d

            with torch.no_grad():
                m_patched = _metric(
                    model.run_with_hooks(d["corrupt"], fwd_hooks=[(hook, patch)]), d
                ).item()
            effects[i] = m_patched - m_corrupt

    diff_mag = fdiff_d.abs().sum(dim=(0, 1))
    mag = f_clean.abs().sum(dim=(0, 1))
    # STRONGER cheap baselines (the strawman-avoidance the reviewer demanded): rank by
    # |f_clean-f_corrupt| restricted to the position(s) that actually DIFFER between
    # clean and corrupt (the subject token), to the answer position, and by position-max
    # instead of position-sum. If attribution still beats these, the causal advantage is
    # not an artifact of diluting all-position diff-magnitude with identical positions.
    diffpos = (d["clean"] != d["corrupt"]).any(0)  # [pos] bool: subject token position
    diff_mag_subjpos = (fdiff_d.abs() * diffpos.view(1, -1, 1)).sum(dim=(0, 1))
    end_pos = d["clean"].shape[1] - 1
    diff_mag_lastpos = fdiff_d[:, end_pos, :].abs().sum(dim=0)
    diff_mag_max = fdiff_d.abs().amax(dim=(0, 1))
    # Decoder-norm-weighted peak change: the exact patch effect is LINEAR in
    # fdiff_i * W_dec[i], so a feature's residual-space impact scales with ||W_dec[i]||.
    # diff_mag_max ignores that norm; this gradient-free baseline restores it and is the
    # most principled cheap competitor to attribution (audit W1). If attribution does NOT
    # beat THIS, attribution's edge is the decoder norm, not the gradient.
    w_norm = W_dec.norm(dim=-1)  # [d_sae]
    diff_mag_max_wdec = diff_mag_max * w_norm
    rankings = {
        "attribution": attribution.argsort(descending=True, stable=True).tolist(),
        "diff_mag": diff_mag.argsort(descending=True, stable=True).tolist(),
        "diff_mag_subjpos": diff_mag_subjpos.argsort(
            descending=True, stable=True
        ).tolist(),
        "diff_mag_lastpos": diff_mag_lastpos.argsort(
            descending=True, stable=True
        ).tolist(),
        "diff_mag_max": diff_mag_max.argsort(descending=True, stable=True).tolist(),
        "diff_mag_max_wdec": diff_mag_max_wdec.argsort(
            descending=True, stable=True
        ).tolist(),
        "magnitude": mag.argsort(descending=True, stable=True).tolist(),
    }
    # only expose the exact ranking when it was actually computed
    if exact:
        rankings = {
            "causal": effects.argsort(descending=True, stable=True).tolist(),
            **rankings,
        }
    return (
        rankings,
        f_clean.detach(),
        f_corrupt.detach(),
        active,
        m_corrupt,
        effects,
        attribution,
    )
