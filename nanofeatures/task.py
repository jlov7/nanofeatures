"""Factual-recall task: clean/corrupt prompt pairs with single-token answers, and a
logit-difference metric. The corrupt prompt is a different fact, so attribution
patching clean->corrupt isolates the features carrying the *subject-specific* signal.
"""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer

# (country, capital). Capitals filtered to single Gemma tokens at runtime.
PAIRS = [
    ("France", "Paris"),
    ("Japan", "Tokyo"),
    ("Italy", "Rome"),
    ("Germany", "Berlin"),
    ("Spain", "Madrid"),
    ("Russia", "Moscow"),
    ("China", "Beijing"),
    ("Egypt", "Cairo"),
    ("Greece", "Athens"),
    ("Cuba", "Havana"),
    ("Peru", "Lima"),
    ("Iran", "Tehran"),
    ("Austria", "Vienna"),
    ("Poland", "Warsaw"),
    ("Norway", "Oslo"),
    ("Sweden", "Stockholm"),
    ("Portugal", "Lisbon"),
    ("Ireland", "Dublin"),
    ("Kenya", "Nairobi"),
    ("Chile", "Santiago"),
]

# "...is the city of" cues a city name as the literal next token (cleaner/legible),
# while the contrastive metric (correct vs wrong capital) isolates the factual signal.
TEMPLATE = "The capital of {country} is the city of"


def single_token_pairs(model: HookedTransformer):
    """Keep pairs whose capital is a single token (with leading space)."""
    out = []
    for country, capital in PAIRS:
        tok = model.to_tokens(" " + capital, prepend_bos=False)
        if tok.shape[1] == 1:
            out.append((country, capital, int(tok[0, 0])))
    return out


def make_pairs(model: HookedTransformer, seed: int = 0):
    """Build clean/corrupt batches. Each clean fact is paired with a different
    fact (rotated by 1) as its corrupt counterpart. Returns dict of tensors."""
    pairs = single_token_pairs(model)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(pairs), generator=g).tolist()
    pairs = [pairs[i] for i in perm]
    n = len(pairs)

    clean_prompts, corrupt_prompts, ans_clean, ans_corrupt = [], [], [], []
    for i in range(n):
        c_country, _c_cap, c_tok = pairs[i]
        d_country, _d_cap, d_tok = pairs[(i + 1) % n]  # corrupt = next pair
        clean_prompts.append(TEMPLATE.format(country=c_country))
        corrupt_prompts.append(TEMPLATE.format(country=d_country))
        ans_clean.append(c_tok)
        ans_corrupt.append(d_tok)

    clean = model.to_tokens(clean_prompts)
    corrupt = model.to_tokens(corrupt_prompts)
    # prompts vary in length -> track per-row last real position (pad-safe, the
    # lesson from nanocircuits' IOI bug).
    end_clean = torch.tensor([model.to_tokens(p).shape[1] - 1 for p in clean_prompts])
    end_corrupt = torch.tensor(
        [model.to_tokens(p).shape[1] - 1 for p in corrupt_prompts]
    )
    return {
        "clean": clean,
        "corrupt": corrupt,
        "ans_clean": torch.tensor(ans_clean, device=clean.device),
        "ans_corrupt": torch.tensor(ans_corrupt, device=clean.device),
        "end_clean": end_clean.to(clean.device),
        "end_corrupt": end_corrupt.to(clean.device),
        "countries": [p[0] for p in pairs],
        "capitals": [p[1] for p in pairs],
    }


def logit_diff(logits, ans_clean, ans_corrupt, end_pos):
    """metric = logit(clean answer) - logit(corrupt answer) at each row's end position.
    High on the clean run (model recalls the right capital); low/negative on corrupt."""
    b = torch.arange(logits.shape[0], device=logits.device)
    last = logits[b, end_pos, :]
    return last[b, ans_clean] - last[b, ans_corrupt]
