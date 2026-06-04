"""Unit tests for nanofeatures' statistical core — no model download required.

Covers the pad-safe metric and the paired-bootstrap machinery the CI claims rest on.
"""

import torch

from nanofeatures.run_m5 import bootstrap_suff, ci
from nanofeatures.task import logit_diff


def test_logit_diff_reads_per_row_end_pos():
    # ragged answer positions: row 0 at pos 1, row 1 at pos 2; a misleading value sits
    # at the final index of row 0 to catch a fixed [-1] read.
    logits = torch.zeros(2, 3, 5)
    logits[0, 1, 0] = 4.0  # clean answer id 0 at the real end
    logits[0, 2, 0] = -99.0  # final position (must NOT be read for row 0)
    logits[1, 2, 3] = 6.0  # clean answer id 3
    ans_clean = torch.tensor([0, 3])
    ans_corrupt = torch.tensor([1, 4])
    end = torch.tensor([1, 2])
    out = logit_diff(logits, ans_clean, ans_corrupt, end)
    assert torch.allclose(out, torch.tensor([4.0, 6.0]))


def test_bootstrap_suff_endpoints():
    clean = torch.tensor([10.0, 12.0, 8.0, 11.0])
    corrupt = torch.tensor([-10.0, -8.0, -12.0, -9.0])
    idx = torch.arange(4)
    # patched == clean -> recovered fraction ~1.0
    assert abs(bootstrap_suff(clean, corrupt, clean, idx) - 1.0) < 1e-6
    # patched == corrupt -> recovered fraction ~0.0
    assert abs(bootstrap_suff(clean, corrupt, corrupt, idx) - 0.0) < 1e-6


def test_bootstrap_suff_is_paired_on_idx():
    # the SAME resample idx applied to two patched vectors must give a difference equal
    # to evaluating each on that resample — the property the paired-gap CI relies on.
    clean = torch.tensor([10.0, 12.0, 8.0, 11.0])
    corrupt = torch.tensor([-10.0, -8.0, -12.0, -9.0])
    a = torch.tensor([5.0, 6.0, 4.0, 5.5])
    b = torch.tensor([0.0, 1.0, -1.0, 0.5])
    idx = torch.tensor([0, 0, 2, 3])
    gap = bootstrap_suff(clean, corrupt, a, idx) - bootstrap_suff(
        clean, corrupt, b, idx
    )
    # recompute directly on the same resample
    ca, cz = clean[idx].mean(), corrupt[idx].mean()
    direct = ((a[idx].mean() - cz) / (ca - cz)) - ((b[idx].mean() - cz) / (ca - cz))
    assert abs(gap - direct.item()) < 1e-6


def test_ci_orientation():
    vals = [float(i) for i in range(101)]  # 0..100
    med, lo, hi = ci(vals)
    assert abs(med - 50.0) < 1e-6
    assert lo < med < hi  # lo is the 2.5pct, hi the 97.5pct — never inverted
    assert abs(lo - 2.5) < 1.0 and abs(hi - 97.5) < 1.0
