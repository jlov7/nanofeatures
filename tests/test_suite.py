"""Unit tests for the M7 task suite + gate — no model download required.

Covers the suite's structural integrity (so a malformed task can't silently skew the
aggregate) and the behavior gate's decision logic on synthetic logit-difference vectors.
"""

import torch

from nanofeatures import tasks
from nanofeatures.circuit import IOI_NAMES, IOI_TEMPLATE
from nanofeatures.run_suite import BASELINES, gate


def test_spearman_monotone():
    from nanofeatures.run_distributedness import _spearman

    assert abs(_spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-9
    assert abs(_spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-9


def test_mannwhitney_perfect_separation():
    from math import comb

    from nanofeatures.run_distributedness import _mannwhitney_exact

    # every 'multi' value exceeds every 'single' value -> max U, exact p = 1 / C(N, n_multi)
    single, multi = [1.0, 2.0, 3.0], [4.0, 5.0]
    u, p = _mannwhitney_exact(single, multi)
    assert u == 9.0  # rank-sum of the top two ranks (4 + 5)
    assert abs(p - 1.0 / comb(5, 2)) < 1e-9
    # overlapping groups -> not significant
    _, p2 = _mannwhitney_exact([1.0, 4.0], [2.0, 3.0])
    assert p2 > 0.05


def test_interactions_mean_suff():
    # greedy uses mean sufficiency = (mean_patched - mean_corrupt)/(mean_clean - mean_corrupt).
    from nanofeatures.run_interactions import _mean_suff

    clean = torch.tensor([10.0, 12.0, 8.0])
    corrupt = torch.tensor([-10.0, -8.0, -12.0])
    assert abs(_mean_suff(clean, corrupt, clean) - 1.0) < 1e-6  # patched==clean -> 1
    assert (
        abs(_mean_suff(clean, corrupt, corrupt) - 0.0) < 1e-6
    )  # patched==corrupt -> 0


def test_edges_spearman_vs_exact_helper():
    # M14's paired bootstrap ranks each edge method against the exact mediated effect by
    # averaging the per-example [b,n_u,n_d] matrix over resampled rows, flattening, and
    # rank-correlating. A method identical to exact must recover it perfectly; its negation
    # must anti-correlate. If the mean/flatten/pairing wiring drifted, these would break.
    from nanofeatures.run_edges import _spearman_vs_exact

    g = torch.Generator().manual_seed(0)
    exact = torch.randn(5, 4, 4, generator=g)
    idx = torch.arange(5)
    assert _spearman_vs_exact(exact.clone(), exact, idx) > 0.999
    assert _spearman_vs_exact(-exact, exact, idx) < -0.999


def test_neuron_basis_is_identity_sae():
    # the NeuronBasis shim must be a true identity SAE: encode(x)==x and W_dec==I, so
    # feature_rankings ranks/patches raw residual dimensions. If encode or W_dec drifted
    # from identity, the "neuron basis" control would silently measure something else.
    from nanofeatures.model import NeuronBasis

    nb = NeuronBasis(d_model=5, hook_name="blocks.7.hook_resid_post", device="cpu")
    x = torch.randn(2, 3, 5)
    assert torch.equal(nb.encode(x), x)
    assert torch.equal(nb.W_dec, torch.eye(5))
    assert nb.cfg.d_sae == 5
    assert nb.cfg.metadata.hook_name == "blocks.7.hook_resid_post"


def test_ioi_template_has_duplicate_subject():
    # the IOI structure REQUIRES the subject {b} to appear twice and the indirect
    # object {a} once -> the answer is {a}. If {b} were not duplicated this would not be
    # IOI and the distributed-circuit claim would be invalid.
    assert IOI_TEMPLATE.count("{b}") == 2, "subject must be mentioned twice (IOI)"
    assert IOI_TEMPLATE.count("{a}") == 1, "indirect object mentioned once"
    assert len(IOI_NAMES) == len(set(IOI_NAMES)), "duplicate names would collide"
    assert len(IOI_NAMES) >= 16, "need enough names for a reasonable n"


def test_task_suite_well_formed():
    # every task is a (pairs, template) with a {subject} slot and distinct single-word
    # answers (a duplicate answer would create degenerate zero-signal contrastive pairs).
    assert len(tasks.TASK_SUITE) >= 6
    for name, (pairs, template) in tasks.TASK_SUITE.items():
        assert "{subject}" in template, f"{name}: template missing {{subject}} slot"
        assert len(pairs) >= 8, f"{name}: too few candidate pairs"
        answers = [a for _, a in pairs]
        assert len(set(answers)) == len(answers), f"{name}: duplicate answer tokens"
        subjects = [s for s, _ in pairs]
        assert len(set(subjects)) == len(subjects), f"{name}: duplicate subjects"


def test_backcompat_aliases_present():
    # run_m5/run_m6 import these names; the suite refactor must preserve them.
    assert tasks.ANTONYM_TASK[0] is tasks.ANTONYMS
    assert tasks.PAIRS is tasks.CAPITALS
    assert tasks.TEMPLATE == tasks.CAPITALS_TEMPLATE


def test_baselines_exclude_gradient_methods():
    # the cheap-baseline set must never include the methods under test, or the
    # "strongest cheap baseline" comparison would be circular.
    assert "attribution" not in BASELINES
    assert "causal" not in BASELINES
    assert "diff_mag_max" in BASELINES  # the strong gradient-free competitor


class _FakeModel:
    """Minimal stand-in so gate() can run without loading Gemma. per_row_ld calls
    model(d[key]); we branch on the input's first token (clean tokens are 0, corrupt
    tokens are 1) so clean and corrupt runs can return different logits."""

    def __init__(self, clean_logits, corrupt_logits):
        self._clean = clean_logits
        self._corrupt = corrupt_logits

    def __call__(self, tokens):
        return self._clean if int(tokens[0, 0]) == 0 else self._corrupt


def _d_for(n, n_pos=2):
    return {
        "clean": torch.zeros(n, n_pos, dtype=torch.long),
        "corrupt": torch.ones(n, n_pos, dtype=torch.long),
        "ans_clean": torch.zeros(n, dtype=torch.long),
        "ans_corrupt": torch.ones(n, dtype=torch.long),
        "end": torch.full((n,), n_pos - 1),
        "n": n,
    }


def _logits(n, clean_ans, corrupt_ans):
    """logits favoring ans id 0 by `clean_ans` and id 1 by `corrupt_ans` at end pos."""
    x = torch.zeros(n, 2, 3)
    x[:, -1, 0] = clean_ans
    x[:, -1, 1] = corrupt_ans
    return x


def test_gate_passes_when_model_does_task():
    # clean run prefers id 0 (+10); corrupt run prefers id 1 (-10) -> positive, separated.
    model = _FakeModel(_logits(10, 5.0, -5.0), _logits(10, -5.0, 5.0))
    ok, info = gate(model, _d_for(10), min_n=8, min_sep=1.0)
    assert ok
    assert info["mean_clean_logit_diff"] > 0 and info["clean_minus_corrupt_sep"] > 1.0


def test_gate_fails_on_too_few_pairs():
    model = _FakeModel(_logits(4, 5.0, -5.0), _logits(4, -5.0, 5.0))
    ok, info = gate(model, _d_for(4), min_n=8, min_sep=1.0)
    assert not ok and "single-token pairs" in info["reason"]


def test_gate_fails_when_model_cannot_do_task():
    # clean and corrupt runs identical -> zero separation -> gate rejects.
    model = _FakeModel(_logits(10, 0.0, 0.0), _logits(10, 0.0, 0.0))
    ok, info = gate(model, _d_for(10), min_n=8, min_sep=1.0)
    assert not ok and "behavior gate failed" in info["reason"]


def test_multihop_spearman_helpers():
    from nanofeatures.run_multihop import _ci, _rankdata, _spearman

    assert _rankdata([10.0, 20.0, 30.0]) == [1.0, 2.0, 3.0]
    assert _rankdata([5.0, 5.0, 9.0]) == [1.5, 1.5, 3.0]  # ties share the mean rank
    assert _spearman([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]) > 0.999
    assert _spearman([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]) < -0.999
    med, lo, hi = _ci([0.0, 0.5, 1.0, 0.5, 0.5])
    assert lo <= med <= hi


def test_calibrate_recommendation_and_summary():
    from nanofeatures.calibrate import CalibrationReport, KCell, _recommend

    # the recommendation maps each TOST verdict to the right action; distributedness nudges
    # the inconclusive case toward attribution.
    assert "attribution" in _recommend("attr_win", 1.0)
    assert "cheap baseline" in _recommend("equivalent", 1.0)
    assert "beats attribution" in _recommend("cheap_win", 1.0)
    assert "distributed" in _recommend("inconclusive", 1.5)
    assert "distributed" not in _recommend("inconclusive", 1.0)

    rep = CalibrationReport(
        layer=7,
        n=20,
        distributedness_pr=1.49,
        delta=0.05,
        cells=[
            KCell(
                k=32,
                attribution_suff=(0.62, 0.5, 0.74),
                strongest_cheap="diff_mag_max",
                strongest_cheap_suff=(0.37, 0.25, 0.49),
                attr_minus_strongest=(0.25, 0.15, 0.35),
                tost_verdict="attr_win",
                recommendation=_recommend("attr_win", 1.49),
            )
        ],
    )
    s = rep.summary()
    assert "distributed" in s and "k= 32" in s and "attr_win" in s


def test_tost_verdict_four_way():
    from nanofeatures.run_equivalence import tost_verdict

    d = 0.05
    # significant positive gap -> attribution win (even if small)
    assert tost_verdict(0.01, 0.04, 0.015, 0.035, d) == "attr_win"
    # significant negative gap -> cheap baseline win
    assert tost_verdict(-0.04, -0.01, -0.035, -0.015, d) == "cheap_win"
    # 90% CI inside +-delta and 95% CI straddles 0 -> equivalent (TOST passes)
    assert tost_verdict(-0.03, 0.03, -0.02, 0.02, d) == "equivalent"
    # wide CI crossing 0 and exceeding delta -> inconclusive
    assert tost_verdict(-0.09, 0.08, -0.07, 0.06, d) == "inconclusive"
    # significance takes precedence over equivalence (tiny-but-significant is a win, not a tie)
    assert tost_verdict(0.001, 0.03, 0.005, 0.02, d) == "attr_win"


def test_precision_compare_cells_flatten():
    # the M17 precision comparison flattens a run_scale report to {(task,layer,k): cell} over
    # GATED tasks only; an ungated task must be dropped so it can't be matched across runs.
    from nanofeatures.run_precision_compare import _cells

    report = [
        {"task": "skipme", "gated": False, "info": {"reason": "x"}},
        {
            "task": "capitals",
            "gated": True,
            "layers": {"20": {"16": {"attr_beats_strongest": True}}},
        },
    ]
    cells = _cells(report)
    assert set(cells) == {("capitals", "20", "16")}
    assert cells[("capitals", "20", "16")]["attr_beats_strongest"] is True
