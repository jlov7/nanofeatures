"""calibrate(): the one-call calibration harness - the tool form of this whole study.

The artifact's thesis is a discipline: before you claim causal attribution discovered an SAE
feature circuit, calibrate it against the STRONGEST cheap baseline, put a paired-bootstrap CI on
the gap, EQUIVALENCE-test (TOST) the "tie" instead of reading it off non-significance, and check
the task's positional distributedness so you know which regime you are in. People reimplement
pieces of this and get it wrong (weak baseline, no CI, non-significance read as equivalence). This
gives the full protocol in one call on YOUR model / SAE / task.

    from nanofeatures.calibrate import calibrate
    from nanofeatures.model import load_gemma, load_sae
    from nanofeatures.circuit import aligned_pairs
    from nanofeatures.tasks import TASK_SUITE

    model = load_gemma()
    sae, hook = load_sae(7)
    d = aligned_pairs(model, *TASK_SUITE["capitals"])
    report = calibrate(model, sae, hook, d, layer=7)
    print(report.summary())

Returns a CalibrationReport: per-k strongest cheap baseline, attribution's sufficiency, the gap
with a 95% CI, the four-way TOST verdict (attr_win / equivalent / cheap_win / inconclusive) vs a
pre-registered diff_mag_max at margin delta, the task's positional distributedness PR, and a plain
recommendation. Composes only already-tested primitives; it does not re-derive the math.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .circuit import feature_rankings, positional_distributedness
from .run_equivalence import tost_verdict
from .run_m5 import bootstrap_suff, ci, per_row_ld, per_row_patched_ld
from .run_suite import BASELINES

COMPARATOR = (
    "diff_mag_max"  # the pre-registered cheap comparator for the equivalence test
)


@dataclass
class KCell:
    k: int
    attribution_suff: tuple  # (median, lo, hi)
    strongest_cheap: str
    strongest_cheap_suff: tuple
    attr_minus_strongest: tuple  # 95% CI of the gap over the strongest cheap baseline
    tost_verdict: (
        str  # attr_win | equivalent | cheap_win | inconclusive (vs diff_mag_max)
    )
    recommendation: str


@dataclass
class CalibrationReport:
    layer: int
    n: int
    distributedness_pr: float
    delta: float
    cells: list = field(default_factory=list)

    def summary(self) -> str:
        topo = (
            "distributed (signal spans positions) -> expect the gradient to earn its cost"
            if self.distributedness_pr > 1.3
            else "single-position-like -> expect a cheap baseline to suffice"
        )
        lines = [
            f"calibration @ layer {self.layer}  (n={self.n}, distributedness PR={self.distributedness_pr:.2f}: {topo})",
            f"  equivalence margin delta = {self.delta:.0%}; comparator = {COMPARATOR}",
        ]
        for c in self.cells:
            a, lo, hi = c.attr_minus_strongest
            lines.append(
                f"  k={c.k:>3}: attribution {c.attribution_suff[0]:+.0%} vs strongest cheap "
                f"({c.strongest_cheap}) {c.strongest_cheap_suff[0]:+.0%} | "
                f"gap {a:+.0%} [{lo:+.0%},{hi:+.0%}] | TOST: {c.tost_verdict} -> {c.recommendation}"
            )
        return "\n".join(lines)


def _pct(vals, qs):
    t = torch.tensor(vals)
    return [torch.quantile(t, q).item() for q in qs]


def _recommend(verdict: str, pr: float) -> str:
    if verdict == "attr_win":
        return (
            "use causal attribution (it beats the cheap baseline by a real margin here)"
        )
    if verdict == "cheap_win":
        return f"use the cheap baseline ({COMPARATOR}); it beats attribution here"
    if verdict == "equivalent":
        return f"a cheap baseline ({COMPARATOR}) suffices (provably within margin)"
    return "inconclusive at this n; gather more pairs" + (
        " (distributed task: lean toward attribution)" if pr > 1.3 else ""
    )


def calibrate(
    model,
    sae,
    hook,
    d,
    layer: int,
    ks=(16, 32, 64),
    boot: int = 5000,
    seed: int = 0,
    delta: float = 0.05,
) -> CalibrationReport:
    """Run the full calibration protocol on one (model, SAE, task). See module docstring."""
    ks = list(ks)
    clean_ld = per_row_ld(model, d, "clean")
    corrupt_ld = per_row_ld(model, d, "corrupt")
    out = feature_rankings(model, sae, hook, d, exact=False)
    rankings, f_clean, f_corrupt = out[0], out[1], out[2]

    methods = ["attribution", *BASELINES]
    cache = {
        (m, k): per_row_patched_ld(
            model, sae, hook, d, rankings[m][:k], f_clean, f_corrupt
        )
        for m in methods
        for k in ks
    }
    g = torch.Generator().manual_seed(seed)
    n = d["n"]
    idxs = [torch.randint(0, n, (n,), generator=g) for _ in range(boot)]

    pr = positional_distributedness(model, d, layer)[0]
    report = CalibrationReport(layer=layer, n=n, distributedness_pr=pr, delta=delta)
    for k in ks:
        boot_s = {
            m: [bootstrap_suff(clean_ld, corrupt_ld, cache[(m, k)], ix) for ix in idxs]
            for m in methods
        }
        per = {m: ci(v) for m, v in boot_s.items()}
        strongest = max(BASELINES, key=lambda b: per[b][0])
        gap_strong = ci(
            [a - s for a, s in zip(boot_s["attribution"], boot_s[strongest])]
        )
        # TOST vs the pre-registered comparator
        comp_gap = [a - c for a, c in zip(boot_s["attribution"], boot_s[COMPARATOR])]
        lo95, hi95 = _pct(comp_gap, [0.025, 0.975])
        lo90, hi90 = _pct(comp_gap, [0.05, 0.95])
        verdict = tost_verdict(lo95, hi95, lo90, hi90, delta)
        report.cells.append(
            KCell(
                k=k,
                attribution_suff=per["attribution"],
                strongest_cheap=strongest,
                strongest_cheap_suff=per[strongest],
                attr_minus_strongest=gap_strong,
                tost_verdict=verdict,
                recommendation=_recommend(verdict, pr),
            )
        )
    return report
