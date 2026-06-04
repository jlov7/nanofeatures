"""The one figure: gradient attribution beats a free baseline ONLY on a distributed
circuit. Reads the committed reports (no hardcoded numbers) and plots, at the common cell
(layer 7, k=64), the attribution − strongest-cheap-baseline sufficiency gap with
paired-bootstrap 95% CIs, for both the SAE-feature basis and the raw-neuron basis. Single-
token tasks cluster at zero; IOI sits far above; the two bases agree (boundary is about
task topology, not the SAE).

    uv run python -m nanofeatures.make_figure   # -> docs/boundary.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAYER, K = "7", "64"
SINGLE = [
    "capitals",
    "country_language",
    "past_tense",
    "comparative",
    "plural",
    "antonyms",
    "successor",
]
LABELS = {
    "capitals": "capitals",
    "country_language": "country→lang",
    "past_tense": "past-tense",
    "comparative": "comparative",
    "plural": "plural",
    "antonyms": "antonyms",
    "successor": "successor",
    "ioi": "IOI",
}


def _gap(cell):
    m, lo, hi = cell["attr_minus_strongest"]
    return 100 * m, 100 * (m - lo), 100 * (hi - m)  # median, lower err, upper err (pp)


def _suite_gaps(path):
    out = {}
    for t in json.loads(Path(path).read_text()):
        if t.get("gated"):
            out[t["task"]] = _gap(t["layers"][LAYER][K])
    return out


def _ioi_gap(path):
    return _gap(json.loads(Path(path).read_text())["layers"][LAYER][K])


def main() -> None:
    sae = _suite_gaps("reports/m7_suite.json")
    sae["ioi"] = _ioi_gap("reports/m8_ioi.json")
    neu = _suite_gaps("reports/m9_neuron_suite.json")
    neu["ioi"] = _ioi_gap("reports/m9_neuron_ioi.json")

    tasks = SINGLE + ["ioi"]
    xs = list(range(len(tasks)))

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.axhline(0, color="#888", lw=1, ls="--", zorder=1)
    # shade the distributed region
    ax.axvspan(
        len(SINGLE) - 0.5, len(tasks) - 0.5, color="#ffd9d9", alpha=0.5, zorder=0
    )

    def series(data, dx, marker, fill, label):
        ys = [data[t][0] for t in tasks]
        lo = [data[t][1] for t in tasks]
        hi = [data[t][2] for t in tasks]
        ax.errorbar(
            [x + dx for x in xs],
            ys,
            yerr=[lo, hi],
            fmt=marker,
            ms=8,
            mfc=(fill or "none"),
            mec="#1f3b73" if fill else "#b2182b",
            ecolor="#1f3b73" if fill else "#b2182b",
            color="#1f3b73" if fill else "#b2182b",
            capsize=3,
            lw=1.5,
            label=label,
            zorder=3,
        )

    series(sae, -0.10, "o", "#1f3b73", "SAE-feature basis")
    series(neu, +0.10, "s", None, "raw-neuron basis (control)")

    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[t] for t in tasks], rotation=25, ha="right")
    ax.set_ylabel(
        "attribution − strongest cheap baseline\n(sufficiency, percentage points)"
    )
    ax.set_title(
        "Gradient attribution beats a free baseline only on a distributed circuit",
        fontsize=13,
        fontweight="bold",
    )
    ax.text(
        (len(SINGLE) - 1) / 2,
        ax.get_ylim()[1] * 0.92,
        "single-token tasks: tie",
        ha="center",
        fontsize=10,
        color="#444",
    )
    ax.text(
        len(SINGLE) + 0.0,
        ax.get_ylim()[1] * 0.92,
        "distributed\n(IOI): wins",
        ha="center",
        fontsize=10,
        color="#b2182b",
    )
    ax.legend(loc="upper left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(
        0.5,
        -0.02,
        "Gemma-2-2B + Gemma Scope SAEs · layer 7 · k=64 · paired-bootstrap 95% CI (B=5000). "
        "Both bases agree → the boundary is task topology, not the SAE.",
        ha="center",
        fontsize=8.5,
        color="#555",
    )

    out = Path("docs/boundary.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


def distributedness_figure():
    """M10: the attribution−cheap gap vs measured per-position distributedness, colored by
    the single- vs multi-position split. Shows the clean group separation."""
    r = json.loads(Path("reports/m10_distributedness.json").read_text())
    single, multi = [], []
    for row in r["rows"]:
        x = row["distributedness_PR"]
        y, ylo, yhi = 100 * row["gap_med"], 100 * row["gap_lo"], 100 * row["gap_hi"]
        (single if row["task"] in SINGLE else multi).append(
            (x, y, y - ylo, yhi - y, row["task"])
        )

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.axhline(0, color="#888", lw=1, ls="--", zorder=1)
    for pts, color, mark, lab in [
        (single, "#1f3b73", "o", "single-position signal (1 token differs)"),
        (multi, "#b2182b", "s", "multi-position signal (≥2 tokens / IOI)"),
    ]:
        ax.errorbar(
            [p[0] for p in pts],
            [p[1] for p in pts],
            yerr=[[p[2] for p in pts], [p[3] for p in pts]],
            fmt=mark,
            ms=9,
            color=color,
            ecolor=color,
            capsize=3,
            lw=1.5,
            ls="none",
            label=lab,
            zorder=3,
        )
        for x, y, _lo, _hi, name in pts:
            ax.annotate(
                LABELS.get(name, name),
                (x, y),
                textcoords="offset points",
                xytext=(7, 4),
                fontsize=8,
                color=color,
            )
    ax.set_xlabel(
        "measured distributedness  (participation ratio of per-position causal recovery)"
    )
    ax.set_ylabel(
        "attribution − strongest cheap baseline\n(sufficiency, percentage points)"
    )
    ax.set_title(
        "Attribution's advantage tracks how distributed the circuit is",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="upper left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    rho = r["spearman_rho"]
    lo, hi = r["spearman_rho_ci"]
    p = r["single_vs_multi_p_onesided"]
    fig.text(
        0.5,
        -0.02,
        f"Gemma-2-2B · layer 7 · k=32 · 11 tasks · single-vs-multi separation Mann-Whitney "
        f"p={p:.3f}; Spearman ρ={rho:+.2f} [{lo:+.2f},{hi:+.2f}]. Error bars: paired-bootstrap 95% CI.",
        ha="center",
        fontsize=8.5,
        color="#555",
    )
    out = Path("docs/distributedness.png")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


def edge_2x2_figure():
    """M14: the readout 2x2. An edge score recovers the exact mediated effect only if its readout
    is BOTH causal and position-resolved. Cells colored by Spearman-vs-exact (IOI, L5->7); only the
    both-properties cell works, reached by the gradient (eap) OR a gradient-free finite-difference
    (cheap_fdpos)."""
    import numpy as np

    edges = {
        c["task"]: c
        for c in json.loads(Path("reports/m14_edges.json").read_text())
        if c.get("l1") == 5 and c.get("l2") == 7
    }
    fdpos = {
        c["task"]: c
        for c in json.loads(Path("reports/m14_edges_fdpos.json").read_text())
    }
    raw = edges["ioi"]["spearman_vs_exact"]
    sv = {k: (v[0] if isinstance(v, list) else v) for k, v in raw.items()}
    fdv = fdpos["ioi"]["cheap_fdpos_vs_exact"]
    fd = fdv[0] if isinstance(fdv, list) else fdv
    # rows top->bottom: causal Yes, causal No ; cols left->right: pos-resolved No, Yes
    M = np.array([[sv["cheap_node"], sv["eap"]], [sv["mag"], sv["cheap"]]])
    labels = [
        [
            f"cheap_node\n(node ablation)\nρ={sv['cheap_node']:+.2f}",
            f"eap (gradient)  ρ={sv['eap']:+.2f}\ncheap_fdpos (grad-free)  ρ={fd:+.2f}",
        ],
        [
            f"mag\n(summed |Δact|)\nρ={sv['mag']:+.2f}",
            f"cheap\n(per-pos |Δact|)\nρ={sv['cheap']:+.2f}",
        ],
    ]
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    im = ax.imshow(M, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                labels[i][j],
                ha="center",
                va="center",
                fontsize=10.5,
                fontweight="bold" if (i == 0 and j == 1) else "normal",
                color="#111",
            )
    ax.set_xticks([0, 1], ["No", "Yes"])
    ax.set_yticks([0, 1], ["Yes", "No"])
    ax.set_xlabel("position-resolved readout?", fontsize=12)
    ax.set_ylabel("causal readout?", fontsize=12)
    ax.set_title(
        "An edge needs a causal AND position-resolved readout",
        fontsize=13,
        fontweight="bold",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman ρ vs exact mediated edge effect", fontsize=10)
    fig.text(
        0.5,
        -0.02,
        "Gemma-2-2B · IOI · layers 5→7 · top 24×24 edges. Only the both-properties cell recovers "
        "the exact edge, and the gradient is just the cheap way to get there (a gradient-free "
        "per-position finite-difference does as well).",
        ha="center",
        fontsize=8.5,
        color="#555",
        wrap=True,
    )
    out = Path("docs/edge_2x2.png")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
    distributedness_figure()
    edge_2x2_figure()
