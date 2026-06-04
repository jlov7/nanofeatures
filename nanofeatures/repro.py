"""Reproducibility manifest: record the exact environment that produced the reports.

The reviewers' top credibility note was that few-pp CIs are quoted without pinning the software
that generated them. This captures, verifiably, the resolved versions of every load-bearing
package plus the model/SAE identifiers, and (best effort) the HuggingFace commit hash each model
artifact resolved to in the local cache. Run `uv run python -m nanofeatures.repro` and commit
`reports/repro_manifest.json`; a stranger can then match the environment exactly. This does not
re-pin the load calls (TransformerLens does not expose `revision` cleanly), but it makes the exact
provenance of the numbers checkable, which is the actual gap.
"""

from __future__ import annotations

import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PACKAGES = [
    "torch",
    "transformer_lens",
    "sae_lens",
    "transformers",
    "numpy",
    "huggingface_hub",
]

# the model/SAE artifacts this study loads (see model.py)
ARTIFACTS = {
    "gemma-2-2b": "transformer_lens HookedTransformer",
    "gemma-2-9b": "transformer_lens HookedTransformer (no_processing)",
    "gpt2": "transformer_lens HookedTransformer",
    "gemma-scope-2b-pt-res-canonical": "sae_lens SAE",
    "gemma-scope-9b-pt-res-canonical": "sae_lens SAE",
    "gpt2-small-res-jb": "sae_lens SAE",
}


def _ver(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def _hf_revisions():
    """Best-effort: resolve the cached commit hash for each HF repo, so the exact weights are
    identifiable even though the load calls do not pin a revision."""
    out = {}
    try:
        from huggingface_hub import scan_cache_dir

        cache = scan_cache_dir()
        wanted = {
            "google/gemma-2-2b",
            "google/gemma-2-9b",
            "gpt2",
            "openai-community/gpt2",
        }
        wanted |= {
            "google/gemma-scope-2b-pt-res-canonical",
            "google/gemma-scope-9b-pt-res-canonical",
            "jbloom/GPT2-Small-SAEs-Reformatted",
        }
        for repo in cache.repos:
            if repo.repo_id in wanted or any(
                w in repo.repo_id for w in ("gemma", "gpt2", "SAE")
            ):
                revs = sorted(r.commit_hash for r in repo.revisions)
                out[repo.repo_id] = revs
    except Exception as e:  # cache scan is best-effort; never fail the manifest
        out["_note"] = f"hf cache scan unavailable: {e}"
    return out


def manifest() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {p: _ver(p) for p in PACKAGES},
        "artifacts": ARTIFACTS,
        "hf_cached_revisions": _hf_revisions(),
    }


def main() -> None:
    m = manifest()
    out = Path("reports/repro_manifest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=2))
    print(f"wrote {out}")
    print("packages:", {k: v for k, v in m["packages"].items()})


if __name__ == "__main__":
    main()
