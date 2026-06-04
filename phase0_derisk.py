"""Phase-0 de-risk gate for the feature-level circuits program (Gemma-2-2B + Gemma
Scope SAEs). Confirms the tooling actually loads and works on this machine BEFORE
committing to the program. Three gates; each prints PASS/FAIL. Defensive about
sae-lens API differences across versions.
"""

import time

import torch

device = (
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
print(f"[env] torch={torch.__version__} device={device}")

# ---- Gate 1: load Gemma-2-2B in a hookable framework + forward pass ----
t0 = time.time()
from transformer_lens import HookedTransformer  # noqa: E402

model = HookedTransformer.from_pretrained("gemma-2-2b", device=device)
model.eval()
prompt = "The Eiffel Tower is in the city of"
with torch.no_grad():
    logits = model(prompt)
top = logits[0, -1].topk(5).indices.tolist()
preds = [model.to_string([t]) for t in top]
print(
    f"[gate1 PASS] gemma-2-2b loaded in {time.time() - t0:.1f}s; n_layers={model.cfg.n_layers}; "
    f"top-5 next tokens: {preds}"
)
mem = torch.mps.current_allocated_memory() / 1e9 if device == "mps" else 0.0
print(f"[gate1] approx MPS allocated: {mem:.1f} GB")

# ---- Gate 2: load a Gemma Scope SAE + confirm it reconstructs ----
from sae_lens import SAE  # noqa: E402

# Gemma Scope residual-stream SAEs. Try the canonical release; print directory on fail.
LAYER = 20
release = "gemma-scope-2b-pt-res-canonical"
sae_id = f"layer_{LAYER}/width_16k/canonical"
try:
    loaded = SAE.from_pretrained(release, sae_id, device=device)
    sae = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
    print(
        f"[gate2] loaded SAE {release} :: {sae_id}; d_sae={sae.cfg.d_sae}, "
        f"hook={sae.cfg.metadata.hook_name}"
    )
except Exception as e:
    print(
        f"[gate2] direct load failed ({type(e).__name__}: {str(e)[:120]}); listing directory..."
    )
    from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

    d = get_pretrained_saes_directory()
    gemma = [k for k in d if "gemma-scope-2b" in k]
    print("  available gemma-scope-2b releases:", gemma[:10])
    raise

# reconstruct activations at the SAE hook point
hook = sae.cfg.metadata.hook_name
with torch.no_grad():
    _, cache = model.run_with_cache(prompt, names_filter=hook)
    acts = cache[hook]  # [batch, pos, d_model]
    feats = sae.encode(acts)  # [batch, pos, d_sae]
    recon = sae.decode(feats)  # [batch, pos, d_model]
cos = (
    torch.nn.functional.cosine_similarity(
        acts.flatten(0, 1), recon.flatten(0, 1), dim=-1
    )
    .mean()
    .item()
)
l0 = (feats[0, -1] > 0).sum().item()
print(
    f"[gate2 PASS] reconstruction cos-sim={cos:.3f}; active features at last pos (L0)={l0}"
)

# ---- Gate 3: a basic faithfulness signal (feature necessity) ----
# Ablate the top-k active features at the last position and measure the drop in the
# answer logit. A meaningful feature set should move the output.
answer = model.to_single_token(" Paris")
with torch.no_grad():
    base_logit = model(prompt)[0, -1, answer].item()

    topk = feats[0, -1].topk(20).indices

    def ablate(act, hook, _topk=topk):  # TransformerLens calls with hook= kwarg
        f = sae.encode(act)
        f[:, -1, _topk] = 0.0
        return act + (
            sae.decode(f) - sae.decode(sae.encode(act))
        )  # replace recon component

    abl_logit = model.run_with_hooks(prompt, fwd_hooks=[(hook, ablate)])[
        0, -1, answer
    ].item()

print(
    f"[gate3 PASS] ' Paris' logit base={base_logit:+.2f} -> ablate-top20-features={abl_logit:+.2f} "
    f"(Δ={abl_logit - base_logit:+.2f})"
)
print(
    "[phase0] all gates passed -> feature-level program is grounded on this machine."
    if abl_logit != base_logit
    else "[phase0] gate3 produced no change - investigate."
)
