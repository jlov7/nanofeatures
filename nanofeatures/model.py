"""Load Gemma-2-2B into TransformerLens (hookable) and Gemma Scope SAEs via sae-lens."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer


def pick_device(prefer: str | None = None) -> str:
    if prefer:
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_gemma(device: str | None = None, dtype=None) -> HookedTransformer:
    device = pick_device(device)
    kwargs = {"dtype": dtype} if dtype is not None else {}
    model = HookedTransformer.from_pretrained("gemma-2-2b", device=device, **kwargs)
    model.eval()
    return model


def load_gpt2(device: str | None = None) -> HookedTransformer:
    """GPT-2-small — the second model (generality check; IOI's home turf)."""
    device = pick_device(device)
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    model.eval()
    return model


def load_gemma9b(device: str | None = None, dtype=torch.bfloat16) -> HookedTransformer:
    """Gemma-2-9B — the scale-generality point. Default bf16 (~18GB) fits 48GB unified memory
    with headroom; our method's working set (one cached residual layer + one backward) is small.
    Pass dtype=torch.float32 (with device="cpu") for the precision-validation run: fp32 is ~36GB,
    which fits 48GB only on CPU (no MPS activation overhead and, critically, no MPS watchdog that
    a swap-thrashing fp32 model on the GPU could trip). Uses `from_pretrained_no_processing`:
    TransformerLens advises it for reduced precision, and it is also MORE correct here because
    it loads weights exactly as HF (no LayerNorm folding, which would add error), so the
    resid_post hook matches the activations Gemma Scope 9B SAEs were trained on."""
    device = pick_device(device)
    model = HookedTransformer.from_pretrained_no_processing(
        "gemma-2-9b", device=device, dtype=dtype
    )
    model.eval()
    # Attribution needs the gradient w.r.t. the RESIDUAL ACTIVATION at one hook, never the
    # parameter gradients. With params requiring grad, the attribution backward allocates a
    # full set of param-grad buffers (~36GB in fp32, on top of the ~36GB fp32 model) -> >48GB
    # -> the OS memory-killer (jetsam) terminates the run. Disabling param grad removes that
    # buffer; the activation grad is identical (the grab hook makes the residual a grad leaf),
    # so attribution values are unchanged (verified byte-identical on 2B). This is what makes
    # the fp32 precision run fit in 48GB without swapping or crashing.
    model.requires_grad_(False)
    return model


def load_sae9b(
    layer: int, device: str | None = None, width: str = "16k", dtype=torch.bfloat16
):
    """Gemma Scope 9B canonical residual SAE, cast to `dtype` to match the 9B model (an SAE in a
    different dtype would mismatch the residual/grad). Pass dtype=torch.float32 for the fp32
    precision-validation run. Returns (sae, hook_name)."""
    device = pick_device(device)
    loaded = SAE.from_pretrained(
        "gemma-scope-9b-pt-res-canonical",
        f"layer_{layer}/width_{width}/canonical",
        device=device,
    )
    sae = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
    sae = sae.to(dtype)
    return sae, sae.cfg.metadata.hook_name


def load_gpt2_sae(layer: int, device: str | None = None):
    """Joseph Bloom's canonical GPT-2-small residual SAEs. Returns (sae, hook_name)."""
    device = pick_device(device)
    loaded = SAE.from_pretrained(
        "gpt2-small-res-jb", f"blocks.{layer}.hook_resid_pre", device=device
    )
    sae = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
    return sae, sae.cfg.metadata.hook_name


def load_sae(layer: int, device: str | None = None, width: str = "16k"):
    """Gemma Scope canonical residual-stream SAE for a layer. Returns (sae, hook_name).
    `width` selects the SAE width (e.g. "16k" default, "65k") for the width-robustness sweep."""
    device = pick_device(device)
    loaded = SAE.from_pretrained(
        "gemma-scope-2b-pt-res-canonical",
        f"layer_{layer}/width_{width}/canonical",
        device=device,
    )
    sae = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
    return sae, sae.cfg.metadata.hook_name  # sae-lens 6.x: hook under cfg.metadata


class NeuronBasis:
    """Identity 'SAE' shim: encode = identity, W_dec = I, so the SAME feature_rankings /
    sufficiency machinery ranks and patches RAW residual-stream dimensions (neurons) at the
    same residual hook, instead of SAE features. This is the discriminating control for the
    SAE-basis confound: if the cheap-baseline ≈ attribution boundary holds in BOTH the SAE
    basis and the (non-sparse, non-aligned) neuron basis, it is a fact about task topology,
    not an artifact of the Gemma Scope basis being well-aligned. Cf. MIB's neuron-vs-SAE
    comparison (arXiv:2504.13151)."""

    def __init__(self, d_model: int, hook_name: str, device: str):
        self.W_dec = torch.eye(d_model, device=device)  # [d_model, d_model]
        self.cfg = SimpleNamespace(
            d_sae=d_model, metadata=SimpleNamespace(hook_name=hook_name)
        )

    def encode(self, x):
        return x  # features == residual dimensions


def load_neuron_basis(model, layer: int, device: str | None = None):
    """Neuron-basis control at the same residual hook a Gemma Scope res SAE would use."""
    device = pick_device(device)
    hook = f"blocks.{layer}.hook_resid_post"
    assert hook in model.hook_dict, f"{hook} not in model; hooks differ for this model"
    return NeuronBasis(model.cfg.d_model, hook, device), hook
