# Calibrated interpretability

*A two-artifact case that most "method X discovers the circuit" claims shrink or vanish once
you (a) calibrate against the strongest cheap baseline and (b) measure your ground-truth
oracle instead of assuming it.*

---

## The claim

Circuit-discovery papers routinely report that an attribution method recovers a known
circuit: high AUROC against ground truth, or high faithfulness when you patch the selected
nodes or features. The implicit comparison is against nothing, or against a deliberately weak
baseline (random, raw activation magnitude). Two failure modes hide inside that move.

First, a strong cheap baseline you never tried would have done just as well, and the
expensive method takes credit for signal a free heuristic already captures. Second, the
oracle you grade against may not be faithful: a high score against a "ground-truth circuit"
that doesn't actually reproduce the model's behavior is measuring agreement with a label, not
causal discovery.

Run both checks and the headline usually moves, often all the way to "the cheap thing was
already enough." This repo and its predecessor are two small, reproducible demonstrations:
one on toy models with literal ground truth, one on a real LLM.

---

## The protocol

Before claiming a circuit-discovery method *works*, four steps:

1. Calibrate against the strongest cheap baseline you can build, not a strawman. Not just
   random and magnitude, but the best position-aware, contrastive, gradient-free heuristic
   you can think of, plus a small learned probe on structural features. A baseline you tried
   hard to make strong and still beat is evidence; one you could beat is theatre.
2. Measure the oracle. If you grade against a "ground-truth circuit," first check it actually
   reproduces the behavior. Ablate everything else: is it sufficient? Ablate it: is it
   necessary? Throw out the oracles that fail.
3. Put error bars on every gap. Small n and small positive sets make single point estimates
   meaningless. Paired-bootstrap over examples, and report whether each gap's CI excludes
   zero. Treat n.s. as no effect.
4. Show breadth. One task, one layer, one direction is an anecdote. Sweep tasks, layers, and
   sparsities, and report the per-cell breakdown rather than a cherry-picked one.

---

## Demonstration 1: toy models with literal ground truth (`nanocircuits`)

The setup: InterpBench/tracr SIIT transformers and GPT-2 IOI, where the ground-truth circuit
is known. Grade resample-ablation attribution by AUROC against the GT node/edge set. Here's
what calibration did to the headline.

A leave-one-case-out logistic probe on purely structural node features (layer, head index,
is-MLP, never reading activations) learns where circuits tend to live, and it's far stronger
than hand heuristics. Held to it, the apparent causal wins collapse. One case fell from a
+0.19 "win" to a tie. Clean wins survive on only cases 3 and 11 (plus IOI at +0.21), not the
original 4–6/7.

The oracle didn't hold up either. Sufficiency and necessity ablations showed several
InterpBench oracles aren't faithful. Case 21 is only 0.02-sufficient; its labelled circuit
barely reproduces the model, so its high AUROC (0.997) isn't causal discovery and it gets
excluded despite "beating baseline." And with proper significance testing, edge-level
recovery is mostly negative: only 1/7 cases both beats baseline and survives Bonferroni.

The net: a high AUROC against a known circuit can reflect where the circuit sits (a
structural prior) or an unfaithful label, rather than discovery. The honest result is a
methods caution, with the clean wins explicitly scoped.

## Demonstration 2: a real LLM with SAE features (`nanofeatures`)

The setup: Gemma-2-2B + Gemma Scope SAEs. Select a feature circuit by gradient attribution
patching (AtP), and measure faithfulness behaviorally (patch the top-k features, read the
recovered logit-difference). Seven single-token contrastive tasks (factual recall, antonyms,
three morphologies, sequence) plus IOI, a distributed circuit; layers {5,7,9(,11)};
k∈{16,32,64}; paired-bootstrap 95% CIs against the strongest cheap baseline. What calibration
revealed wasn't a flat negative but a boundary.

The baseline variant decides the apparent result. Ranking features by *summed* |Δactivation|
(the natural cheap baseline) loses to attribution by 17–35pp, which looks decisive. Ranking
by *peak* per-position |Δactivation| (`diff_mag_max`, equally gradient-free, same cache)
closes almost all of that gap. The summed baseline was a strawman; the peak one is the honest
competitor.

Against that honest baseline, on single-token tasks the gradient buys almost nothing:
attribution beats `diff_mag_max` (CI excludes 0) in only 9/63 task×layer×k cells, all ≤6pp,
and it's occasionally beaten. On a distributed circuit it's a different story. Same harness,
same metric, same baselines, but on IOI attribution beats `diff_mag_max` at 12/12 cells by
+15 to +45pp, every CI clear of zero. The negative result is scoped to single-position
signal. Exact per-feature ablation, meanwhile, beats the 1-backward-pass approximation in
only 3/21 single-token and 3/12 IOI cells (all ≤7pp), so AtP ≈ exact throughout.

Three checks make the boundary hard to wave away. It's basis-independent: rerun the identical
ladder in the raw residual-neuron basis (not SAE features) and it reproduces, 9/63
single-token and 12/12 IOI, so the verdict tracks task topology rather than the Gemma Scope
SAE making selection trivial. It's predictable as a CUE here (the sequel, nanoassembly, shows this cue does not survive as a reliable cheap routing rule once feature confusability and depth enter): across 11 tasks, every task whose contrastive
signal spans more than one token position (2-token-subject recall, IOI) shows attribution
winning (+13 to +28pp), and every single-position task ties (≤+5pp), a perfect separation at
Mann-Whitney p=0.003. (That's the categorical claim; the continuous distributedness
correlation, Spearman +0.62, corroborates it but is weaker.) Two honesty checks sharpen the
single-token side. First, "tie" is tested, not assumed: a two-one-sided equivalence test (TOST,
5pp margin, pre-registered against diff_mag_max) over the 63 single-token cells returns 10 small
attribution wins, 13 genuine equivalences, 4 cheap-baseline wins, and 36 inconclusive at this n,
so the honest statement is "no meaningful gradient advantage," not "exactly equal." Second, the
distributedness label is confounded with sample size in this task set (every multi-position task
is small-n), so we subsample each task to matched n: the gap is flat in n (at n=8, IOI is +22pp
while single-token tasks are +4 to +5pp), confirming the boundary is topology, not sample size.
And it replicates across scale:
GPT-2-small with Joseph Bloom's SAEs reproduces it almost number-for-number (single-token
10/63, IOI 9/9 at +42 to +145pp), and Gemma-2-9B with Gemma Scope 9B SAEs (bf16, on a 48GB
laptop) holds the same line (single-token tie/small-edge, IOI 3/3 at +18 to +40pp). Three
models from 124M to 9B, three SAE families, one boundary. The bf16 9B run is not a precision
artifact: re-running the identical ladder in fp32 (on CPU, with parameter grad disabled so the
attribution backward fits) reproduces every cell, all 15 within bootstrap noise, maximum
sufficiency drift 0.7pp. Node sufficiency effects are large (10 to 90pp), so reduced precision
does not touch them.

It also survives the strongest methodological objection, that greedy single-feature ranking
can't see interactions. An interaction-aware greedy-exact circuit (add the feature that most
raises joint sufficiency) does not favour attribution on single-token tasks: on capitals all
three are close, and on antonyms the joint oracle beats both first-order methods while cheap
actually edges attribution. On IOI, attribution already tracks the joint oracle while cheap
lags it by +34pp. So accounting for interactions doesn't flip the single-token tie in
attribution's favour, and it surfaces a separate fact: on some tasks the faithful circuit is
interaction-heavy, and there you need joint selection, not either first-order rule.

Push from nodes to cross-layer edges (which feature at one layer feeds which at the next) and
the node-level tie does not carry over, which is itself part of the calibration. Define an
exact mediated edge effect with no gradient: patch the upstream feature to clean, read how much
it moves the downstream feature, move the downstream feature by exactly that, and measure the
metric change. Every candidate edge score factors as transfer times a readout proxy and shares
the same exactly-measured transfer, so they differ only in the readout, and the question
becomes what kind of readout an edge needs. The gradient edge score (EAP) recovers the exact
effect almost perfectly on both a single-token task and IOI (Spearman +0.99, on Gemma-2-2B and
again on GPT-2-small, across three layer pairs), the AtP ≈ exact fact at the edge level, and it
is not a tautology: the second-order residual is 7 to 12 percent and the perturbations are not
infinitesimal. No cheap (cached, no-extra-forward) score recovers it, on either task, but, as
the controls show, a gradient-free score that pays for the right measurement does. The mechanism
is sharp because the controls lay out a clean 2x2 over two readout properties, causal and
position-resolved. An activation-magnitude readout is position-resolved but not causal (+0.2 on
the single-token task, ~0 on IOI). A node-ablation readout is causal but collapsed to one scalar
per feature, which makes it anti-correlated with the exact edge (−0.2). Both fail. A
gradient-free per-position finite-difference readout, which is both causal and position-resolved,
recovers the exact edge as well as the gradient (+0.99, indistinguishable from EAP). So the claim
is not "you need the gradient" but the sharper one that an edge needs a causal, position-resolved
readout, of which the gradient is simply the cheap analytic instance (one backward pass versus
n_pos forwards); drop either property and the score breaks. The
behavioral check agrees where it can resolve the small per-edge effects: a gradient-selected
edge circuit recovers behavior matching the exact-selected ceiling and beats every
gradient-free-selected circuit on the distributed circuit (and on GPT-2, where edges are large,
a magnitude-selected edge circuit recovers negative behavior), while on single-token tasks the
edges are individually too small for the ranking advantage to surface behaviorally. So the
calibration cuts both ways: the cheap baseline suffices for node selection on single-position
tasks, and for edges it does not, with the gradient supplying exactly the two readout
properties no cheaper score has. The finite-difference control is robust to its one free knob:
sweeping the probe step over a factor of eight (h in 0.25, 0.5, 1, 2) leaves the recovery at
+0.99 and indistinguishable from EAP at every step, so it reads a real causal+position-resolved
quantity, not a tuned one. The one place reduced precision does bite is here, and pinning down
why is instructive rather than fatal. In bf16 the edge recoveries collapse to noise (EAP vs exact
from +0.99 to about 0, second-order residual from 0.10 to 0.95). The tempting explanation, that
the tiny final logit-difference cancels, is wrong: IOI's metric baseline is small (about 1.7), its
bf16 unit-in-last-place is about 0.007, and its edge effect (about 0.05) sits roughly seven times
above that floor, yet IOI collapses as hard as capitals. Instrumenting each intermediate in bf16
against fp32 on CPU locates the real culprit: the gradient readout is bf16-robust (relative error
about 0.03), but the exact mediated-effect reference and the transfer measurement are differences
of near-identical full-network forward passes under a tiny added perturbation, and bf16 cannot
resolve the small induced change (the exact-effect ranking's correlation to fp32 falls to about
0.2 to 0.5). So the collapse is mostly the exact reference degrading, not the gradient; the
gradient itself survives reduced precision. It reproduces on CPU, so it is a property of the dtype,
not of the MPS backend. Computing or validating cross-layer edge effects therefore runs in fp32 by
design, because the gold-standard reference is a fragile perturbation-difference.
That is the precise mirror of the node result that survives bf16 at 9B: precision tolerance
scales with effect size, large for node sufficiency, small for mediated edges.

The 2x2 is not a single-layer-pair artifact. Extending the exact mediated-effect machinery to a
genuine 2-hop chain (patch u at L1 to clean, read the induced change in m at L2, patch m by exactly
that, read the induced change in d at L3, move d by that, measure the metric) reproduces the same
lattice on both tested chains (5 to 7 to 9, 4 to 7 to 10) and both task types: the causal and
position-resolved readouts (the gradient eap and the gradient-free finite-difference) recover the
exact 2-hop effect at rho 0.93 to 0.98, while magnitude (position-resolved, not causal) and
node-ablation (causal, not position-resolved) fail, the latter anti-correlated. The gradient-free
readout keeps pace with the gradient across the chain, so the law for which edges matter is
depth-robust; the only thing that drifts over a wider span is eap's value-scale fidelity (the
ranking holds but the linear fit loosens), so longer-range magnitude estimates still want the exact
effect. The structural claim is therefore general: an edge needs a causal and position-resolved
readout, in a single pair or a multi-layer path.

The mechanism is measured, not asserted, and it's what makes the boundary predictable.
`diff_mag_max` is position-blind: it takes the largest change at any position. So it ties the
gradient exactly when the signal is one token swap, and fails when the causally-relevant
feature isn't the biggest-change one. On IOI attribution puts its features at the
causally-relevant positions (the IO name, the second subject mention) while the cheap baseline
wastes a third of its picks on the first subject mention, where the change is large but the
readout barely looks. So the calibrated answer to "do you need attribution?" is "only if your
circuit is distributed," and that's knowable in advance. Both load-bearing facts here (cheap
baselines rival featurized methods; AtP ≈ exact) are already established (MIB,
arXiv:2504.13151; AtP\*, Kramár et al. 2024). The contribution is the clean summed-vs-peak
flip and the CI-backed topology boundary, not a new method.

---

## Why this is worth saying

Both artifacts cut against incentive. The rewarded result is "my method discovers circuits."
nanocircuits lands on "the apparent win was a structural-prior or unfaithful-oracle
artifact." nanofeatures lands somewhere more useful than a flat negative: a boundary, where
the cheap baseline was enough on single-position tasks and the expensive method earns its
cost exactly where theory says it should, on distributed circuits. In both cases, applying
the calibration discipline changed the conclusion you would have published. The contribution
isn't a new method. It's that discipline, plus two small, CI-backed, reproducible
demonstrations that it bites, on toy ground truth and on a real model.

Reproduce: each repo is a few hundred LOC of method, runs on one laptop, no GPU cluster.
See [`nanofeatures/README.md`](README.md) and [`nanocircuits`](https://github.com/jlov7/nanocircuits).
