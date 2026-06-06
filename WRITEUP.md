# When is expensive interpretability actually worth it?

*A calibrated answer for SAE feature circuits. A free, gradient-free baseline matches
gradient attribution patching on single-token tasks, then loses to it by 15–45 points on a
distributed circuit. The boundary is predictable from the task. Everything here runs on one
laptop, and it overturned its own first draft twice.*

---

## The question

Mechanistic interpretability has a growing toolbox for finding circuits, the small set of
components (or lately, sparse-autoencoder features) that carry a behavior. The prestigious
tools are causal. There's activation patching, where you swap a component's value between a
clean and a corrupt run and measure the effect, and its cheap gradient approximation,
attribution patching (AtP). Papers report that these methods "recover the circuit," and the
implicit comparison is usually against nothing, or against a deliberately weak baseline.

So here's the uncomfortable question. How much of the win survives if you calibrate against
the strongest *cheap* baseline you can build? And if it survives, when?

`nanofeatures` answers that for SAE feature circuits on a real model (Gemma-2-2B + Gemma
Scope SAEs), with paired-bootstrap confidence intervals on every number. The answer isn't
"expensive methods are useless," and it isn't "they're essential." It's a boundary, and the
boundary turns out to be the interesting part.

## Setup, in one paragraph

Pick a task with single-token answers (say *"The capital of {country} is the city of"* →
the capital). Build aligned clean/corrupt pairs that differ only in the subject. At one
layer, score every active SAE feature by how much it carries the clean-vs-corrupt
logit-difference, take the top-k as the "circuit," patch them back in, and measure how much
of the behavior they recover. That recovered fraction is sufficiency. Then compare selection
rules on the same activations: exact per-feature ablation, gradient attribution (AtP), and a
set of cheap gradient-free heuristics. The question is whether the expensive rules pick a
more faithful circuit than the cheap ones, by how much, and with what error bars.

## Result 1: the baseline you pick decides the answer

The natural cheap baseline ranks features by *summed* |Δactivation| over positions. Against
that one, attribution looks dominant, winning by 17–35 points. But it's a strawman: summing
dilutes the contrastive signal across positions that are identical between clean and corrupt.
Rank instead by the *peak* per-position |Δactivation| (`diff_mag_max`, just as free, same
activation cache) and almost the entire gap disappears.

That's the whole game. Calibrate against a weak baseline and you "prove" your method is
necessary; calibrate against the right one and you find out whether it actually is.

## Result 2: on single-token tasks, the gradient buys ~nothing

Against the honest `diff_mag_max` baseline, across seven tasks (factual recall, antonyms,
three flavors of morphology, and number succession), 3 layers and 3 sparsities each, gradient
attribution beats it in only 9 of 63 cells, by ≤6 points, and it loses outright in a few.
Exact ablation beats the cheap gradient approximation in only 3 of 21. The ladder collapses
at the top:

I did not want to lean on "not significant" as if it meant "equal," so I tested it. A
two-one-sided equivalence test (TOST, margin 5 points, comparator pre-registered as
`diff_mag_max` to avoid a winner's-curse pick) on all 63 cells gives a four-way split: 10
small attribution wins, 13 statistical equivalences, 4 cheap-baseline wins, and 36 inconclusive
at this n. So the truthful claim is not "they are equal everywhere" but "attribution never wins
by a meaningful margin here, sometimes loses, and is provably equivalent in a fifth of cells",
much weaker than the IOI win, which is the whole point.

```
magnitude ≈ random (~0)  <<  diff_mag (summed)  <<  diff_mag_max ≈ attribution ≈ exact causal
```

For these tasks, a free heuristic is as good as attribution patching. Both of these facts
have precedent: MIB (arXiv:2504.13151) shows cheap baselines rival SAE-featurized methods,
and AtP\* established that AtP ≈ exact. What's here is a clean replication on SAE features,
with CIs.

## Result 3: on a distributed circuit, the gradient is essential

An adversarial reviewer made the right objection. Single-token tasks differ at one position,
so the causal signal is concentrated and any contrastive selector finds it. The tie is
almost guaranteed. Gradient methods exist for the *distributed* circuits, with path
cancellation, that single-token tasks leave out.

So we ran the identical ladder on IOI (indirect-object identification: the subject name
appears twice, the signal spans three name positions, and the known circuit has cancelling
paths). The tie vanishes. Attribution beats `diff_mag_max` at 12/12 cells by +15 to +45
points, every CI clear of zero.

![Attribution beats a free baseline only on a distributed circuit](docs/boundary.png)

And we measured why, rather than asserting it. `diff_mag_max` is position-blind: it takes the
biggest activation change at any position. When the signal is one token swap, the
biggest-change feature is the relevant one, so the heuristic ties the gradient. On IOI the
biggest change sits at the *first* subject mention, which the final-token logit barely reads.
Comparing which positions are causally relevant (by full-residual recovery) against where
each method's top features peak: attribution puts 25 of its top 32 features at the
indirect-object position that matters, while `diff_mag_max` wastes 11 of 32 on that first
subject mention. The cheap baseline chases the biggest change; the gradient chases the
biggest effect on the readout.

## Result 4: the boundary is predictable, and it isn't the SAE

Two points (single-token vs IOI) is an anecdote, so we made it quantitative and ruled out the
obvious confound.

Take predictability first. For each task we measured a ranking-free distributedness score:
the participation ratio of per-position causal recovery. Patch the full residual at each
position, see how many positions actually carry the signal. We also added four
2-token-subject tasks to fill the middle. Across 11 tasks, every task whose signal spans more
than one position (2-token recall, IOI) shows attribution winning (+13 to +28pp); every
single-position task ties (≤+5pp). The separation is perfect, Mann-Whitney one-sided
p = 0.003.

![Attribution's advantage tracks how distributed the circuit is](docs/distributedness.png)

The continuous score corroborates it (Spearman ρ = +0.62 [+0.01, +0.89]) but it's noisy, so
the load-bearing claim is the categorical single-vs-multi split, reported as exactly that
rather than a smooth law. Either way, you can tell in advance, from whether the contrastive
signal is single- or multi-position, whether the gradient is worth its cost.

There is an obvious confound I had to rule out: in this task set every multi-position task is
also small-n (the modal-length filter shrinks them), so "wins on distributed circuits" could be
"wins when n is small." So I subsampled each task to matched n and recomputed the gap. It is
flat in n: at n=8, IOI is still +22% while capitals and past-tense are +4–5%, and each task
stays in its lane all the way up. If n drove it, capitals at n=8 would look like IOI; it does
not. The boundary is topology, not sample size (`run_nsweep`).

Now the confound. Maybe "cheap ties attribution" is just the well-aligned Gemma Scope basis
making selection trivial? We re-ran the entire ladder in the raw residual-neuron basis (an
identity SAE: rank and patch residual dimensions directly). The boundary reproduces exactly,
9/63 single-token and 12/12 IOI. It's task topology, not the SAE.

It isn't the model or the scale either. Re-running everything on GPT-2-small with a different
author's SAEs (Joseph Bloom's residual SAEs) reproduces the boundary almost number-for-number:
single-token 10/63 (vs Gemma's 9/63), IOI 9/9 with attribution ahead by +42 to +145pp. And it
holds all the way up to Gemma-2-9B with Gemma Scope 9B SAEs (run in bf16 to fit a 48GB laptop):
single-token a tie-to-small-edge (3/12 cells, +3 to +12pp), IOI a clean win (3/3, +18 to +40pp).
Three models from 124M to 9B, three unrelated SAE families, the same line. It also reproduces at
a 4x-wider SAE (Gemma Scope width 65k): the single-token tie is even cleaner there (0/21) and
IOI still wins.

And the bf16 the 9B run uses isn't quietly doing the work. I re-ran the whole 9B ladder in fp32
on CPU (parameter grad off, so the attribution backward fits in memory) and compared it cell by
cell: all 15 cells agree on the verdict, every one inside the bootstrap noise, the largest
sufficiency drift 0.7pp. The node numbers are real, not a rounding accident, because node
sufficiency effects are large (tens of points); reduced precision has nothing to erase.

And it survives the strongest objection to the whole setup: that ranking features one at a
time can't see interactions, which is exactly where a gradient is supposed to earn its keep.
So I built the interaction-aware gold standard, a greedy circuit that adds the feature most
raising joint sufficiency at each step, and raced it against both top-k methods. It does not
rescue attribution on single-token tasks. On capitals all three are close; on antonyms the
joint oracle beats both first-order methods, and cheap actually edges attribution. On IOI,
attribution already tracks the joint oracle while cheap lags it by +34pp. So accounting for
interactions doesn't flip the single-token tie in attribution's favor. It does surface a
separate, honest fact: on some tasks the faithful circuit is genuinely interaction-heavy, and
there you need joint selection rather than either first-order rule.

## Result 5: for edges, the gradient is what ranks them, and you can say exactly why

Everything so far is about nodes, which single features matter. But circuits are also edges,
which feature at one layer feeds which at the next, and edge attribution (EAP) is precisely
where a gradient is supposed to be irreplaceable. So I pushed the same question one level up.
If the cheap baseline ties the gradient for picking nodes on single-token tasks, does that
survive when the thing you're scoring is the connection?

It doesn't, and the way it fails is unusually clean because the experiment is controlled.
Define an exact mediated edge effect with no gradient: for an edge from feature u to feature d
a layer later, patch u to its clean value, read the exact amount that moves d, then move d by
exactly that and measure the change in the metric. That's the indirect effect of u routed
through d, the gold standard an edge score should recover. Every candidate score factors as
transfer times a readout, and they all use the *same* exactly measured transfer, so they
differ only in the readout. The question becomes: what kind of readout does an edge need?

The gradient recovers the exact edge almost perfectly, Spearman +0.99, on capitals and IOI, on
Gemma-2-2B and again on GPT-2-small, across three different layer pairs. (That's AtP at the
edge level, and I checked it isn't a tautology: the second-order gap between the gradient and
the exact effect is 7 to 12 percent, and the perturbations aren't infinitesimal.) No
free score recovers it, and the controls lay out a clean 2x2 over two readout properties,
causal and position-resolved. A score that uses the downstream feature's activation change as
the readout is position-resolved but not causal: it manages +0.2 on the single-token task and
about zero on IOI. A score that uses a node ablation (move the downstream feature on its own,
see how much the answer moves) is genuinely causal but collapses everything to one number per
feature, and comes out *anti*-correlated with the true edge, around −0.2. Both fail. Then the
clincher: a gradient-free readout that is both causal and position-resolved, built from a
per-position finite-difference probe, recovers the exact edge as well as the gradient (+0.99,
indistinguishable from EAP). So the claim isn't "you need the gradient." It's that an edge
needs a readout with both properties, and the gradient is just the cheap way to get them, one
backward pass instead of n_pos forward probes. Drop either property and the score breaks.

![An edge needs a causal AND position-resolved readout](docs/edge_2x2.png)

Does the ranking advantage matter behaviorally? I patched the top edges each method selects and
read the recovered behavior. On the distributed circuit it does: the gradient-selected edge
circuit recovers as much as the exact-selected one and beats every gradient-free-selected
circuit, and on GPT-2, where the edges are large, a magnitude-selected circuit recovers
*negative* behavior. On single-token tasks the per-edge effects are tiny and the methods tie
behaviorally, so the gradient's ranking advantage there is real but doesn't show up in
behavior. The topology boundary, in other words, comes back at the behavioral edge level even
though the ranking result holds everywhere.

Two things I checked before trusting any of this. The finite-difference control has one free
knob, the probe step, so I swept it eightfold (h from 0.25 to 2): the +0.99 recovery and the tie
with EAP hold at every step, so it isn't reading an artifact of one step size. And precision: in
bf16 the whole edge structure collapses into rounding noise (EAP vs exact from +0.99 to about
zero). My first explanation was the obvious one, that the tiny final logit-difference cancels, and
it was wrong: IOI's metric baseline is small (~1.7), its bf16 ulp is ~0.007, and its edge effect
(~0.05) sits about seven times above that floor, yet IOI collapses as hard as capitals. So I
instrumented each intermediate in bf16 against fp32 on CPU. The gradient readout is fine (relative
error ~0.03). What breaks is the exact mediated-effect reference and the transfer measurement,
both of which are differences of near-identical full-network forward passes under a tiny added
perturbation, where bf16 cannot resolve the small induced change (the exact effect's ranking
correlation to fp32 drops to 0.2 to 0.5). In other words it is the gold-standard reference that
degrades, not the gradient, and it degrades on CPU too, so it is the dtype, not the MPS backend.
The edge work runs in fp32 because validating an edge means resolving a fragile
perturbation-difference, even though the gradient itself is precision-robust, and that contrast is
the point: how much precision you need scales with how big the quantity you are differencing is.

And it isn't a single-layer-pair trick. I extended the exact mediated-effect to a genuine 2-hop
chain (u at L1, through m at L2, to d at L3, composing the exact transfers hop by hop) and the same
2x2 fell out on both chains I tried and both task types: the causal and position-resolved readouts
(the gradient and the gradient-free finite-difference alike) recover the exact 2-hop effect at rho
0.93 to 0.98, magnitude and node-ablation fail. So the readout law is a property of edge
attribution itself, not of one layer pair, and the gradient-free version keeps pace across the
chain. The one thing that drifts over a wider span is the gradient's value-scale fidelity, not its
ranking, so longer-range magnitude estimates still want the exact effect.

So the calibration discipline doesn't always end with "the cheap thing was enough." For nodes
on single-position tasks it did. For edges, ranking them faithfully needs the gradient, and the
value is knowing precisely what it buys you (a causal, position-resolved readout) and where
that buys you behavior (distributed circuits) before you spend the compute.

## Five times the data corrected me

The reason I trust these numbers is that the project kept telling me I was wrong, and I kept the
corrections in:

1. **Summed to peak.** My first baseline summed per-position activation change; attribution
   looked decisive. The honest baseline takes the *peak* position, and the gap mostly closed.
2. **Flat-negative to boundary.** I first read the single-token result as "attribution is
   useless," then ran IOI and found it is essential there. The finding was the boundary, not
   either half.
3. **"Tie" to "inconclusive."** Calling non-significance a tie was sloppy; the TOST says only 13
   of 63 single-token cells are provably equivalent and 36 are inconclusive at this n. Weaker,
   truer.
4. **Position-resolution is not the whole story.** I hypothesized the gradient's node advantage
   was purely position-resolution, and built a gradient-free position-weighted score to prove it.
   It did not close the gap, and hurt on single-token tasks. So position-resolution and
   feature-direction causality are separable, and the cheap score lacks both (`run_posaware`).
5. **The bf16 edge mechanism.** I first wrote that bf16 cancels the tiny final logit-difference.
   Instrumenting it showed that is wrong (the gradient is bf16-robust); the fragile step is the
   exact mediated-effect reference, a perturbation-difference. The corrected mechanism is in
   Result 5.

## The takeaway: a calibration discipline

None of this is a new method. It's a discipline, with a result attached:

1. Calibrate against the strongest cheap baseline you can build, not a strawman. (Peak, not
   summed, and a learned probe where one applies.)
2. Measure your oracle; don't assume it. The companion repo,
   [`nanocircuits`](https://github.com/jlov7/nanocircuits), shows toy-model "ground-truth" circuits that don't
   actually reproduce the behavior they're graded against.
3. Put paired-bootstrap CIs on every gap, and when you want to claim two methods *tie*, run an
   equivalence test (TOST) against a pre-registered comparator. Non-significance is not
   equivalence; at small n the honest verdict is often "inconclusive."
4. Show breadth, and find the boundary. One task is an anecdote; the boundary is the finding.
5. Watch your numerics. The same effect can be bf16-safe (large node-sufficiency effects) or
   bf16-fatal (tiny edge perturbation-differences); check, don't assume.

Do that, and "do you need expensive causal attribution for SAE feature circuits?" has a crisp
answer: only when the circuit is distributed across positions, which you can check before you
spend the compute.

## What this is and isn't

It's a small (a few hundred lines), laptop-scale, reproducible, CI-backed calibration of
existing methods. It's been through repeated adversarial audits (statistics, code, concept,
numerics); the sample-size confound became the n-sweep, the basis-confound became the neuron
control, and the "tie" became a TOST. It corrected its own claims five times (above), and it
reports every number that didn't go its way.

It is not a new attribution method, and it doesn't claim attribution is useless. The headline
methods rank features one at a time; the interaction-aware check shows that doesn't hide an
attribution advantage on single-token tasks, and the edge result (Result 5) carries the
question to cross-layer connections, where ranking edges faithfully needs a causal,
position-resolved readout (the gradient is its cheap analytic instance, a gradient-free
finite-difference recovers exact just as well) while the cached scores fail. The honest soft
spot there is the behavioral edge metric: a single-layer-pair edge
set is a sliver of the full circuit, so absolute edge-circuit recovery is small on Gemma and
only large on GPT-2; the robust edge result is the rank-recovery of the exact effect, and a
full multi-layer behavioral edge reconstruction is left open. The tasks are small (n = 7–30).
The weights aren't revision-pinned, and the distributedness metric is a coarse proxy. All of
that is in the repo's limitations, stated plainly, because the whole point of the artifact is
that you can trust its numbers.

## Reproduce

```bash
git clone … && cd nanofeatures && uv sync
TRANSFORMERLENS_ALLOW_MPS=1 uv run python -m nanofeatures.run_suite --layers 5 7 9   # single-token side
TRANSFORMERLENS_ALLOW_MPS=1 uv run python -m nanofeatures.run_ioi --layers 5 7 9 11  # distributed side
TRANSFORMERLENS_ALLOW_MPS=1 uv run python -m nanofeatures.run_distributedness         # the boundary, quantified
TRANSFORMERLENS_ALLOW_MPS=1 uv run python -m nanofeatures.run_model2                  # second model (GPT-2-small)
TRANSFORMERLENS_ALLOW_MPS=1 uv run python -m nanofeatures.run_scale --layers 20       # 9B scale (Gemma-2-9B, bf16)
TRANSFORMERLENS_ALLOW_MPS=1 uv run python -m nanofeatures.run_mechanism               # the position-blindness mechanism
TRANSFORMERLENS_ALLOW_MPS=1 uv run python -m nanofeatures.run_interactions            # interaction-aware (greedy) check
TRANSFORMERLENS_ALLOW_MPS=1 uv run python -m nanofeatures.run_edges --pairs 5,7 3,9 6,7  # cross-layer edges: gradient-free ladder vs exact + behavioral circuit
uv run pytest -q                                                                      # 16 model-free tests
```

Full details, every table, and the failure modes are in the [README](README.md); the
unifying argument across both repos is in [THESIS.md](THESIS.md).
