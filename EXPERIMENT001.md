# Experiment 001 — the inward head

Protocol fixed before any real model run (2026-11-25). No results yet.

Question: does a closed-loop controller over layer address, spectral aperture and dwell, with a persistent omitted-state residue, cause more structured, seed-dependent latent dynamics than fixed or nonresponsive controls in a frozen ordinary LLM?

## Mechanism

A rolling context of continuous embeddings is passed through the actual frozen model once each tick. No KV cache, no text generation as feedback except in the discrete baseline. The native logits provide a fixed-size candidate shortlist. The head reads an intermediate residual, projects its centered state into a nested PCA subspace, and reads it through the model's final norm and native unembedding on the shortlist. Native and lens scores are blended. A soft mixture of the resulting top eight token embeddings is the next input; no argmax token is fed in latent arms. Snapshot tokens are observations only. The PCA basis is fitted separately per layer on a fixed random unlabeled token calibration batch; this is a variance aperture, NOT proven semantic grain.

Feedback control uses cosine change of the full residual at each address. Too little change opens the lens (lower rank); excessive change closes it (higher rank). The controller chooses a layer with change closest to a target, only after a dwell interval. The optional memory is an EMA of the part omitted by the previous read, separately by layer; its bounded contribution is applied only to the lens, never directly to the model residual. All weights frozen; no external retrieval, no labels or training.

## Arms

- discrete: native categorical sample, same candidate size and temperature, one actual token per tick
- fixed: fixed layer/rank and soft carrier, no residue
- random: exogenous random address/rank, same dwell schedule as periodic
- periodic: exogenous cyclic address/rank
- adaptive: feedback controller, no residue
- residue: same controller plus slow memory
- fixed_residue: memory without adaptive grain/address
- reset: residue arm with a single memory erasure halfway through the run
- shuffled: residue arm with coordinate-permuted memory (same norm, destroys directional alignment)

All arms use one full frozen forward per tick at identical context length for a given seed, same calibration and native shortlist. Discrete baseline has a different readout operation but equal model-forward budget. The random/periodic schedules are not perfectly action-matched to adaptive dwell; report actual action frequencies. Run-level RNG seeds are paired; random arm uses a separate controller RNG. Reset and residue are deterministic and must match exactly before the intervention (within numerical repeatability).

## Fixed evaluation

Default seeds: oak, winter, melody, river, justice, engine, cabbage, memory. Repeat each with RNG seeds 0,1,2 (deterministic arms repeat for reproducibility, not independent samples). Use 64 ticks. Treat each seed concept as the analysis unit; report per-concept means, not thousands of correlated ticks. Pilot with 16 ticks, then freeze settings before full run.

Primary diagnostics: input embedding step cosine change, long-lag recurrence (max cosine to inputs 4–16 ticks earlier), fraction of repeated snapshot IDs, effective snapshot vocabulary (exp of empirical entropy), seed specificity (same-concept versus other-concept cosine of mean final hidden states in last half; exclude identical run). These are mechanical measures, not semantic coherence or reasoning. Report native entropy and shortlist entropy, actual layer/rank/dwell, residue norm, and nonfinite states. Readout words are qualitative only.

Gate 0: finite run, frozen parameters, exact number of forwards and no discrete feedback in latent arms. Reset trajectories must coincide before reset. No claims if these fail.
Gate 1: residue must reduce long-lag recurrence relative to fixed and both exogenous arms by >=0.02 on average AND retain seed specificity at least as well as fixed (within 0.02). Also report discrete baseline. This is an exploratory threshold, not a statistical significance claim. If fixed or random is as good, do not claim adaptive benefit.
Gate 2: residue vs adaptive and fixed_residue, and reset/shuffled controls: a mechanistic memory claim requires a directional difference from shuffled and a measurable post-reset trajectory divergence without pre-reset divergence. Divergence alone does not establish improvement.
Gate 3 (future): blinded human or independent-model semantic evaluation on held-out seed families, then actual tasks. Do not use native lens words as proof of thought, intelligence, consciousness or correctness.

No retuning against the full evaluation set. All logs, including collapse and failures, retained. PCA on random single-token activations may be poorly suited to contextual states; logit lens is uncalibrated; convex embedding mixtures are not necessarily in-distribution; rolling context changes normal inference. The shortlist anchors exploration to native output and may suppress true lateral jumps. These are deliberate conservative first-build constraints, not evidence for semantic hierarchy.
