# Pre-registration — Aff(ℤₙ) family, plain vs. twisted (FROZEN 2026-09-15, before any affine model is trained)

Decisions: n ∈ {11, 13, 15, 16, 21}; recipe R-Wu primary, R-Chughtai as regime control on n = 15; 5 seeds per cell;
Step 0 dihedral pilot runs first; this document is frozen as written. Nulls (Monte Carlo, no model data) are attached as
`results/affine_nulls.md` before training starts; they may not be changed afterwards.

## Models
OneLayerMLP exactly as Chughtai: W_x, W_y ∈ ℝ^{|G|×256}, W ∈ ℝ^{512×128}, W_U ∈ ℝ^{128×|G|}, no bias, ReLU; init randn/√fan_in with
torch.manual_seed(seed). Task: (x, y) ↦ x·y over all |G|² pairs; train split 40% chosen by the seed; full-batch cross-entropy.
- R-Wu: Adam, lr 1e-2, betas (0.9, 0.98), L2 weight decay 2e-4, 25,000 epochs.
- R-Chughtai: AdamW, lr 1e-3, betas (0.9, 0.98), decoupled weight decay 1.0, 250,000 epochs.
Cells: R-Wu × {11, 13, 15, 16, 21} × seeds 1–5 (25 models); R-Chughtai × {15} × seeds 1–5 (5); R-Wu at width m = 512 × {15} × seeds 1–5 (5, secondary).
Controls: untrained networks at init, 10 seeds per n (NOT memorizers — lesson from S₅ P2).
Inclusion: a model enters the analysis iff final test accuracy ≥ 0.99; excluded models are listed with their accuracy.
Dev/confirm: seeds 1–2 are dev (pipeline bugs only), seeds 3–5 confirm; no statistic or bar changes after dev except a recorded bug fix.

## Group and blocks
Aff(ℤₙ) = {x ↦ ax + b}, mul(g,h) = g∘h (apply h then g); both coset sides always taken. T = translations, M_x = stabilizer of x.
Character table by Burnside–Dixon (validated by orthogonality and Σd² = |G|). Blocks under test = the irreps in Ind_M(1) = ⊕_{d|n} ρ_d
(dim φ(d)) with dim ≥ 4: n=11: {10}; 13: {12}; 15: {4, 8}; 16: {4, 8}; 21: {6, 12}. Other irreps reported descriptively.

## Statistics (per neuron j, side s ∈ {x, y}; u = centered column of W_s @ W[side block])
- S1  e_ρ(u) = ‖P_ρ u‖²/‖u‖² for every irrep; block-dominant if some e_ρ ≥ 0.9; live if energy ≥ 1e-8 × max in that side.
- S2  α_plain(u) = max over x ∈ ℤₙ and side ∈ {l, r} of ‖Q_{M_x} u‖²/‖u‖²   (plain coset projectors of the n point stabilizers).
- S3  α_twist(u) = max over frequency pairs {k, −k} (k ≠ 0) and side ∈ {l, r} of ‖(Q^{ψ_k}_T + Q^{ψ_{−k}}_T) u‖²/‖u‖²
      (real single-frequency subspace: f(g·t) or f(t·g) ∝ cos/sin(2πk b/n) — "one Fourier mode in the translation").
- Bars: aligned_plain ⇔ α_plain ≥ 0.90; aligned_twist ⇔ α_twist ≥ 0.90 (nulls: p99 ≤ 0.40 for every block under test; see attachment).
- Per model and block B: F_plain(B), F_twist(B) = energy-weighted fraction of B-dominant live columns that are aligned_plain / aligned_twist.

## Predictions
- P-A (which basis). For each block under test and each included model: exactly one of F_plain(B) ≥ 0.5, F_twist(B) ≥ 0.5.
  Falsified-neither if, for a block, the majority of confirm models have both < 0.3 (no description fits).
  Falsified-both if the majority have both ≥ 0.5 (the families are not exclusive in practice; report the per-column overlap).
  Declared expectations (not part of the test): S₅ evidence → plain; modular-arithmetic evidence → twisted.
- P-B (slack). Slack of the plain realization = #divisors(n) − 1 (1 for 11, 13; 3 for 15, 21; 4 for 16); twisted slack = 0.
  Prediction: median over confirm seeds of F_plain(top block) is lower for composite n than for prime n by ≥ 0.2.
  Falsified if the gap is < 0.05 or reversed. (If P-A says twisted everywhere, P-B is moot and reported as such.)
- P-C (regime). On n = 15, R-Wu and R-Chughtai give the same P-A verdict per block for ≥ 4 of 5 seed pairs. Falsified if ≤ 2.
- P-D (width, secondary). m = 512 on n = 15 gives the same P-A verdict as m = 128 for ≥ 4 of 5 seeds; otherwise reported, not tested.
- Controls: untrained networks have F_plain = F_twist = 0 (α at null level) for every block.

## Descriptive (reported, not tested)
Side used by twisted columns (frequency-permuting vs. fixed-frequency) per input; side used by plain columns (function of b vs. b/a);
neurons per block; organization of the χ ≠ 1 blocks and of the linear characters; per-block α distributions; irrep-energy curves over
training (which blocks form first).

## Step 0 (dihedral pilot; released Chughtai D59/D61 MLPs; frozen with this document)
Per neuron/side: e_k = energy in V_{ρ_k}; single-frequency if max_k e_k ≥ 0.9. For single-frequency columns, A_k = the 2×2 Fourier
coefficient at ρ_k (orthogonal realization); r1 = σ₁²/(σ₁²+σ₂²). P0: ≥ 90% of single-frequency energy has r1 ≥ 0.999
(null for a random 2×2: reported). The pilot cannot separate plain from twisted on D_n (twisted is a tautology; plain reduces to r1).

## Amendment 1 (2026-09-15, recorded BEFORE any affine model was trained)
The attached nulls (`results/affine_nulls.md`) contradict the sentence "nulls: p99 ≤ 0.40 for every block under test". Correct values:
plain p99 ≤ 0.39 for every dim ≥ 8 block and 0.52 (n=21, d=7), 0.75–0.76 for the two dim-4 blocks; twisted p99 0.33–0.50 for
dim ≥ 8 blocks, 0.69 (n=21, d=7), and 0.90 for the dim-4 blocks (n=15 d=5; n=16 d=8), i.e. ≈ 1% of random columns clear the
0.90 twisted bar there. Bars stay at 0.90 for all blocks (a 1% per-column false-positive rate cannot produce F_twist ≥ 0.5).
Per-column descriptive statistics for the two dim-4 blocks are flagged low-power; P-A/P-B/P-C/P-D are unchanged.

## Dev-phase observation (2026-09-15, before confirm seeds)
Under R-Wu (25k epochs), composite-n dev models all reached test acc 1.0 (aff15 s1,s2: ≥0.99 by epoch 500; aff16 s1,s2: by 600–700;
aff21 s1: 1.0), but prime-n dev models mostly did not (aff11 s1 0.016, aff13 s1 0.016, aff13 s2 0.032; aff11 s2 grokked at epoch 6800).
Pipeline unaffected. A longer schedule (R-Wu, 100k epochs; same optimizer and hyperparameters) is being tested on the failed dev seeds.
Decision rule, fixed now: if the 100k schedule groks the primes, confirm seeds for n ∈ {11, 13} run at 100k epochs and composites stay
at 25k; P-B is then computed as pre-registered but reported with the schedule difference flagged, and composites at 100k are added as a
robustness check if time permits. If primes still fail, P-B is reported as untestable at m = 128 and the prime cells are dropped.

## Dev-phase findings recorded BEFORE any confirm model was analyzed (2026-09-15/16)
Trainer validation: our R-Chughtai S5 seed 2 reproduces Chughtai's released seed 2 (sign 0.077/0.075, std 0.819/0.813, explained
0.897/0.889, F_coset 1.0, families S4 0.92 / A5 0.07 / S3xS2 0.01 in both). Random-init controls: F_plain = F_twist = 0 everywhere.
1. CRT factoring. Aff(Z_15) ≅ Aff(F_3) × Aff(F_5) and Aff(Z_21) ≅ Aff(F_3) × Aff(F_7). Dev models put no energy in the top block
   (d = 15, 21) and solve the two factors separately (blocks d = 5 resp. 7, plus d = 3). The top-block verdict there is 'empty'
   (a pre-registered outcome). Consequence for P-B: the only composite n without factoring is 16; the prime-vs-composite
   comparison will be reported for n = 16 vs {11, 13} as the clean case and for the factor blocks of 15/21 separately.
2. Sides: x-input aligns with right cosets (function of x^{-1}(0)), y-input with left cosets (function of y(0)); equal energy in
   the two halves of every grokked model (ratio 0.99–1.03). Twisted: LEFT = frequency permuted by a (k·a^{-1}), RIGHT = fixed frequency
   (the description in the plan text had these swapped; statistics unaffected — both sides are always in the max).
3. Dev verdicts (frozen families, bar 0.90): aff11 s2 top: plain (F_plain 0.994); aff15 s1,s2 d=5: plain (1.0); aff21 s1 d=7:
   plain (1.0); aff16 s1: d=16 plain (1.0), d=8 twist (1.0); aff16 s2: d=16 and d=8 'neither' (both α at null level).
4. The aff16 s2 'neither' neurons are, descriptively, induced from the intermediate subgroup T ⋊ K', K' = {1,7,9,15} ≅ V4 ⊂ Z_16^×:
   each row a carries one frequency pair, ranging over the K'-orbit {3,5,11,13} (5·a^{-1} on K', 9·a^{-1} on 3K'). This is a third
   exact realization (Ind of a 4-dim irrep of an index-2 subgroup), neither plain nor twisted. A DESCRIPTIVE statistic is added
   (not a test): support-orbit size = smallest |O ∪ −O| over K-orbits O of the block's frequencies (K ≤ units) such that the
   fixed-frequency projector summed over O captures ≥ 0.9 of the column's energy; 2 = fixed-frequency twisted, intermediate values
   = intermediate induction, full = plain / permuted-twisted / generic.
5. Prime grokking is marginal: aff11 seed 1 memorized in the 25k-epoch run (test 0.016) but the identical configuration run with
   --epochs 100000 (same seed, recipe, device) had test acc 1.0 by epoch 25k — the divergence can only come from non-deterministic
   CPU reductions, so at m = 128 the primes sit at a grokking threshold under R-Wu. Decision rule (above) stands: prime confirm seeds
   at 100k epochs once aff13's 100k dev run is in; all runs report the epoch at which test acc first reached 0.99.
6. aff21 seed 2: d=7 plain (F_plain 1.0), top block empty — same as seed 1.
7. 'empty' verdicts explained (descriptive): aff15 s5 and aff15-R-Chughtai s1 carry the F20 factor in the sibling irrep
   rho_{5,chi} = 4 ⊗ sgn(S3) (zero energy in rho_{5,1}); aff16 s3–s5 carry the d=8 block in its sibling rho_{8,chi}. The network
   chooses among character-twisted siblings of a block at random across seeds (the S5 std/std' choice). The frozen "blocks under test"
   named only rho_{d,1}; siblings are now scored DESCRIPTIVELY with the same families plus the plain-with-character family
   (M_x, lambda) for every non-trivial character lambda of the units (Ind_M(lambda) contains rho_{d,chi} iff lambda|K_d = chi).
8. Rank (descriptive, Wu-style): the coefficient matrix of every block-dominant column is rank one (r1 >= 0.99 for 100% of energy)
   in every model except aff16 s2's top block (74% of energy rank 2) — the intermediate-induced structure of item 4 is not a
   projected rho-set; aff16 s4's 'neither' columns ARE rank one, with intermediate stabilizers.

## Amendment 2 (2026-09-16, applying the recorded decision rule)
Aff(Z_13) at m = 128 under R-Wu does not grok: seeds 1, 2 at 25k epochs (test 0.016, 0.032) and seed 1 at 100k epochs (final 0.016;
maximum test accuracy during training 0.017). Aff(Z_11) groks at 100k in 2 of 2 seeds (epochs 8,000 and 12,500) and at 25k in 1 of 2.
Per the rule: the n = 13 cell is DROPPED from P-A/P-B at m = 128; n = 11 confirm seeds 2–5 run at 100k epochs (launched); P-B's
prime side is n = 11 only and is reported with that caveat. Optional, NOT pre-registered: n = 13 under R-Chughtai (the pre-registered
alternative recipe), to be run after the confirm queues finish and reported as exploratory.
9. The sibling block 8β of Aff(Z_16) (used by seeds 3, 4, 5) is 'neither' under every pre-registered/descriptive family, and the
   reason is a genuinely different subgroup: its neurons are constant (alpha 0.99–1.00, trivial character) on left cosets of an
   order-16 subgroup H = {(a, c(a)), (a, c(a)+8)} that contains NO point stabilizer and is not conjugate to any S_{x,8}
   (Aff(Z_16) has extra index-8 subgroup classes from the 2-torsion of Z_16). The frozen 'plain' family (point stabilizers) is one
   subgroup class among several valid coset realizations; the network picks per block and per seed. A uniform descriptive sweep over
   all subgroup classes of index <= 16 with all linear characters and Monte-Carlo nulls (`scripts/affine_subgroup_sweep.py`) replaces the
   ad-hoc fingerprints as the descriptive answer to 'which subgroup'.
10. Aff(Z_11) confirm seeds 3–5 (100k): all grokked (epochs 13,500 / 6,500 / 30,500).
