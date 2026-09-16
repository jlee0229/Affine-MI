# Plan (c): the affine family Aff(ℤₙ) — plain cosets vs. twisted cosets where both are exact

Status 2026-09-15: plan only. Feasibility computation done (`scripts/affine_feasibility.py`); nothing trained.

## 1. Why this group, in one paragraph

On S₅ the coset and irrep descriptions coincide up to slack 1 for every learned irrep, and the one place the network could have used a
sign-twisted realization with less slack (5b via F20~sgn, 6 cosets) it never did — it used plain S₃×S₂ cosets (10 cosets, slack 5) in
all 50 seeds. On dihedral groups the question cannot be asked: the twisted description is a tautology (every irrep-sparse neuron is a
twisted-ℤₙ function) and the plain-coset test degenerates into Wu's rank-1 check (60% of random irrep vectors clear 0.90). Aff(ℤₙ) is
the smallest family where BOTH descriptions are exact realizations of the same irrep AND occupy different subspaces with sharp nulls:

| description | subgroup / character | what a neuron looks like | subspaces in top block V_ρ (Aff ℤ₁₅, dim 64) | null mean / p99 |
|---|---|---|---|---|
| PLAIN (Stander, Wu) | point stabilizer M_x = {g : g(x)=x}, trivial character, either side | a function of the translation coordinate only (or of b/a), filtered to one divisor block: Ramanujan-sum-like | 30 × dim 8 | 0.27 / 0.38 |
| TWISTED (induced, "Fourier") | translation subgroup T, character ψ_k, either side | a single Fourier frequency in b, the same for all a (one side) or permuted k ↦ a⁻¹k (other side) | 16 × dim 8, tiling V_ρ | 0.19 / 0.25 |

Overlap between a plain and a twisted subspace: 0.125 (generic). Same picture for n = 13 (prime, top block dim 12), 16, 21.
The permutation rep on ℤₙ decomposes as ⊕_{d|n} ρ_d with dim ρ_d = φ(d) (verified: 1⊕2⊕4⊕8 for n=15; 1⊕1⊕2⊕4⊕8 for n=16;
1⊕2⊕6⊕12 for n=21; 1⊕12 for n=13). So the plain realization's slack in block d is (#divisors − 1) irreps, growing with n's
factorization, while the twisted realization has slack 0 everywhere. That is the graded-slack test S₅ could not provide.

The composition task's network must pick one (or mix). Either answer is a result:
- H_plain: neurons are functions of the translation only, divisor-filtered → Stander/Wu's picture survives on an M-group where a
  strictly cheaper twisted basis exists; "networks use plain cosets" becomes a general claim and the dissolution is one-directional.
- H_twist: neurons are single Fourier modes in b with a-permuted frequency → the induced-basis description is what the network does;
  Nanda's clock generalizes to non-abelian groups along the induced representation; the S₅ F20~sgn non-use was about S₅, not cosets.
- Mixed by block / by n (e.g. twisted for prime n, plain for composite): the slack gradient decides — the most informative outcome.

## 2. Steps

**Step 0 (optional, ~1 h, free data): dihedral rank-1 pilot.** Chughtai's released D59/D61 MLPs (8 models, all test acc 1.0,
10–15 frequencies each). Test only what is non-vacuous there: per neuron, σ₁²/‖A‖²_F of its 2×2 coefficient matrix per frequency
(bar 0.999; random null mean 0.89). Confirms/denies Wu's rank-1 observation on an M-group. Cannot separate plain from twisted — say so.

**Step 1 (~1 day): `affine` package.** Generalize `cosetprobe` from a fixed S₅ table to any multiplication table:
group data + Burnside–Dixon character table (done in `scripts/affine_feasibility.py`, validated by orthogonality on 4 groups);
projectors: isotypic, twisted for arbitrary (H, λ); the three theoretically motivated families — stabilizers M_x (plain),
(T, ψ_k) (twisted), and the intermediate (T ⋊ K_d, ψ_k ⊗ χ) that realizes the lower blocks exactly; plus a Stander-style
sweep over all subgroups with trivial/sign characters as a control. Nulls per block per family (Monte Carlo, as for S₅).
Self-tests: Σd² = |G|; Ind_M(1) = ⊕_d ρ_d; Ind_T(ψ_k) = ρ_{n/gcd(k,n)} ⊗ (characters of K_d); twisted subspaces tile V_ρ.

**Step 2 (~1 day): trainer.** One-hidden-layer MLP exactly as Chughtai (W_x, W_y, W, W_U; no bias; ReLU), full-batch on a 40%
split, cross-entropy. Two recipes, because the S₅ 6d result showed the regime decides which irreps appear:
  R-Wu: Adam, lr 1e-2, wd 2e-4, 25k epochs (their models grok 10× faster);  R-Chughtai: AdamW, lr 1e-3, wd 1.0, 250k epochs.
Validation before any affine run: reproduce Chughtai S₅ seed 2 with R-Chughtai (grokking, key irreps sign+standard, F_coset ≈ 1),
and Chughtai C113 (Fourier features). Save final weights, train/test curves, and per-irrep energy curves.

**Step 3 (compute: laptop, hours): sweep.** n ∈ {11, 13} (prime; slack-1 plain realization, like S₅'s S4) and {15, 16, 21}
(composite; 4–5 divisor blocks). Orders 110–252; pairs 12k–64k. 5 seeds × 5 groups × R-Wu = 25 models, plus R-Chughtai on n=15
(5 seeds) as the regime control. Estimated 2–20 min per model on the M2 (MPS). Dev/confirm: seeds 1–2 dev, 3–5 confirm, per group.

**Step 4: pre-registered analysis (to be frozen before Step 3 finishes).** Per neuron and side, per divisor block:
S1 irrep-energy profile; S2 α_plain = max over stabilizer-coset subspaces; S3 α_twist = max over single-frequency subspaces;
bar 0.90 for both (null p99 ≤ 0.38 everywhere). Report F_plain and F_twist (energy-weighted) per block per model.
Predictions: P-A "which basis" — for each block the majority of energy is aligned with exactly one family (F_plain ≥ 0.5 xor
F_twist ≥ 0.5); falsified if both < 0.3 (neither description) or both ≥ 0.5 (they are not exclusive on this group — check the overlap
statistic). P-B "slack gradient" — if H_plain holds, plain neurons' b-profiles must be spectrally confined to one orbit
{k : gcd(k,n) = n/d}; measure the confinement and its cost (fraction of neurons per block vs. #divisors). P-C "regime" — R-Wu and
R-Chughtai agree on the family choice on n=15 (falsified if they disagree, which would itself be the finding). Controls: random init
(not memorizers — lesson from S₅ P2). Descriptive: which side (frequency-permuting vs. fixed) the twisted neurons use, per input.

**Step 5 (phase 2, the action-side paper): train on the action (g, x) ↦ g(x).** Same groups, target = the permutation rep
⊕_d ρ_d directly; the plain description is now "a function of x", the twisted one "a Fourier mode in x moved by a". Same statistics.
This is the experiment with no prior work at all; Steps 1–4 build every tool it needs.

## 3. Decisions needed
- n set as above, or add n=20 (φ=8, divisors 1,2,4,5,10,20)? Recommend the five listed; add 20 only if 15/16/21 disagree.
- Recipe: R-Wu primary (cheap), R-Chughtai as control on one n. Or both everywhere (doubles compute, still a laptop job).
- Seeds: 5 per group (as above) or 10.
- Do Step 0 (dihedral pilot) at all? Recommend yes — an hour, and it settles whether "rank-1 per irrep" is safe to assume on M-groups.
- Freeze Step 4 as written, or amend, before training starts.

## 4. Compute
Everything on this laptop. Analysis is seconds per model; training is minutes per model; the whole sweep is an afternoon (R-Wu)
or an overnight run (adding R-Chughtai everywhere). No GPU needed.
