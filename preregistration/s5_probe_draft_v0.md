# Pre-registration draft — S5 monomiality probe (v0, 2026-09-08, NOT yet agreed)

Status: no coset statistics have been computed on any model. Only (i) Chughtai's published per-irrep energies,
(ii) the subgroup↔irrep dictionary, and (iii) Monte-Carlo nulls (no model data) have been looked at.

## Objects
- Models: Chughtai `OneLayerMLP` S5 runs. Primary set = 50 mainline seeds (embed 256, hidden 128, all test acc 1.0).
  Controls = the 24 non-grokking sweep models (test acc ≈ 0) — same statistics, prediction: null-level coset alignment.
  Development seeds 1–10; confirmation seeds 11–50 (statistics frozen before touching 11–50).
- Per neuron j: left/right PRE-activations as functions on G:
  u_j^x(x) = (W_x W[:E])[x, j],  u_j^y(y) = (W_y W[E:])[y, j],  each centered.  (Stander's object; hidden = ReLU(u^x + u^y).)
  Convention: Chughtai's model computes x*y in sympy order (apply x then y) = Stander's σ_l σ_r with σ_l = y, σ_r = x.
- Subgroup families: the 19 conjugacy classes; "small-index" = index ≤ 12: A5, S4, F20, A4, S3xS2, D10 (+ sign-twisted versions).
  Both coset sides (g*H and H*g) always considered.

## Statistics (per neuron, per side)
- S1 irrep energy profile: fraction of ||u||² in each isotypic V_ρ (basis-free; reproduces Chughtai's percent_* to ≤1e-3).
- S2 coset alignment α_H(u) = max over conjugates & sides of ||Q_H u||²/||u||².  Note α_H = 1 − C_H where C_H is Stander's
  coset-concentration (within-coset variance / total variance). Sign-twisted variant α_H^sgn with Q_H^sgn.
- S3 purity: for best (H*, ρ*), fraction of ||Q_{H*} u||² inside V_{ρ*}.
- Nulls: u uniform on sphere of V_ρ (pure-irrep null) → results/null_coset_scores.md. For mixed neurons: randomize direction
  within each V_ρ keeping the neuron's S1 energies (isotypic-rotation null).

## Bars (from the nulls, before looking)
- Coset-aligned := α_H ≥ 0.90 for some small-index family H (null p99 ≤ 0.83 for every index-≤12 family; null mass above 0.90 ≈ 0).
- Energy-weighted fraction of coset-aligned pre-activations, F_coset.

## Predictions & falsifiers
- P1 (replication, with null): F_coset ≥ 0.5 in every mainline seed. FALSIFIED if median F_coset < 0.5 → Stander's
  "almost every neuron" does not hold in Chughtai's models; nothing to dissolve on this substrate.
- P2 (controls): memorizers have F_coset at null level (< 0.05). FALSIFIED if memorizers are also coset-aligned →
  coset alignment is an artifact of the statistic, not of generalization.
- P3 (single-irrep purity of multi-irrep coset neurons — Stander's unexplained observation, App. G.2/Table 5):
  among coset-aligned neurons on D10, S3xS2, A4 (and ~sgn variants), purity ≥ 0.9 for ≥ 90% of their energy.
  FALSIFIED if such neurons routinely mix irreps (e.g. 5a and 5b on D10) → neurons are coset-first / irrep-agnostic
  and the induced-module picture (neuron ∈ Im Q_H ∩ V_ρ) is wrong.
- P4 (slack-1 signature): coset-aligned neurons on index-k subgroups with mult 1 take k distinct values with exactly one
  linear constraint (rank of coset-value vector = d_ρ = k − 1). Stander's eq. (1) (S4 neuron, values 4,2,0,−2,−4) is an instance.
- P5 (6d): vacuous on these models (6d energy ≈ 0 in all grokked runs). Report the λ₁-twisted F20 energy and C5-coset energy
  in the 6d block as a noise-level check only. The twisted-clock prediction is NOT tested here.
- P6 (irrep-selection contrast): document that Morwani's Thm (quadratic act., L_{2,3}) predicts 6d present for S5 while
  ReLU+CE+wd models never use it (0/59 grokked). Not a test; a framing fact.

## Decisions still open (user)
D1 object (pre-act per side vs post-ReLU hidden) · D2 bar (0.90 vs 0.95) · D3 include P3/P4 · D4 model set (MLP only vs +transformers)
D5 include memorizer controls · D6 dev/confirm seed split.
