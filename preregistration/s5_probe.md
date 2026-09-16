# Pre-registration v1 — S5 coset-alignment probe (FROZEN 2026-09-12, before any model statistic was computed)

Decisions (user, 2026-09-12): D1 pre-activations per side · D2 bar 0.90 · D3 bare experiment only (P1, P2; P3/P4 deferred) · D4 MLPs only · D5 memorizers as control · D6 develop on mainline seeds 1–10, confirm on 11–50; statistics frozen here.

## Objects
- Models: Chughtai `OneLayerMLP` S5. DEV = S5_MLP_seed1..10. CONFIRM = S5_MLP_seed11..50. CONTROL = the 20 hidden-dim-sweep runs with test acc < 0.10 (memorizers). Transformers excluded.
- Per neuron j and side s∈{x,y}: u = centered column of W_s @ W[side block] (120 values, a function on G). Both coset sides (g*H, H*g) always included in the max.
- PRIMARY families (index ≤ 12; sign-twisted variants kept only where distinct): A5, S4, S4~sgn, F20, F20~sgn, A4, S3xS2, S3xS2~sgn, D10.

## Statistics
- S1 irrep energy profile e_ρ = ‖P_ρ u‖²/‖u‖².
- S2 α_F(u) = max over projectors in family F of ‖Q u‖²/‖u‖². (α = 1 − Stander's C_H.)
- Live column: energy ≥ 1e-8 × max column energy in that side.
- Coset-aligned ⇔ live and max over PRIMARY families α_F ≥ 0.90.
- F_coset = Σ energy of aligned live columns / Σ energy of live columns (per model; also per side).

## Null (computed before any model data; results/null_union_primary.md)
Pure-irrep random unit vector, max over PRIMARY families: P(α ≥ 0.90) = 2e-4 (std), 1e-4 (std'), 0 (5a, 5b, 6d); sign = 1 by construction (any sign-irrep vector is A5-coset-constant — sign neurons are coset neurons by definition and are counted as such).

## Predictions and falsifiers
- P1: F_coset ≥ 0.5 in every DEV and CONFIRM model. Falsified if the median F_coset over CONFIRM is < 0.5.
- P2: F_coset < 0.05 in every CONTROL model. Falsified if the median over CONTROL is ≥ 0.05.
- Descriptive (not tests): energy-weighted crosstab of best family × dominant irrep among aligned columns, to be compared with the dictionary (S4→std, S4~sgn→std', F20→5a, F20~sgn/S3xS2→5b, A5→sgn); which side (g*H vs H*g) aligns for the x- and y-inputs.

## Procedure
1. Run DEV. Inspect only for pipeline bugs (NaNs, dead columns, side/indexing errors). No change to statistics or bar is permitted after this point except to fix a demonstrated bug, which must be recorded here.
2. Run CONFIRM and CONTROL. Report P1, P2 verbatim.

## Recorded change after DEV (2026-09-12)
Bug (labelling only): family attribution used strict '>' in dictionary order, so ties at α=1 between a subgroup and its
supergroup (A4⊂S4, D10⊂F20) were credited to the smaller subgroup. Fixed: families visited by increasing index, later
family must exceed incumbent by 1e-6. The aligned/not decision (max over primary families ≥ 0.90) is unaffected.

## P3 v2 (frozen 2026-09-15, before computation; P4 folded in)
For each aligned column (α_primary ≥ 0.90) with best primary projector Q (family F, conjugate, side): v = Q u (the coset part).
Constituent energies c_ρ(v) = ‖P_ρ v‖²/‖v‖² over the non-trivial irreps ρ in Ind_F (trivial is zero after centering).
purity(v) = max_ρ c_ρ(v). Multi-constituent families: A4 {sgn,std,std'}, S3xS2 {std,5b}, D10 {sgn,5a,5b}, S3xS2~sgn {sgn,std',5a}, S4~sgn {sgn,std'}.
(S4, F20, A5 have one non-trivial constituent — purity ≡ 1 — reported but not tested.)
P3: energy-weighted fraction of aligned energy on multi-constituent families with purity ≥ 0.9 is ≥ 0.9 (PASS) / < 0.5 (FAIL) / else inconclusive. Reported per family. All 50 mainline seeds; no dev/confirm split (statistic fully specified by the dictionary).
P4 (as drafted in v0) is withdrawn: for S4 its "one constraint" is the centering constraint; elsewhere it is P3's constituent decomposition.
