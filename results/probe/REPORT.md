# S5 coset-alignment probe — results (2026-09-12)

Pre-registration: `preregistration/s5_probe.md` (frozen before any model statistic; one labelling fix recorded after DEV).
Raw outputs: `results/probe/{dev,confirm,control,randinit}_per_{model,neuron}.csv`.

## Pre-registered verdicts

| phase | n | F_coset median | min | max | models ≥ 0.5 | models < 0.05 |
|---|---|---|---|---|---|---|
| DEV (seeds 1–10) | 10 | 1.000 | 0.979 | 1.000 | 10 | 0 |
| CONFIRM (seeds 11–50) | 40 | 1.000 | 0.903 | 1.000 | 40 | 0 |
| CONTROL (20 memorizers) | 20 | 0.125 | 0.000 | 0.537 | 1 | 6 |

- **P1 PASS.** Every generalizing model has ≥ 90% of its pre-activation energy in columns with coset alignment ≥ 0.90 (median 100%). Energy-weighted median α = 1.000; the 90th-percentile α among live columns rounds to 1.000. Stander's "almost every neuron" holds in Chughtai's models.
- **P2 FAIL at the pre-registered threshold.** Predicted F_coset < 0.05 in every control; observed median 0.125, 6/20 below 0.05. The separation between phases is nonetheless complete: max CONTROL 0.537 < min CONFIRM 0.903.

## Post-hoc analyses (NOT pre-registered; labelled as such)

1. **Control alignment is mostly the sign circuit.** 72% of the controls' aligned energy sits in sign-dominated columns (9.0% of total energy; non-sign aligned = 3.6%). A sign-irrep vector is A5-coset-constant by definition, and the sign circuit is the first to form (Stander §5.1). Excluding sign-dominated columns: CONTROL F_coset median 0.000, CONFIRM median 1.000.
2. **The rest tracks partial generalization.** Spearman(test_acc, F_coset) over controls = 0.76. Controls with test acc 0.05–0.08 contain fully formed S4 neurons (e.g. hidden_80_seed2 x-neuron 1: coset means +14.3, −13.1, −0.6, +5.6, −6.2, within-coset std 0.05).
3. **Random initialization (10 seeds, Chughtai's init) is the real null**: F_coset = 0.000 in all ten; median α 0.17; 99th-percentile α 0.26. Nothing approaches the 0.90 bar.

Reading: "memorizer" was the wrong null. Those models are partially trained (they carry the sign circuit and circuit fragments), not random functions. P2's threshold should have been set against the sign-excluded statistic or against untrained networks; either version passes trivially. This is recorded as a design error, not reinterpreted as a pass.

## Descriptive (matches the dictionary, both phases)

Energy-weighted family × dominant-irrep crosstab, CONFIRM: S4→std 0.673 · F20→5a 0.136 · A4→std′ 0.056 (+std 0.020) · A5→sgn 0.035 · S3xS2→5b 0.026 · D10→5a 0.024 · S3xS2~sgn→std′/sgn 0.012 · S4~sgn→std′ 0.005. Every nonzero cell is one Ind_H(1)/Ind_H(sgn) predicts; **F20~sgn is 0.000 in all 50 models** — 5b is always realized by plain S3×S2 cosets (10 cosets, slack 5), never by the sign-twisted F20 realization (6 cosets, slack 1).

Coset side: x-input aligns with H·g (96%), y-input with g·H (99.8%) — the translation of Stander's eq. (1) into the models' operation; an indexing error would have scrambled this.

"A4" columns (≈8% of energy; seeds 1, 6, 10, 18, 22, 23, 25, 27, 29, 32): genuinely A4-constant (median α_S4 0.02), i.e. functions of (S4-coset, parity) jointly; dominant irrep std′. Parked for P3 (purity).

Slack-1 signature visible in every S4 example: 5 coset values summing to ≈ 0 (e.g. −7.75, +7.75, −3.79, +3.79, 0.00). Parked for P4.

## P3 (purity; run 2026-09-15 on all 50 mainline seeds; `results/probe/p3_per_column.csv`)

**PASS at the pre-registered bar.** Among aligned columns on multi-constituent families (A4, S3×S2, D10, S3×S2~sgn, S4~sgn; 3,511 columns, 16.5% of aligned energy), 97.4% of energy has purity ≥ 0.9 in one irrep. Per family: D10 1.000, A4 0.951, S3×S2 0.986, S3×S2~sgn 1.000, S4~sgn 1.000. Stander's "coset neurons always concentrate on one irrep" holds in Chughtai's models, at the 90% level.

Wrinkle (stricter bar, not pre-registered): at purity ≥ 0.99 the same families pass only 34% (A4), 51% (S3×S2), 51% (S3×S2~sgn), 13% (S4~sgn), 98% (D10) — whereas S4, F20, A5 sit at 100% *by construction* (their Ind has one non-trivial constituent, so after centering purity ≡ 1). Every family where leakage is possible shows 1–10% leakage into the partner constituent: S4~sgn columns carry ~1% sign; A4 columns are ~97% one of std/std′ with ~3% of the other. So the "A4" columns are best read as slightly impure S4-type (plain or sign-twisted) neurons — enough impurity to tip the tie-break to A4, not enough to be mixed-irrep circuits. The seeds where they dominate (1, 10, 14, 18, 22, 23, 25, 27, 29, 32, 42, 50) are candidates for Wu et al.'s "roughly half we don't fully understand."

P4 withdrawn (see preregistration/s5_probe.md): its S4 content is the centering constraint; elsewhere it is P3.

## Step 0 — dihedral rank-1 pilot (P0; run 2026-09-15; `results/probe/dihedral_pilot_per_{model,column}.csv`)
**PASS.** Chughtai's 8 released D59/D61 MLPs (all test acc 1.0): 91–100% of live pre-activation energy is single-frequency
(≥ 0.9 in one 2-dim irrep ρ_k; the remainder is the sign irrep); among single-frequency columns, 100% of energy has
r1 = σ₁²/(σ₁²+σ₂²) ≥ 0.999 (5th percentile 0.99997). Random-2×2 null: mean 0.893, P(≥0.999) = 0.064. Wu et al.'s rank-one
observation holds on an M-group. The reflection-coset alignment sits at 0.998–0.9996 for real neurons and 0.93 for random ones —
the geometric vacuity predicted in docs/plan_affine.md §1; it says nothing about plain vs. twisted.
