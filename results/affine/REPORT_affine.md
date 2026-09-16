# Aff(ℤₙ) family — results (2026-09-16)

Pre-registration: `preregistration/affine_family.md` (frozen 2026-09-15; amendments 1–2 and dev-phase items 1–10 recorded there).
Nulls: `results/affine_nulls.md`, `results/affine_nulls_plainchar.md`. Raw: `results/affine/final_per_{model,column}.csv`,
`results/affine/sweep_per_{model,column}.csv`, `results/affine/all_rank_per_{model,column}.csv`. Models: `runs/affine/*`.

## Models
35 trained, 31 included (test acc ≥ 0.99). Excluded: Aff(ℤ₁₃) ×3 (never groks under R-Wu at m=128, up to 100k epochs; Amendment 2)
and Aff(ℤ₁₁) seed 1 at 25k (memorized; the same seed groks at 100k — Aff(ℤ₁₁) is at a grokking threshold, item 5).
Cells: R-Wu m=128 — n=11 (100k; 6 models incl. one 25k), 15, 16, 21 (25k; 5 seeds each); R-Chughtai n=15 ×5; R-Wu m=512 n=15 ×5.
Composite n grok by epoch 300–800; n=11 by epoch 4,000–30,500. Trainer validated against Chughtai's released S5 seed 2.

## Pre-registered verdicts

**P-A (which basis; block-dominant columns, frozen families, bar 0.90)** — confirm seeds:
| n | block (dim) | plain | twist | neither | both | empty | verdict |
|---|---|---|---|---|---|---|---|
| 11 | 11 (10) | 3 | 0 | 0 | 0 | 0 | PASS — plain |
| 15 | 5 (4) | 2 | 0 | 0 | 0 | 1 | PASS — plain |
| 15 | 15 (8) | — | — | — | — | 3 | no test (CRT: top block unused) |
| 16 | 16 (8) | 2 | 0 | 1 | 0 | 0 | PASS — plain |
| 16 | 8 (4) | — | — | — | — | 3 | no test (sibling 8β used) |
| 21 | 7 (6) | 1 | 0 | 0 | 1 | 1 | unclear (1 plain, 1 both, 1 sibling) |
| 21 | 21 (12) | — | — | — | — | 3 | no test (CRT) |
Dev seeds agree (n=11 3 plain; n=15 d5 2 plain; n=16 top 1 plain 1 neither, d8 1 twist 1 neither; n=21 d7 2 plain).
Twisted (single-frequency) organization never carries a block-dominant majority in any tested block. Where it appears it is either
simultaneous with plain (a plain neuron whose b-profile is one cosine: n=21 s3 d7 'both'; n=16 s1 d8) or in a sibling block
(n=21 s5 7β: C_twist 0.81).

**P-B (slack)**: median F_plain(top block), confirm seeds: n=11 0.995 vs n=16 0.740 → gap 0.255 ≥ 0.2 → PASS as pre-registered.
Caveats: n=13 dropped; n=15/21 top blocks empty (CRT), so "composite" = n=16 only; the n=16 median is {1.00, 0.74, 0.00} — the
reduction is not a uniform loss of plain-ness but seed-dependent alternative organizations (below).

**P-C (regime, n=15)**: frozen verdicts agree in 3/5 seeds → inconclusive (PASS needs ≥4, FALSIFIED ≤2). Sibling-agnostic
(component-level, plain ∪ plain-with-character vs twisted) agreement 4/5. The disagreements are the choice of sibling irrep
(ρ₅ vs ρ₅⊗sgn), not a different organization. R-Chughtai seed 1's sibling block is plain-with-character AND 54% single-frequency —
the long-schedule/weight-decay-1 regime produces sparser b-profiles; the organization is the same.

**P-D (width, n=15)**: frozen verdicts agree in 3/5 → reported (not a pass); sibling-agnostic 5/5. Width does not change organization.

**Controls**: random-init F_plain = F_twist = 0 for all 50 models × blocks; α at the null means.

## Descriptive findings (recorded before confirm analysis where noted)
1. **CRT factoring.** Aff(ℤ₁₅) ≅ S₃ × F₂₀, Aff(ℤ₂₁) ≅ S₃ × Aff(F₇): every model leaves the top (tensor) block empty and solves the
   factors separately. A factored solution appears whenever the group is a direct product.
2. **Sibling choice.** Networks use a character-twisted sibling ρ_{d,χ} = ρ_d ⊗ χ instead of ρ_d in ~40% of cells (n=15: 5 of 15
   models on d=5; n=16: 3 of 5 on d=8; n=21: 2 of 5 on d=7). The organization inside the sibling is plain-with-character.
3. **Rank one.** The block coefficient matrix of every block-dominant column is rank one (r1 ≥ 0.99 for 100% of energy) in every
   model except Aff(ℤ₁₆) seed 2's top block (74% of energy rank 2). Wu et al.'s projected-ρ-set form holds almost universally.
4. **Which subgroup (uniform sweep, index ≤ 16, all linear characters, Monte-Carlo nulls; informative for blocks of dim ≥ 6).**
   n=11: point stabilizers, trivial character, 99–100% in all 6 models (null p99 0.29). n=21 d=7: mod-7 point stabilizer
   (45–69%) plus its index-2 subgroup (31–55%) — neurons mixing plain and plain-with-character, as the S5 'A4' columns did.
   n=16 top block: seed 1 point stabilizer (98%); seed 5 a non-conjugate complement of the translations with a character (54%);
   seed 3 a mixture of classes; seeds 2 and 4 explained by no coset structure (0%) — seed 2 is rank 2 (two frequency assignments on
   the two cosets of T⋊K′, K′ ≅ V₄), seed 4 is rank 1 with b supported on a K′-orbit of frequencies. The sibling 8β (seeds 3–5) is
   exactly constant (α = 1.00, trivial character) on cosets of an order-16 subgroup containing no point stabilizer (Aff(ℤ₁₆)'s
   2-torsion gives extra index-8 classes); all point-stabilizer families score 0 there.
5. **Sides.** x-input ↔ right cosets (a function of x⁻¹(0)), y-input ↔ left cosets (a function of y(0)); equal energy in the two
   halves of every grokked model. Twisted: LEFT = frequency permuted by a, RIGHT = fixed frequency.
6. **Grokking.** Composite n grok ~50× faster than n=11; n=13 never groks at m=128 under R-Wu (exploratory R-Chughtai run pending).

## One-paragraph reading
On the affine family the network almost always builds rank-one projected ρ-sets (Wu's form) whose stabilizer is a point
stabilizer, i.e. the plain coset organization Stander described on S5, even though an exact single-frequency (induced)
basis with zero slack is available in every block. The single-frequency basis appears only when a plain neuron's b-profile
happens to be one cosine, or in an occasional sibling block. What varies by seed is which exact realization the network
uses: which sibling irrep, which subgroup class (point stabilizer, character-twisted stabilizer, non-conjugate complement),
and, on the one composite group without CRT factoring, sometimes a structure that is not a coset structure at all (rank 2,
or a vector supported on a K′-orbit). The pre-registered dichotomy was answered in favour of plain cosets; the structure
worth writing about is in the cases the dichotomy did not name.

### Sweep addendum (all 31 included models; `results/affine/sweep_per_model.csv`)
n=21 d=7 (dim 6, null p99 0.68): confirm seeds 3, 5 — mod-7 point stabilizer 86% / 27% plus its index-2 subgroup 14% / 73%
(trivial character); the 7β siblings (seeds 4, 5) — 98–100% plain-with-character on that index-2 subgroup. R-Chughtai n=15 seeds 4–5:
99–100% on the mod-5 point stabilizer, cleaner than R-Wu's spread across nested classes (dim-4 block; nulls weak). Exploratory n=13
R-Chughtai: pending (seed 1 at 75k/250k, memorized so far).
