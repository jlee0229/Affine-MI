# cosets vs irreps

Code, trained models and results for a pre-registered test of whether the neurons of one-hidden-layer MLPs trained on
finite-group multiplication are organized by cosets of subgroups (Stander et al., arXiv:2312.06581) or by irreducible
representations (Chughtai et al., arXiv:2302.03025; Wu et al., arXiv:2410.07476), on S5, D59/D61 and the affine groups
Aff(Z_n) for n = 11, 13, 15, 16, 21. The S5 and dihedral analyses run on Chughtai et al.'s released models; the affine
models were trained here. Pre-registrations were frozen before the corresponding models were analysed and are kept
unedited in `preregistration/`. The findings are written up in `writeup/cosets_irreps_s5.html` and summarized in
`results/probe/REPORT.md` (S5, dihedral) and `results/affine/REPORT_affine.md` (affine family).

## Setup

    uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
    git clone --depth 1 https://github.com/bilal-chughtai/rep-theory-mech-interp external/rep-theory-mech-interp   # for the S5/dihedral scripts
    .venv/bin/python -m pytest -q tests

Dependencies are numpy, pandas and torch. Everything ran on one M2 laptop (analyses in seconds, an affine model in
2–20 minutes; the exact training queues are `runs/launch_*.sh`).

## Reproducing the tables

| result | script | writes |
|---|---|---|
| S5 coset alignment, P1/P2 (`results/probe/REPORT.md`) | `scripts/s5_coset_probe.py --runs S5_MLP_seed1 ... --out results/probe/<phase>` | `results/probe/<phase>_per_{model,neuron}.csv` |
| random-init control | `scripts/s5_random_init_control.py` | `results/probe/randinit_*.csv` |
| single-irrep purity, P3 | `scripts/s5_purity.py` | `results/probe/p3_per_column.csv` |
| dihedral rank-1 pilot, P0 | `scripts/dihedral_rank1_pilot.py` | `results/probe/dihedral_pilot_*.csv` |
| affine nulls (attached to the pre-registration) | `scripts/affine_nulls.py` | `results/affine_nulls.md` |
| affine verdicts P-A–P-D (`results/affine/REPORT_affine.md`) | `scripts/affine_tally.py` | `results/affine/final_per_{model,column}.csv` |
| rank of block coefficient matrices | `scripts/affine_rank.py --runs runs/affine/* --out results/affine/all` | `results/affine/all_rank_*.csv` |
| which subgroup (index ≤ 16, all characters) | `scripts/affine_subgroup_sweep.py` | `results/affine/sweep_*.csv` |

`make s5` and `make affine` run these in order. The S5 subgroup ↔ irrep dictionary and the null distributions in
`results/*.md` are produced by the self-test and `scripts/affine_nulls.py`; none of them use model data.

## Models

`runs/affine/<group>_<recipe>_m<width>_seed<k>/` holds the final weights (`model.pt`, a state dict of W_x, W_y, W, W_U),
the config and the train/test curve of every affine run, including the ones that did not generalize. Architecture and
initialization are Chughtai et al.'s (embedding 256, hidden 128 unless `m512`, no bias, ReLU); the task is all |G|^2
ordered pairs with a 40% training split drawn by the seed. `wu` is Adam, lr 1e-2, weight decay 2e-4, 25k epochs
(100k for n = 11); `chughtai` is AdamW, lr 1e-3, weight decay 1.0, 250k epochs. `runs/validation/` is our retraining of
Chughtai's S5 seed 2, which reproduces their released model's statistics to two decimals. Group elements are indexed as
in `cosetprobe/affine.py`; the S5 indexing follows Chughtai's code (sympy 1.11.1 order, pinned in `data/`).

`scripts/train.py --group aff15 --recipe wu --seed 3 --out runs/affine/aff15_wu_m128_seed3` retrains one model.
Grokking of the prime groups at width 128 is fragile (Aff(Z_11)) or absent (Aff(Z_13)); details in the report.
