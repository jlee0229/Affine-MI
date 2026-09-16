Third-party model releases used by the S5 and dihedral analyses. They are not vendored; clone them here:

    git clone --depth 1 https://github.com/bilal-chughtai/rep-theory-mech-interp external/rep-theory-mech-interp

That repository ships the final weights of every run in `rep_theory/batch_experiments/` (about 700 MB). The S5 scripts read
`S5_MLP_seed*`, `S5_MLP_hidden_dim_*`, `D59_MLP_seed*` and `D61_MLP_seed*` from there. To point the code somewhere else,
set `CHUGHTAI_RUNS=/path/to/batch_experiments`.

Stander et al.'s code (https://github.com/dashstander/sn-grok) is not needed by anything here; their weights are on Google Drive
(links in their notebook).
