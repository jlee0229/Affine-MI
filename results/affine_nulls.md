# Nulls for the affine pre-registration (attached before training; Monte Carlo, no model data)

u = random unit vector in V_rho (block under test). alpha_plain = max over 2n stabilizer-coset subspaces; alpha_twist = max over real single-frequency pair subspaces (both sides). N = 4000.

| n | block d | dim rho | dim V_rho | plain: mean / p95 / p99 / P(>=0.9) | twisted: mean / p95 / p99 / P(>=0.9) | max plain–twisted overlap |
|---|---|---|---|---|---|---|
| 11 | 11 | 10 | 100 | 0.196 / 0.251 / 0.283 / 0.0000 | 0.297 / 0.364 / 0.399 / 0.0000 | 0.200 |
| 13 | 13 | 12 | 144 | 0.160 / 0.202 / 0.230 / 0.0000 | 0.246 / 0.296 / 0.329 / 0.0000 | 0.167 |
| 15 | 5 | 4 | 16 | 0.508 / 0.682 / 0.764 / 0.0000 | 0.684 / 0.839 / 0.897 / 0.0085 | 0.500 |
| 15 | 15 | 8 | 64 | 0.266 / 0.342 / 0.392 / 0.0000 | 0.371 / 0.456 / 0.503 / 0.0000 | 0.250 |
| 16 | 8 | 4 | 16 | 0.491 / 0.674 / 0.745 / 0.0000 | 0.689 / 0.848 / 0.904 / 0.0110 | 0.500 |
| 16 | 16 | 8 | 64 | 0.248 / 0.325 / 0.369 / 0.0000 | 0.370 / 0.458 / 0.502 / 0.0000 | 0.250 |
| 21 | 7 | 6 | 36 | 0.341 / 0.459 / 0.523 / 0.0000 | 0.487 / 0.608 / 0.685 / 0.0000 | 0.333 |
| 21 | 21 | 12 | 144 | 0.168 / 0.208 / 0.236 / 0.0000 | 0.246 / 0.299 / 0.329 / 0.0000 | 0.167 |