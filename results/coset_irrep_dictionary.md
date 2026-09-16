# S5 subgroup ↔ irrep dictionary

Ind_H(1) = irreps spanned by functions constant on cosets of H (either side).  Ind_H(sgn) = spanned by sign-twisted coset functions.
Rank of coset projector = |G/H| = number of cosets.

| # | H | order | #cosets | #conj | normal | Ind_H(1) | Ind_H(sgn|H) |
|---|---|---|---|---|---|---|---|
| 0 | 1 | 1 | 120 | 1 | Y | 1 + sgn + 4·std + 4·std' + 5·5a(2,2,1) + 5·5b(3,2) + 6·6d | 1 + sgn + 4·std + 4·std' + 5·5a(2,2,1) + 5·5b(3,2) + 6·6d |
| 1 | C2<(2,2)> | 2 | 60 | 15 |  | 1 + sgn + 2·std + 2·std' + 3·5a(2,2,1) + 3·5b(3,2) + 2·6d | 1 + sgn + 2·std + 2·std' + 3·5a(2,2,1) + 3·5b(3,2) + 2·6d |
| 2 | C2<(2)> | 2 | 60 | 10 |  | 1 + 3·std + std' + 2·5a(2,2,1) + 3·5b(3,2) + 3·6d | sgn + std + 3·std' + 3·5a(2,2,1) + 2·5b(3,2) + 3·6d |
| 3 | C3 | 3 | 40 | 10 |  | 1 + sgn + 2·std + 2·std' + 5a(2,2,1) + 5b(3,2) + 2·6d | 1 + sgn + 2·std + 2·std' + 5a(2,2,1) + 5b(3,2) + 2·6d |
| 4 | C4 | 4 | 30 | 15 |  | 1 + std + std' + 2·5a(2,2,1) + 5b(3,2) + 6d | sgn + std + std' + 5a(2,2,1) + 2·5b(3,2) + 6d |
| 5 | V4 | 4 | 30 | 15 |  | 1 + 2·std + 5a(2,2,1) + 2·5b(3,2) + 6d | sgn + 2·std' + 2·5a(2,2,1) + 5b(3,2) + 6d |
| 6 | V4 | 4 | 30 | 5 |  | 1 + sgn + std + std' + 2·5a(2,2,1) + 2·5b(3,2) | 1 + sgn + std + std' + 2·5a(2,2,1) + 2·5b(3,2) |
| 7 | C5 | 5 | 24 | 6 |  | 1 + sgn + 5a(2,2,1) + 5b(3,2) + 2·6d | 1 + sgn + 5a(2,2,1) + 5b(3,2) + 2·6d |
| 8 | C6 | 6 | 20 | 10 |  | 1 + std + std' + 5b(3,2) + 6d | sgn + std + std' + 5a(2,2,1) + 6d |
| 9 | S3 | 6 | 20 | 10 |  | 1 + 2·std + 5b(3,2) + 6d | sgn + 2·std' + 5a(2,2,1) + 6d |
| 10 | S3' | 6 | 20 | 10 |  | 1 + sgn + std + std' + 5a(2,2,1) + 5b(3,2) | 1 + sgn + std + std' + 5a(2,2,1) + 5b(3,2) |
| 11 | D8 | 8 | 15 | 15 |  | 1 + std + 5a(2,2,1) + 5b(3,2) | sgn + std' + 5a(2,2,1) + 5b(3,2) |
| 12 | D10 | 10 | 12 | 6 |  | 1 + sgn + 5a(2,2,1) + 5b(3,2) | 1 + sgn + 5a(2,2,1) + 5b(3,2) |
| 13 | A4 | 12 | 10 | 5 |  | 1 + sgn + std + std' | 1 + sgn + std + std' |
| 14 | S3xS2 | 12 | 10 | 10 |  | 1 + std + 5b(3,2) | sgn + std' + 5a(2,2,1) |
| 15 | F20 | 20 | 6 | 6 |  | 1 + 5a(2,2,1) | sgn + 5b(3,2) |
| 16 | S4 | 24 | 5 | 5 |  | 1 + std | sgn + std' |
| 17 | A5 | 60 | 2 | 1 | Y | 1 + sgn | 1 + sgn |
| 18 | S5 | 120 | 1 | 1 | Y | 1 | sgn |

## Which subgroup classes can carry each irrep via plain coset-constancy (ρ ⊂ Ind_H(1)), sorted by #cosets (smallest first = most 'coset-like')

- **1** (d=1): S5[1 cosets, mult 1], A5[2 cosets, mult 1], S4[5 cosets, mult 1], F20[6 cosets, mult 1], A4[10 cosets, mult 1], S3xS2[10 cosets, mult 1], D10[12 cosets, mult 1], D8[15 cosets, mult 1], C6[20 cosets, mult 1], S3[20 cosets, mult 1], S3'[20 cosets, mult 1], C5[24 cosets, mult 1], C4[30 cosets, mult 1], V4[30 cosets, mult 1], V4[30 cosets, mult 1], C3[40 cosets, mult 1], C2<(2)>[60 cosets, mult 1], C2<(2,2)>[60 cosets, mult 1], 1[120 cosets, mult 1]
- **sgn** (d=1): A5[2 cosets, mult 1], A4[10 cosets, mult 1], D10[12 cosets, mult 1], S3'[20 cosets, mult 1], C5[24 cosets, mult 1], V4[30 cosets, mult 1], C3[40 cosets, mult 1], C2<(2,2)>[60 cosets, mult 1], 1[120 cosets, mult 1]
- **std** (d=4): S4[5 cosets, mult 1], A4[10 cosets, mult 1], S3xS2[10 cosets, mult 1], D8[15 cosets, mult 1], C6[20 cosets, mult 1], S3[20 cosets, mult 2], S3'[20 cosets, mult 1], C4[30 cosets, mult 1], V4[30 cosets, mult 1], V4[30 cosets, mult 2], C3[40 cosets, mult 2], C2<(2)>[60 cosets, mult 3], C2<(2,2)>[60 cosets, mult 2], 1[120 cosets, mult 4]
- **std'** (d=4): A4[10 cosets, mult 1], C6[20 cosets, mult 1], S3'[20 cosets, mult 1], C4[30 cosets, mult 1], V4[30 cosets, mult 1], C3[40 cosets, mult 2], C2<(2)>[60 cosets, mult 1], C2<(2,2)>[60 cosets, mult 2], 1[120 cosets, mult 4]
- **5a(2,2,1)** (d=5): F20[6 cosets, mult 1], D10[12 cosets, mult 1], D8[15 cosets, mult 1], S3'[20 cosets, mult 1], C5[24 cosets, mult 1], C4[30 cosets, mult 2], V4[30 cosets, mult 1], V4[30 cosets, mult 2], C3[40 cosets, mult 1], C2<(2)>[60 cosets, mult 2], C2<(2,2)>[60 cosets, mult 3], 1[120 cosets, mult 5]
- **5b(3,2)** (d=5): S3xS2[10 cosets, mult 1], D10[12 cosets, mult 1], D8[15 cosets, mult 1], C6[20 cosets, mult 1], S3[20 cosets, mult 1], S3'[20 cosets, mult 1], C5[24 cosets, mult 1], C4[30 cosets, mult 1], V4[30 cosets, mult 2], V4[30 cosets, mult 2], C3[40 cosets, mult 1], C2<(2)>[60 cosets, mult 3], C2<(2,2)>[60 cosets, mult 3], 1[120 cosets, mult 5]
- **6d** (d=6): C6[20 cosets, mult 1], S3[20 cosets, mult 1], C5[24 cosets, mult 2], C4[30 cosets, mult 1], V4[30 cosets, mult 1], C3[40 cosets, mult 2], C2<(2)>[60 cosets, mult 3], C2<(2,2)>[60 cosets, mult 2], 1[120 cosets, mult 6]