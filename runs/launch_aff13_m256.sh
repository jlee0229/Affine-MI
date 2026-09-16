#!/bin/zsh
# EXPLORATORY: does Aff(Z_13) grok at m=256 (p^2 = 169 <= 256)?
cd "$(dirname "$0")/.."
P=${PYTHON:-python}
for s in 1 2; do $P scripts/train.py --group aff13 --recipe wu --seed $s --epochs 100000 --hidden 256 --device mps --log_every 250 --out runs/affine/aff13_wu100k_m256_seed$s > runs/logs/aff13_wu100k_m256_seed$s.log 2>&1; done
echo AFF13_M256_DONE
