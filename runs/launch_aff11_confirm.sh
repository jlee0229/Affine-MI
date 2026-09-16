#!/bin/zsh
cd "$(dirname "$0")/.."
P=${PYTHON:-python}
for s in 2 3 4 5; do $P scripts/train.py --group aff11 --recipe wu --seed $s --epochs 100000 --device cpu --log_every 250 --out runs/affine/aff11_wu100k_m128_seed$s > runs/logs/aff11_wu100k_seed$s.log 2>&1; done
echo AFF11_CONFIRM_DONE
