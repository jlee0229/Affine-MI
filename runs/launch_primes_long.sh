#!/bin/zsh
cd "$(dirname "$0")/.."
P=${PYTHON:-python}
$P scripts/train.py --group aff11 --recipe wu --seed 1 --epochs 100000 --device cpu --log_every 250 --out runs/affine/aff11_wu100k_m128_seed1 > runs/logs/aff11_wu100k_seed1.log 2>&1
$P scripts/train.py --group aff13 --recipe wu --seed 1 --epochs 100000 --device cpu --log_every 250 --out runs/affine/aff13_wu100k_m128_seed1 > runs/logs/aff13_wu100k_seed1.log 2>&1
echo PRIMES_LONG_DONE
