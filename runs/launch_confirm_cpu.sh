#!/bin/zsh
cd "$(dirname "$0")/.."
P=${PYTHON:-python}
for g in aff15 aff16; do for s in 3 4 5; do $P scripts/train.py --group $g --recipe wu --seed $s --device cpu --log_every 100 --out runs/affine/${g}_wu_m128_seed$s > runs/logs/${g}_wu_seed$s.log 2>&1; done; done
for s in 1 2 3 4 5; do $P scripts/train.py --group aff15 --recipe wu --seed $s --hidden 512 --device cpu --log_every 100 --out runs/affine/aff15_wu_m512_seed$s > runs/logs/aff15_wu_m512_seed$s.log 2>&1; done
echo CONFIRM_CPU_DONE
