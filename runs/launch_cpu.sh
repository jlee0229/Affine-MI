#!/bin/zsh
cd "$(dirname "$0")/.."
P=${PYTHON:-python}
for g in aff11 aff13 aff15 aff16; do for s in 1 2; do $P scripts/train.py --group $g --recipe wu --seed $s --device cpu --log_every 100 --out runs/affine/${g}_wu_m128_seed$s > runs/logs/${g}_wu_seed$s.log 2>&1; done; done
echo CPU_QUEUE_DONE
