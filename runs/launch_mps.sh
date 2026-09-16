#!/bin/zsh
cd "$(dirname "$0")/.."
P=${PYTHON:-python}
$P scripts/train.py --group s5 --recipe chughtai --seed 2 --device mps --log_every 250 --out runs/validation/s5_chughtai_m128_seed2 > runs/logs/s5_chughtai_seed2.log 2>&1
for s in 1 2; do $P scripts/train.py --group aff21 --recipe wu --seed $s --device mps --log_every 100 --out runs/affine/aff21_wu_m128_seed$s > runs/logs/aff21_wu_seed$s.log 2>&1; done
echo MPS_QUEUE_DONE
