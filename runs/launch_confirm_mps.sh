#!/bin/zsh
cd "$(dirname "$0")/.."
P=${PYTHON:-python}
for s in 1 2 3 4 5; do $P scripts/train.py --group aff15 --recipe chughtai --seed $s --device mps --log_every 250 --out runs/affine/aff15_chughtai_m128_seed$s > runs/logs/aff15_chughtai_seed$s.log 2>&1; done
for s in 3 4 5; do $P scripts/train.py --group aff21 --recipe wu --seed $s --device mps --log_every 100 --out runs/affine/aff21_wu_m128_seed$s > runs/logs/aff21_wu_seed$s.log 2>&1; done
echo CONFIRM_MPS_DONE
