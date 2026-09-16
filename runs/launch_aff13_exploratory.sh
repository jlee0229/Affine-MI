#!/bin/zsh
# EXPLORATORY (not pre-registered): does Aff(Z_13) grok under R-Chughtai at m=128?  Launch only after the confirm queues finish.
cd "$(dirname "$0")/.."
P=${PYTHON:-python}
for s in 1 2; do $P scripts/train.py --group aff13 --recipe chughtai --seed $s --device mps --log_every 250 --out runs/affine/aff13_chughtai_m128_seed$s > runs/logs/aff13_chughtai_seed$s.log 2>&1; done
echo AFF13_EXPLORATORY_DONE
