PYTHON ?= .venv/bin/python

.PHONY: env test s5 affine nulls train-affine

env:
	uv venv --python 3.11 .venv
	uv pip install --python $(PYTHON) -e ".[dev]"

test:
	$(PYTHON) -m pytest -q tests

# S5 probe on Chughtai's released models (needs external/rep-theory-mech-interp, see external/README.md)
s5:
	$(PYTHON) scripts/s5_coset_probe.py --out results/probe/dev --runs $(shell for i in $$(seq 1 10); do printf "S5_MLP_seed%d " $$i; done)
	$(PYTHON) scripts/s5_coset_probe.py --out results/probe/confirm --runs $(shell for i in $$(seq 11 50); do printf "S5_MLP_seed%d " $$i; done)
	$(PYTHON) scripts/s5_random_init_control.py
	$(PYTHON) scripts/s5_purity.py
	$(PYTHON) scripts/dihedral_rank1_pilot.py

# analyses of the trained affine models in runs/affine (weights are committed)
affine:
	$(PYTHON) scripts/affine_tally.py
	$(PYTHON) scripts/affine_rank.py --out results/affine/all --runs $(wildcard runs/affine/aff*_seed*)
	$(PYTHON) scripts/affine_subgroup_sweep.py

nulls:
	$(PYTHON) scripts/affine_nulls.py

# retrain the affine models (hours on a laptop; see runs/launch_*.sh for the exact queues that produced runs/affine)
train-affine:
	PYTHON=$(PYTHON) runs/launch_cpu.sh
	PYTHON=$(PYTHON) runs/launch_mps.sh
