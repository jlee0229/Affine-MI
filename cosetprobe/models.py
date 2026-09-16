import json, torch
from pathlib import Path

# Chughtai et al.'s released models (git clone https://github.com/bilal-chughtai/rep-theory-mech-interp external/rep-theory-mech-interp)
import os
RUNS_DIR = Path(os.environ.get("CHUGHTAI_RUNS", Path(__file__).resolve().parent.parent / "external/rep-theory-mech-interp/rep_theory/batch_experiments"))


def list_runs(prefix="S5_MLP"):
    return sorted([p.name for p in RUNS_DIR.iterdir() if p.name.startswith(prefix) and (p / "model.pt").exists()],
                  key=lambda s: (len(s), s))


def load_run(name):
    d = RUNS_DIR / name
    sd = torch.load(d / "model.pt", map_location="cpu", weights_only=True)
    cfg = json.load(open(d / "cfg.json"))
    summ = json.load(open(d / "summary_metrics.json"))
    key = open(d / "key_reps.txt").read().split()
    return dict(name=name, sd=sd, cfg=cfg, summary=summ, key_reps=key)


def all_pairs(n=120):
    X, Y = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    return X.reshape(-1), Y.reshape(-1)


def forward(sd, x, y):
    """OneLayerMLP forward (no bias). Returns logits (N, 120), hidden (N, m)."""
    e = torch.cat([sd["W_x"][x], sd["W_y"][y]], dim=-1)
    h = torch.relu(e @ sd["W"])
    return h @ sd["W_U"], h


@torch.no_grad()
def hidden_acts(sd):
    """Hidden activations on all 120x120 pairs as a numpy array of shape (120, 120, m) indexed [x, y, neuron]."""
    X, Y = all_pairs()
    _, h = forward(sd, X, Y)
    return h.numpy().reshape(120, 120, -1)
