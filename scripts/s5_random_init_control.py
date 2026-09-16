"""POST-HOC control (not pre-registered): untrained networks at Chughtai's init, same probe."""
import pickle, torch, numpy as np, pandas as pd
from cosetprobe import S5, Chars
from cosetprobe.probe import Probe, summarize
G = S5(); C = Chars(G); classes = pickle.load(open("data/s5_subgroup_classes.pkl", "rb")); pr = Probe(G, C, classes)
def init(seed, n=120, E=256, h=128):
    torch.manual_seed(seed)
    return {"W_x": torch.randn(n, E) / np.sqrt(E), "W_y": torch.randn(n, E) / np.sqrt(E), "W": torch.randn(2 * E, h) / np.sqrt(2 * E), "W_U": torch.randn(h, n) / np.sqrt(h)}
dfs = [pr.analyze_model(init(s), f"random_init_seed{s}") for s in range(1, 11)]
df = pd.concat(dfs, ignore_index=True); df.to_csv("results/probe/randinit_per_neuron.csv", index=False)
S = summarize(df); S.to_csv("results/probe/randinit_per_model.csv", index=False)
pd.set_option("display.width", 200)
print(S[["run", "n_live", "n_aligned", "F_coset", "median_alpha_primary"]].to_string(index=False))
print(f"\nrandom-init F_coset: median {S.F_coset.median():.4f}, max {S.F_coset.max():.4f}; median alpha_primary {S.median_alpha_primary.median():.3f}")
w = df[df.live]; order = np.argsort(w.alpha_primary.values); cw = np.cumsum(w.energy.values[order]) / w.energy.sum()
print("alpha_primary energy-weighted deciles:", " ".join(f"{q:.0%}:{w.alpha_primary.values[order][np.searchsorted(cw, q)]:.3f}" for q in [.1,.3,.5,.7,.9,.99]))
