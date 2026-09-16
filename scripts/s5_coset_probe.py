import sys, pickle, argparse, time, numpy as np, pandas as pd
from cosetprobe import S5, Chars, load_run
from cosetprobe.probe import Probe, summarize, SHORT
ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
G = S5(); C = Chars(G); classes = pickle.load(open("data/s5_subgroup_classes.pkl", "rb")); pr = Probe(G, C, classes)
print("families:", list(pr.bank), "| primary:", pr.primary)
t0 = time.time(); dfs = []
for run in a.runs:
    R = load_run(run); dfs.append(pr.analyze_model(R["sd"], run)); print(f"  {run} done ({time.time()-t0:.1f}s)")
df = pd.concat(dfs, ignore_index=True); df.to_csv(f"{a.out}_per_neuron.csv", index=False)
S = summarize(df); S.to_csv(f"{a.out}_per_model.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
print(S[["run", "n_live", "n_aligned", "frac_aligned", "F_coset", "F_coset_x", "F_coset_y", "median_alpha_primary", "families"]].to_string(index=False))
al = df[df.live & df.aligned]
print("\n--- aligned columns: energy-weighted family x dominant-irrep crosstab (all runs pooled) ---")
ct = al.pivot_table(index="family_primary", columns="dominant_irrep", values="energy", aggfunc="sum", fill_value=0)
ct = ct.rename(columns=SHORT); print((ct / ct.values.sum()).round(3).to_string())
print("\n--- unaligned live columns: dominant irrep, energy share ---")
un = df[df.live & ~df.aligned]; print((un.groupby("dominant_irrep").energy.sum() / df[df.live].energy.sum()).rename(SHORT).round(3).to_string())
print("\n--- alpha_primary distribution among live columns (energy-weighted deciles) ---")
w = df[df.live]; order = np.argsort(w.alpha_primary.values); cw = np.cumsum(w.energy.values[order]) / w.energy.sum()
print(" ".join(f"{q:.0%}:{w.alpha_primary.values[order][np.searchsorted(cw, q)]:.3f}" for q in [.1,.2,.3,.4,.5,.6,.7,.8,.9]))

if len(al)==0:
    print("\n(no aligned columns — diagnostics skipped)"); sys.exit(0)
print("\n--- DIAGNOSTICS ---")
print("coset side of best primary projector, by input (l = g*H, r = H*g), energy-weighted:")
al["side_of_H"] = al.proj_primary.str[-1]
print((al.pivot_table(index="side", columns="side_of_H", values="energy", aggfunc="sum", fill_value=0).pipe(lambda t: t.div(t.sum(1), axis=0))).round(3).to_string())
print("\nfor columns labelled A4 / D10 / S3xS2: alpha on the supergroup family (is it really the smaller subgroup?)")
for fam, sup in [("A4", "S4"), ("D10", "F20"), ("S3xS2", "S4")]:
    g = al[al.family_primary == fam]
    if len(g): print(f"  {fam}: n={len(g)}, median alpha_{sup}={g[f'alpha_{sup}'].median():.3f}, share with alpha_{sup}>=0.9: {(g[f'alpha_{sup}']>=0.9).mean():.2f}")
# one raw neuron: its values grouped by the claimed cosets
import re
from cosetprobe.probe import preacts
exs = al[al.family_primary.isin(["S4","F20"])].sort_values("energy", ascending=False)
if len(exs)==0: sys.exit(0)
ex = exs.iloc[0]
R = load_run(ex.run); Ux, Uy = preacts(R["sd"]); u = (Ux if ex.side == "x" else Uy)[:, int(ex.neuron)]
mm = re.match(r"(\w+?)(~sgn)?\[(\d+)\](l|r)", ex.proj_primary); fam, ci, sd_ = mm.group(1), int(mm.group(3)), mm.group(4)
cl = next(c for c in classes if c["name"] == fam); H = cl["conjugates"][ci]
cid, blocks = G.cosets(H, "left" if sd_ == "l" else "right")
print(f"\nexample: {ex.run} side={ex.side} neuron={int(ex.neuron)} family={ex.proj_primary} alpha={ex.alpha_primary:.6f} dominant={SHORT[ex.dominant_irrep]}")
for b, blk in enumerate(blocks): print(f"   coset {b}: mean={u[blk].mean():+.4f}  within-coset std={u[blk].std():.2e}  n={len(blk)}")
