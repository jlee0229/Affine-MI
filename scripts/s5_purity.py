import pickle, numpy as np, pandas as pd
from cosetprobe import S5, Chars, load_run
from cosetprobe.probe import Probe, preacts, SHORT
TAG={"sign":"sgn","standard":"std","standard_sign":"stdp","s5_5d_a":"5a","s5_5d_b":"5b","s5_6d":"6d"}
G = S5(); C = Chars(G); classes = pickle.load(open("data/s5_subgroup_classes.pkl", "rb")); pr = Probe(G, C, classes)
Qmap = {lab: Q for k, Qs in pr.bank.items() for lab, Q in Qs}
ind = {cl["name"]: (cl["ind1"], cl["indsgn"]) for cl in classes}
def constituents(fam):
    name, tw = (fam.split("~") + ["1"])[:2]
    return [r for r in (ind[name][0] if tw == "1" else ind[name][1]) if r != "trivial"]
MULTI = [f for f in pr.primary if len(constituents(f)) >= 2]
print("multi-constituent primary families:", {f: [SHORT[r] for r in constituents(f)] for f in MULTI})
N = pd.concat([pd.read_csv(f"results/probe/{p}_per_neuron.csv") for p in ["dev", "confirm"]], ignore_index=True)
al = N[N.live & N.aligned]
rows = []; cache = {}
for run, g in al.groupby("run"):
    if run not in cache: cache[run] = preacts(load_run(run)["sd"])
    Ux, Uy = cache[run]
    for _, r in g.iterrows():
        u = (Ux if r.side == "x" else Uy)[:, int(r.neuron)]; v = Qmap[r.proj_primary] @ u; n2 = (v ** 2).sum()
        cs = {rho: ((pr.P[rho] @ v) ** 2).sum() / n2 for rho in constituents(r.family_primary)}
        rows.append(dict(run=run, side=r.side, neuron=int(r.neuron), family=r.family_primary, energy=r.energy, alpha=r.alpha_primary,
                         purity=max(cs.values()), top=SHORT[max(cs, key=cs.get)], **{f"c_{TAG[k]}": v_ for k, v_ in cs.items()}))
D = pd.DataFrame(rows); D.to_csv("results/probe/p3_per_column.csv", index=False)
pd.set_option("display.width", 220)
print("\n=== P3: purity of the coset part, per family (energy-weighted) ===")
out = []
for fam, g in D.groupby("family"):
    w = g.energy / g.energy.sum()
    out.append(dict(family=fam, n_cols=len(g), energy_share=g.energy.sum() / D.energy.sum(), frac_purity_ge_0_9=(w * (g.purity >= 0.9)).sum(),
                    frac_purity_ge_0_99=(w * (g.purity >= 0.99)).sum(), wmean_purity=(w * g.purity).sum(),
                    constituent_shares=" ".join(f"{c[2:]}:{(w * g[c]).sum():.2f}" for c in g.columns if c.startswith("c_") and g[c].notna().any()),
                    top_irrep_dist=" ".join(f"{k}:{v:.2f}" for k, v in (g.groupby('top').energy.sum() / g.energy.sum()).sort_values(ascending=False).items())))
S = pd.DataFrame(out).sort_values("energy_share", ascending=False); print(S.to_string(index=False))
m = D[D.family.isin(MULTI)]; w = m.energy / m.energy.sum(); frac = (w * (m.purity >= 0.9)).sum()
print(f"\nP3 statistic (multi-constituent families, {len(m)} columns, {m.energy.sum()/D.energy.sum():.1%} of aligned energy): fraction with purity >= 0.9 = {frac:.3f}")
print("VERDICT:", "PASS" if frac >= 0.9 else ("FAIL" if frac < 0.5 else "INCONCLUSIVE"))
print("\nper-family verdict at the same thresholds:")
for fam in MULTI:
    g = m[m.family == fam]
    if len(g): wf = g.energy / g.energy.sum(); f = (wf * (g.purity >= 0.9)).sum(); print(f"  {fam:10s} {f:.3f}  {'PASS' if f>=0.9 else ('FAIL' if f<0.5 else 'INCONCLUSIVE')}")
print("\nA4 columns: joint distribution of (c_std, c_stdp) — two irreps mixed within one neuron?")
a4 = m[m.family == "A4"]
if len(a4):
    wa = lambda col, g=a4: np.average(g[col], weights=g.energy)
    print(f"  n={len(a4)}; energy-weighted mean c_std={wa('c_std'):.3f} c_stdp={wa('c_stdp'):.3f} c_sgn={wa('c_sgn'):.3f}")
    o = np.argsort(a4.purity.values); cw = np.cumsum(a4.energy.values[o]) / a4.energy.sum()
    print("  purity deciles (energy-weighted):", " ".join(f"{q:.0%}:{a4.purity.values[o][np.searchsorted(cw, q)]:.2f}" for q in [.1,.25,.5,.75,.9]))
    print("  by seed: n cols, share of that seed's aligned energy, weighted-mean purity, mean(c_std, c_stdp, c_sgn)")
    for run, g in a4.groupby("run"):
        tot = D[D.run == run].energy.sum()
        print(f"    {run:16s} n={len(g):3d}  share={g.energy.sum()/tot:.2f}  purity={np.average(g.purity, weights=g.energy):.2f}  ({np.average(g.c_std,weights=g.energy):.2f}, {np.average(g.c_stdp,weights=g.energy):.2f}, {np.average(g.c_sgn,weights=g.energy):.2f})")
