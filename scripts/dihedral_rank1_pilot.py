"""Step 0: rank-1 pilot on Chughtai's released D59/D61 MLPs (pre-registered as P0 in preregistration/affine_family.md)."""
import json, numpy as np, pandas as pd, torch
from cosetprobe.groups import Group
from cosetprobe.models import RUNS_DIR
from cosetprobe.probe import preacts

from cosetprobe.dihedral import dihedral_table, rho

rng = np.random.default_rng(0); A = rng.standard_normal((200000, 2, 2)); sv = np.linalg.svd(A, compute_uv=False); r1null = sv[:, 0] ** 2 / (sv ** 2).sum(1)
print(f"null (random 2x2): mean r1 {r1null.mean():.3f}, P(r1>=0.99) {(r1null>=0.99).mean():.4f}, P(r1>=0.999) {(r1null>=0.999).mean():.5f}")
rows = []; summary = []
for n in [59, 61]:
    G = Group(dihedral_table(n)); two = [r for r in range(len(G.dims)) if G.dims[r] == 2]
    kof = {}
    for r in two:   # identify k by the character on the rotation r^1 (element 1): 2cos(2 pi k / n)
        val = G.chi[r][1].real; kof[r] = int(round(np.argmin([abs(val - 2 * np.cos(2 * np.pi * k / n)) for k in range(1, (n + 1) // 2)]) + 1))
    assert sorted(kof.values()) == list(range(1, (n + 1) // 2)), "frequency identification failed"
    P = {r: G.isotypic_projector(r) for r in range(len(G.dims))}; R = {r: rho(n, kof[r]) for r in two}
    refl = [np.array([G.identity, n + j]) for j in range(n)]
    Qrefl = [G.coset_projector(H, s) for H in refl for s in ["left", "right"]]
    for seed in range(1, 5):
        run = f"D{n}_MLP_seed{seed}"; sd = torch.load(RUNS_DIR / run / "model.pt", map_location="cpu", weights_only=True)
        summ = json.load(open(RUNS_DIR / run / "summary_metrics.json"))
        Ux, Uy = preacts(sd)
        for side, U in [("x", Ux), ("y", Uy)]:
            n2 = (U ** 2).sum(0); live = n2 > 1e-8 * n2.max()
            E = {r: ((P[r] @ U) ** 2).sum(0) / np.maximum(n2, 1e-300) for r in range(len(G.dims))}
            arefl = np.max([((Q @ U) ** 2).sum(0) for Q in Qrefl], axis=0) / np.maximum(n2, 1e-300)
            for j in range(U.shape[1]):
                if not live[j]: continue
                dom = max(E, key=lambda r: E[r][j]); edom = E[dom][j]
                rec = dict(run=run, n=n, side=side, neuron=j, energy=n2[j], dom_dim=int(G.dims[dom]), dom_k=kof.get(dom, 0), e_dom=edom, single=(edom >= 0.9 and G.dims[dom] == 2), alpha_refl=arefl[j])
                if G.dims[dom] == 2:
                    Ak = np.einsum("g,gab->ab", U[:, j], R[dom]); s_ = np.linalg.svd(Ak, compute_uv=False); rec["r1"] = s_[0] ** 2 / (s_ ** 2).sum()
                rows.append(rec)
        d = pd.DataFrame([r for r in rows if r["run"] == run]); sg = d[d.single]
        w = sg.energy / sg.energy.sum()
        summary.append(dict(run=run, test_acc=summ["test_acc"], n_live=len(d), frac_energy_single_freq=sg.energy.sum() / d.energy.sum(),
                            frac_energy_1d=d[d.dom_dim == 1].energy.sum() / d.energy.sum(), n_freqs=sg.dom_k.nunique(),
                            r1_ge_0_999=(w * (sg.r1 >= 0.999)).sum(), r1_ge_0_99=(w * (sg.r1 >= 0.99)).sum(), r1_wmedian=np.interp(0.5, np.cumsum(w.values[np.argsort(sg.r1.values)]), np.sort(sg.r1.values)),
                            alpha_refl_wmedian=np.interp(0.5, np.cumsum(w.values[np.argsort(sg.alpha_refl.values)]), np.sort(sg.alpha_refl.values))))
D = pd.DataFrame(rows); D.to_csv("results/probe/dihedral_pilot_per_column.csv", index=False)
S = pd.DataFrame(summary); S.to_csv("results/probe/dihedral_pilot_per_model.csv", index=False)
pd.set_option("display.width", 220); print(S.round(4).to_string(index=False))
sg = D[D.single]; w = sg.energy / sg.energy.sum()
print(f"\nP0 statistic (all 8 models pooled, {len(sg)} single-frequency columns, {sg.energy.sum()/D.energy.sum():.1%} of live energy): fraction with r1 >= 0.999 = {(w*(sg.r1>=0.999)).sum():.3f}  ->", "PASS" if (w*(sg.r1>=0.999)).sum() >= 0.9 else "FAIL")
o = np.argsort(sg.r1.values); cw = np.cumsum(sg.energy.values[o]) / sg.energy.sum()
print("r1 energy-weighted deciles:", " ".join(f"{q:.0%}:{sg.r1.values[o][np.searchsorted(cw, q)]:.5f}" for q in [.05,.1,.25,.5,.75,.9]))
print("reflection-coset alpha (vacuity check) deciles:", " ".join(f"{q:.0%}:{np.sort(sg.alpha_refl.values)[np.searchsorted(np.cumsum(sg.energy.values[np.argsort(sg.alpha_refl.values)])/sg.energy.sum(), q)]:.4f}" for q in [.1,.5,.9]))
ns = D[~D.single & (D.dom_dim == 2)]; print(f"\nnon-single-frequency 2d-dominant columns: {len(ns)} ({ns.energy.sum()/D.energy.sum():.1%} of energy); their e_dom deciles: {np.percentile(ns.e_dom,[10,50,90]).round(3) if len(ns) else '-'}")
