"""DESCRIPTIVE (not pre-registered): Stander-style sweep — for block-dominant columns of a model, find the (subgroup H, linear
character lam, side) whose twisted-coset projector best explains each column.  Subgroups from 2-generator closures (index <= 64)."""
import json, argparse, numpy as np, pandas as pd, torch
from cosetprobe.affine import Affine
from cosetprobe.probe import preacts
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--blocks", nargs="*", type=int); ap.add_argument("--max_index", type=int, default=64); a = ap.parse_args()
cfg = json.load(open(f"{a.run}/cfg.json")); n = int(cfg["group"][3:]); A = Affine(n); N = A.n
sd = torch.load(f"{a.run}/model.pt", map_location="cpu", weights_only=True); Ux, Uy = preacts(sd)
# subgroups
seen = {}
for x in range(N):
    for y in range(x, N):
        H = tuple(A.closure([x, y]))
        if len(H) >= N // a.max_index and len(H) < N and H not in seen: seen[H] = np.array(H)
classes = []; done = set()
for H in sorted(seen.values(), key=lambda h: (-len(h), tuple(h))):
    if tuple(H) in done: continue
    conj = A.conjugates(H); done.update(conj); classes.append((H, [np.array(c) for c in conj]))
print(f"Aff(Z_{n}): {len(seen)} subgroups (index<={a.max_index}, 2-generated) in {len(classes)} conjugacy classes")
def describe(H):
    els = [A.els[i] for i in H]; ts = sorted({b for (a_, b) in els if a_ == 1}); us = sorted({a_ for (a_, b) in els})
    tmod = min([t for t in ts if t > 0], default=None); return f"|H|={len(H)} idx={N//len(H)} transl={('none' if tmod is None else f'{tmod}Z')} units={len(us)}"
fams = []
for H, conj in classes:
    lams = A.linear_characters(H)
    for li, lam in enumerate(lams):
        Qs = []
        for ci, Hc in enumerate(conj):
            # transport lam to the conjugate: lam_c(g h g^-1) = lam(h)
            g = [g for g in range(N) if set(A.conjugate(H, g).tolist()) == set(Hc.tolist())][0]
            lamc = np.zeros(N, complex); lamc[A.T[A.T[g, H], A.inverse[g]]] = lam[H]
            for side in ("left", "right"):
                Q = A.twisted_projector(Hc, lamc, side); Qs.append(np.asarray(Q))
        fams.append((f"{describe(H)} lam{li}", Qs))
print(f"{len(fams)} (subgroup-class, character) families, {sum(len(q) for _, q in fams)} projectors")
blocks = a.blocks or list(A.blocks_under_test().keys()); rows = []
for side, U in [("x", Ux), ("y", Uy)]:
    n2 = (U ** 2).sum(0); den = np.maximum(n2, 1e-300)
    E = np.stack([((A.isotypic_projector(r) @ U) ** 2).sum(0) for r in range(len(A.dims))]) / den; dom = E.argmax(0)
    for blk in blocks:
        r = A.block[blk]; cols = [j for j in range(U.shape[1]) if dom[j] == r and E[r, j] >= 0.9 and n2[j] > 1e-8 * n2.max()]
        if not cols: continue
        Uc = U[:, cols]; best = np.zeros(len(cols)); lab = [""] * len(cols)
        for name, Qs in fams:
            s = np.max([(np.abs(Q @ Uc) ** 2).sum(0) for Q in Qs], axis=0) / np.maximum((Uc ** 2).sum(0), 1e-300)
            for i in range(len(cols)):
                if s[i] > best[i] + 1e-6: best[i], lab[i] = s[i], name
        for i, j in enumerate(cols): rows.append(dict(side=side, block=blk, neuron=j, energy=n2[j], best_alpha=best[i], best_family=lab[i]))
R = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
for blk, g in R.groupby("block"):
    w = g.energy / g.energy.sum(); print(f"\n--- {a.run.split('/')[-1]} block d={blk}: {len(g)} columns; energy-weighted fraction with best alpha >= 0.9: {(w*(g.best_alpha>=0.9)).sum():.3f}; median best alpha {np.interp(0.5, np.cumsum(w.values[np.argsort(g.best_alpha.values)]), np.sort(g.best_alpha.values)):.3f}")
    print((g[g.best_alpha >= 0.9].groupby("best_family").energy.sum() / g.energy.sum()).sort_values(ascending=False).head(8).round(3).to_string())
