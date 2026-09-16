"""DESCRIPTIVE (Stander-style, with nulls): for every block-dominant column (tested blocks + siblings) of every included model,
the best (subgroup class H, linear character lam, side) with index(H) <= MAX_INDEX. Reports per (model, block) the energy share
of each winning (H-class, character-type) and compares to a Monte-Carlo null."""
import json, glob, numpy as np, pandas as pd, torch
from pathlib import Path
from cosetprobe.affine import Affine
from cosetprobe.probe import preacts
MAX_INDEX = 16; BAR = 0.9; _fam = {}

def describe(A, H):
    n = A.nmod; els = [A.els[i] for i in H]; ts = sorted({b for (a, b) in els if a == 1}); us = sorted({a for (a, b) in els})
    stabx = [x for x in range(n) if set(A.stab[x].tolist()) == set(H.tolist())]
    stab8 = [(x, d) for d in A.divisors if d < n for x in range(n) if set(i for i, (a, b) in enumerate(A.els) if (a * x + b - x) % d == 0) == set(H.tolist())]
    if len(H) == A.n: return "G"
    if stabx: return "M_x (point stabilizer)"
    if stab8: return f"S_x,{stab8[0][1]} (stabilizer mod {stab8[0][1]})"
    if set(A.trans.tolist()) <= set(H.tolist()): return f"T⋊K |K|={len(us)}"
    tmod = min([t for t in ts if t > 0], default=None)
    return f"other |H|={len(H)} units={len(us)} transl={'none' if tmod is None else f'{tmod}Z'}"

def families(n):
    if n in _fam: return _fam[n]
    A = Affine(n); N = A.n; seen = {}
    for x in range(N):
        for y in range(x, N):
            H = tuple(A.closure([x, y]))
            if N // len(H) <= MAX_INDEX and len(H) < N and H not in seen: seen[H] = np.array(H)
    classes = []; done = set()
    for H in sorted(seen.values(), key=lambda h: (-len(h), tuple(h))):
        if tuple(H) in done: continue
        conj = A.conjugates(H); done.update(conj); classes.append((H, [np.array(c) for c in conj]))
    fams = []
    for H, conj in classes:
        lams = A.linear_characters(H); desc = describe(A, H)
        for li, lam in enumerate(lams):
            triv = np.allclose(lam[H], 1); Qs = []
            for Hc in conj:
                g = [g for g in range(N) if set(A.conjugate(H, g).tolist()) == set(Hc.tolist())][0]
                lamc = np.zeros(N, complex); lamc[A.T[A.T[g, H], A.inverse[g]]] = lam[H]
                for side in ("left", "right"): Qs.append((side[0], np.asarray(A.twisted_projector(Hc, lamc, side))))
            fams.append((f"{desc} idx={N//len(H)} {'triv' if triv else 'char'}", np.stack([Q for _, Q in Qs]), [s for s, _ in Qs]))
    _fam[n] = (A, fams); return _fam[n]

def best_family(U, fams):
    n2 = (U ** 2).sum(0); best = np.zeros(U.shape[1]); lab = [""] * U.shape[1]; side = [""] * U.shape[1]
    for name, Qs, sides in fams:
        S = (np.abs(np.einsum("pij,jm->pim", Qs, U)) ** 2).sum(1) / np.maximum(n2, 1e-300); a = S.max(0); am = S.argmax(0)
        for j in range(U.shape[1]):
            if a[j] > best[j] + 1e-6: best[j], lab[j], side[j] = a[j], name, sides[am[j]]
    return best, lab, side

if __name__ == "__main__":
    runs = sorted(d for d in glob.glob("runs/affine/*") if Path(d, "cfg.json").exists() and json.load(open(Path(d) / "cfg.json"))["final"]["test_acc"] >= 0.99)
    rows = []; nulls = {}
    for rd in runs:
        cfg = json.load(open(Path(rd) / "cfg.json")); n = int(cfg["group"][3:]); A, fams = families(n)
        blocks = {str(d): r for d, r in A.blocks_under_test().items()} | A.sibling_blocks()
        sd = torch.load(Path(rd) / "model.pt", map_location="cpu", weights_only=True); Ux, Uy = preacts(sd)
        for lab, r in blocks.items():
            P = A.isotypic_projector(r)
            if (n, lab) not in nulls:
                rng = np.random.default_rng(5); V = np.asarray(P @ rng.standard_normal((A.n, 400))).real; V /= np.linalg.norm(V, axis=0); b, _, _ = best_family(V, fams)
                nulls[(n, lab)] = (b.mean(), np.percentile(b, 99), (b >= BAR).mean())
            for side, U in [("x", Ux), ("y", Uy)]:
                n2 = (U ** 2).sum(0); e = (np.abs(P @ U) ** 2).sum(0).real / np.maximum(n2, 1e-300); cols = np.where((n2 > 1e-8 * n2.max()) & (e >= 0.9))[0]
                if len(cols) == 0: continue
                b, labs, sides = best_family(U[:, cols], fams)
                for c, bb, ll, ss in zip(cols, b, labs, sides): rows.append(dict(run=Path(rd).name, n=n, block=lab, side=side, neuron=int(c), energy=n2[c], alpha=bb, family=ll, coset_side=ss))
        print(f"  {Path(rd).name} done", flush=True)
    D = pd.DataFrame(rows); D.to_csv("results/affine/sweep_per_column.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
    out = []
    for (run, blk), g in D.groupby(["run", "block"]):
        n = int(g.n.iloc[0]); w = g.energy / g.energy.sum(); fam = (g[g.alpha >= BAR].groupby("family").energy.sum() / g.energy.sum()).sort_values(ascending=False)
        out.append(dict(run=run, block=blk, n_cols=len(g), frac_explained=(w * (g.alpha >= BAR)).sum(), null_mean=nulls[(n, blk)][0], null_p99=nulls[(n, blk)][1], null_P90=nulls[(n, blk)][2],
                        families=" | ".join(f"{k}:{v:.2f}" for k, v in fam.head(3).items())))
    S = pd.DataFrame(out); S.to_csv("results/affine/sweep_per_model.csv", index=False); print(S.round(3).to_string(index=False))
