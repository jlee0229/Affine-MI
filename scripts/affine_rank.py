"""DESCRIPTIVE (Wu-style): for block-dominant columns, the coefficient matrix A in the monomial (frequency) basis of the block,
its rank-1-ness r1 = s1^2/sum s^2, and for rank-1 columns the projective stabilizers of the two singular vectors under the
monomial action (g.v)_k = psi_k(t) v_{k a}.  Types: T (contains all translations) / M (a point stabilizer: order phi(n), no
translations) / I (intermediate: other) / 1 (trivial)."""
import json, argparse, numpy as np, pandas as pd, torch
from pathlib import Path
from cosetprobe.affine import Affine, phi
from cosetprobe.probe import preacts

def coef_matrix(f, A, d):
    """f: (n_units, n) array f[a_index, b]. Returns A_mat[k, k'] (phi(d) x phi(d)) = mean over {a : k' a = k} of phi_{k'}(a)."""
    n = A.nmod; units = A.units; F = [k for k in range(1, n) if np.gcd(k, n) == n // d]; idx = {k: i for i, k in enumerate(F)}
    Phi = np.fft.fft(f, axis=1) / n; M = np.zeros((len(F), len(F)), complex); cnt = np.zeros((len(F), len(F)))
    for ai, a in enumerate(units):
        for kp in F:
            k = (kp * a) % n; M[idx[k], idx[kp]] += Phi[ai, kp]; cnt[idx[k], idx[kp]] += 1
    return M / np.maximum(cnt, 1), F

def stabilizer_type(v, A, F, d):
    """Projective stabilizer of v in block d. In block d translations by multiples of d act trivially, so
    M-type (plain) = {g : g(x) = x mod d}, order phi(n)*(n/d); T-type (single frequency) contains all translations."""
    n = A.nmod; idx = {k: i for i, k in enumerate(F)}; v = v / np.linalg.norm(v); stab = []
    for gi, (a, t) in enumerate(A.els):
        w = np.array([np.exp(2j * np.pi * k * t / n) * v[idx[(k * a) % n]] for k in F])
        if abs(abs(np.vdot(v, w)) - 1) < 1e-4: stab.append((a, t))
    order = len(stab); trans = sorted(t for a, t in stab if a == 1); kernel = [t for t in range(n) if t % d == 0]
    if trans == list(range(n)): typ = "T" if order == n * (len(A.units) // phi(d)) else "T+"
    elif trans == kernel and order == len(A.units) * (n // d): typ = "M"
    elif order == len(kernel): typ = "1"
    else: typ = "I"
    return typ, order

ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
rows = []
for rd in a.runs:
    rd = Path(rd); cfg = json.load(open(rd / "cfg.json")); n = int(cfg["group"][3:]); A = Affine(n); nu = len(A.units)
    sd = torch.load(rd / "model.pt", map_location="cpu", weights_only=True); Ux, Uy = preacts(sd)
    for d, r in A.blocks_under_test().items():
        P = A.isotypic_projector(r)
        for side, U in [("x", Ux), ("y", Uy)]:
            n2 = (U ** 2).sum(0); e = (np.abs(P @ U) ** 2).sum(0).real / np.maximum(n2, 1e-300); live = n2 > 1e-8 * n2.max()
            for j in np.where(live & (e >= 0.9))[0]:
                f = np.asarray(P @ U[:, j]).real.reshape(nu, n); M, F = coef_matrix(f, A, d)
                s = np.linalg.svd(M, compute_uv=False); r1 = s[0] ** 2 / (s ** 2).sum(); rank = int((s ** 2 > 0.01 * (s ** 2).sum()).sum())
                u, sv, vh = np.linalg.svd(M); tu, ou = stabilizer_type(u[:, 0].conj(), A, F, d); tv, ov = stabilizer_type(vh[0].conj(), A, F, d)
                rows.append(dict(run=rd.name, n=n, block=d, side=side, neuron=int(j), energy=n2[j], e_block=e[j], r1=r1, rank=rank, stab_u=tu, ord_u=ou, stab_v=tv, ord_v=ov))
R = pd.DataFrame(rows); R.to_csv(f"{a.out}_rank_per_column.csv", index=False)
pd.set_option("display.width", 250)
out = []
for (run, blk), g in R.groupby(["run", "block"]):
    w = g.energy / g.energy.sum(); r1w = (w * (g.r1 >= 0.99)).sum()
    types = (g[g.r1 >= 0.99].assign(t=lambda x: x.stab_u + "/" + x.stab_v).groupby("t").energy.sum() / g.energy.sum()).sort_values(ascending=False)
    out.append(dict(run=run, block=blk, n_cols=len(g), frac_rank1=r1w, wmean_r1=(w * g.r1).sum(), rank_dist=" ".join(f"{k}:{v:.2f}" for k, v in (g.groupby('rank').energy.sum() / g.energy.sum()).sort_index().items()),
                    stabilizer_types_u_v=" ".join(f"{k}:{v:.2f}" for k, v in types.items())))
S = pd.DataFrame(out); S.to_csv(f"{a.out}_rank_per_model.csv", index=False); print(S.round(3).to_string(index=False))
