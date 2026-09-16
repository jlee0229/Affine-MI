"""The bare experiment (pre-registration v1): coset alignment of per-side pre-activations.

For each neuron j and side s in {x, y}:  u = centered column of  W_s @ W[side block]  (a function on G).
S1  irrep energy profile   e_rho = ||P_rho u||^2 / ||u||^2
S2  coset alignment        alpha_F(u) = max over projectors Q in family F of ||Q u||^2 / ||u||^2
                           (family = all conjugates of H x both coset sides; plain or sign-twisted)
bar coset-aligned  <=>  max over PRIMARY families (index <= 12) of alpha_F >= 0.90
F_coset = energy-weighted fraction of aligned columns.
"""
import numpy as np, pandas as pd
from .s5_characters import IRREPS
from .coset_stats import coset_projector_bank

PRIMARY_INDEX = 12
BAR = 0.90
SHORT = {"trivial": "1", "sign": "sgn", "standard": "std", "standard_sign": "std'", "s5_5d_a": "5a", "s5_5d_b": "5b", "s5_6d": "6d"}


def preacts(sd):
    """Return (U_x, U_y): each (120, m), columns centered."""
    Wx, Wy, W = sd["W_x"].numpy().astype(np.float64), sd["W_y"].numpy().astype(np.float64), sd["W"].numpy().astype(np.float64)
    E = Wx.shape[1]
    Ux, Uy = Wx @ W[:E], Wy @ W[E:]
    return Ux - Ux.mean(0), Uy - Uy.mean(0)


class Probe:
    def __init__(self, G, chars, classes):
        self.G, self.C = G, chars
        bank = coset_projector_bank(G, chars, classes, max_cosets=30)
        # dedupe families whose projectors coincide (e.g. A5~sgn == A5): keep first by name
        seen = {}; self.bank = {}
        for k, Qs in bank.items():
            sig = np.round(sum(Q for _, Q in Qs), 6).tobytes()
            if sig in seen: continue
            seen[sig] = k; self.bank[k] = Qs
        self.stacks = {k: (np.stack([Q for _, Q in Qs]), [lab for lab, _ in Qs]) for k, Qs in self.bank.items()}
        self.index = {k: 120 // len(next(cl for cl in classes if cl["name"] == k.split("~")[0])["rep"]) for k in self.bank}
        self.primary = [k for k in self.bank if self.index[k] <= PRIMARY_INDEX]
        self.P = {r: chars.isotypic_projector(r) for r in IRREPS}

    def analyze(self, U, run, side, energy_floor=1e-8):
        n2 = (U ** 2).sum(0); tot = n2.sum(); m = U.shape[1]
        rows = [dict(run=run, side=side, neuron=j, energy=n2[j], energy_frac=n2[j] / tot) for j in range(m)]
        live = n2 > energy_floor * n2.max()
        for r in IRREPS:
            e = ((self.P[r] @ U) ** 2).sum(0) / np.maximum(n2, 1e-300)
            for j in range(m): rows[j][f"e_{SHORT[r]}"] = e[j]
        for j in range(m):
            rows[j]["dominant_irrep"] = max(IRREPS, key=lambda r: rows[j][f"e_{SHORT[r]}"])
        # families visited in order of INCREASING index (largest subgroup first); a later family must beat the
        # incumbent by > TIE to take over, so a neuron constant on S4-cosets (hence also on A4-cosets) is labelled S4.
        TIE = 1e-6
        best_any = np.zeros(m); fam_any = [""] * m; best_pri = np.zeros(m); fam_pri = [""] * m; lab_pri = [""] * m
        for k in sorted(self.stacks, key=lambda k: (self.index[k], k)):
            Qs, labs = self.stacks[k]
            S = ((Qs @ U) ** 2).sum(1) / np.maximum(n2, 1e-300)   # (P, m)
            a = S.max(0); am = S.argmax(0)
            for j in range(m):
                rows[j][f"alpha_{k}"] = a[j]
                if a[j] > best_any[j] + TIE: best_any[j], fam_any[j] = a[j], k
                if k in self.primary and a[j] > best_pri[j] + TIE: best_pri[j], fam_pri[j], lab_pri[j] = a[j], k, labs[am[j]]
        for j in range(m):
            rows[j].update(alpha_primary=best_pri[j], family_primary=fam_pri[j], proj_primary=lab_pri[j],
                           alpha_any=best_any[j], family_any=fam_any[j], live=bool(live[j]),
                           aligned=bool(live[j] and best_pri[j] >= BAR))
        return pd.DataFrame(rows)

    def analyze_model(self, sd, run):
        Ux, Uy = preacts(sd)
        return pd.concat([self.analyze(Ux, run, "x"), self.analyze(Uy, run, "y")], ignore_index=True)


def summarize(df):
    out = []
    for run, g in df.groupby("run", sort=False):
        live = g[g.live]
        rec = dict(run=run, n_cols=len(g), n_live=len(live), n_aligned=int(live.aligned.sum()),
                   frac_aligned=live.aligned.mean(), F_coset=live.energy[live.aligned].sum() / live.energy.sum())
        for s in ["x", "y"]:
            gs = live[live.side == s]; rec[f"F_coset_{s}"] = gs.energy[gs.aligned].sum() / gs.energy.sum()
        fam = live[live.aligned].groupby("family_primary").energy.sum() / live.energy.sum()
        rec["families"] = " ".join(f"{k}:{v:.2f}" for k, v in fam.sort_values(ascending=False).items())
        rec["median_alpha_primary"] = live.alpha_primary.median()
        out.append(rec)
    return pd.DataFrame(out)
