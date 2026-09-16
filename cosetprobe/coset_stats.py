"""Coset-alignment score and its null distribution.

score_[H](u) = max over conjugates H' of H and sides in {left,right} of  ||Q_{H',side} u||^2 / ||u||^2,
where Q is the coset-constant projector (or sign-twisted variant) and u is a centered function on G.
Null: u uniform on the unit sphere of the rho-isotypic subspace V_rho (dim d^2), i.e. u = P_rho g / |P_rho g|, g Gaussian.
Analytic mean of ||Q u||^2 for a single (H', side): k/D with k = mult(rho in Ind_H(chi)) * d, D = d^2; the max over
conjugates/sides is what we tabulate empirically."""
import numpy as np
from .s5_characters import IRREPS


def coset_projector_bank(G, chars, classes, max_cosets=30, twists=("1", "sgn")):
    """Returns dict name -> list of (label, Q) for every conjugate x side x twist; name like 'S4', 'S4~sgn'."""
    bank = {}
    for cl in classes:
        if 120 // cl["order"] > max_cosets or cl["order"] == 120: continue
        for tw in twists:
            key = cl["name"] + ("" if tw == "1" else "~sgn")
            Qs = []
            for ci, H in enumerate(cl["conjugates"]):
                lam = chars.trivial_char(H) if tw == "1" else chars.sign_char(H)
                sides = ["left"] if cl["normal"] else ["left", "right"]
                for side in sides:
                    Q = chars.twisted_projector(H, lam, side).real
                    Qs.append((f"{key}[{ci}]{side[0]}", Q))
            bank[key] = Qs
    return bank


def best_scores(U, Qs):
    """U: (120, N) centered columns. Returns (max score per column, argmax label index)."""
    n2 = (U ** 2).sum(0)
    S = np.stack([((Q @ U) ** 2).sum(0) for _, Q in Qs], 0) / np.maximum(n2, 1e-30)
    return S.max(0), S.argmax(0)


def null_table(G, chars, bank, irreps=None, N=4000, seed=0):
    rng = np.random.default_rng(seed); irreps = irreps or [r for r in IRREPS if r != "trivial"]
    rows = []
    for r in irreps:
        P = chars.isotypic_projector(r)
        U = P @ rng.standard_normal((G.n, N)); U /= np.linalg.norm(U, axis=0)
        for key, Qs in bank.items():
            s, _ = best_scores(U, Qs)
            rows.append((r, key, len(Qs), s.mean(), np.percentile(s, 95), np.percentile(s, 99), s.max()))
    return rows
