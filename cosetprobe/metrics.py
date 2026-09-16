"""Basis-free replication of Chughtai's percent_hidden metric, plus generic irrep-energy decompositions.

Chughtai: hidden (14400 x m) centered per neuron; for irrep rho, energy in the spans of
  {rho(x)_ij}, {rho(y)_ij}, {rho(x*y)_ij}  as functions on G x G, via QR bases.
Those spans are the rho-isotypic subspaces of functions of x only / y only / x*y only, so we can use
central projectors: P_rho applied to the y-mean (x-part), x-mean (y-part), and z=x*y-mean (xy-part)."""
import numpy as np
from .s5_characters import IRREPS


def _xy_mean(H, G):
    """Average of H[x, y, :] over pairs with x*y = z, for each z. Returns (120, m)."""
    m = H.shape[-1]; out = np.zeros((G.n, m)); cnt = np.zeros(G.n)
    Z = G.table
    for x in range(G.n):
        np.add.at(out, Z[x], H[x]); np.add.at(cnt, Z[x], 1)
    return out / cnt[:, None]


def energy_by_irrep(H, G, chars, center=True):
    """H: (120, 120, m). Returns dict irrep -> (x_frac, y_frac, xy_frac, total_frac), fractions of total (centered) energy.
    Also returns the residual fraction (energy not in any x-only/y-only/xy-only irrep subspace)."""
    H = H.astype(np.float64)
    if center: H = H - H.reshape(-1, H.shape[-1]).mean(0)
    tot = (H ** 2).sum()
    fx = H.mean(1)              # (120, m) function of x
    fy = H.mean(0)              # (120, m) function of y
    fz = _xy_mean(H, G)         # (120, m) function of z = x*y
    out = {}; acc = 0.0
    for r in IRREPS:
        P = chars.isotypic_projector(r)
        ex = G.n * ((P @ fx) ** 2).sum() / tot
        ey = G.n * ((P @ fy) ** 2).sum() / tot
        ez = G.n * ((P @ fz) ** 2).sum() / tot
        out[r] = (ex, ey, ez, ex + ey + ez); acc += ex + ey + ez
    return out, 1.0 - acc


def percent_hidden(H, G, chars):
    e, resid = energy_by_irrep(H, G, chars)
    return {f"total_percent_hidden_{r}_rep": v[3] for r, v in e.items()} | {"percent_hidden_explained": 1 - resid}
