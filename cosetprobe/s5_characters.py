"""Character table of S5 (Chughtai's irrep names), isotypic projectors, induced characters, twisted-coset projectors.
All projectors act on functions f: G -> C represented as length-120 arrays indexed by Chughtai's element index."""
import numpy as np
from .s5 import S5

# columns in CLASS_NAMES order: 1^5, 2, 2^2, 3, 3.2, 4, 5
CHAR_TABLE = {
    "trivial":       [1,  1,  1,  1,  1,  1,  1],
    "sign":          [1, -1,  1,  1, -1, -1,  1],
    "standard":      [4,  2,  0,  1, -1,  0, -1],   # (4,1)
    "standard_sign": [4, -2,  0,  1,  1,  0, -1],   # (2,1,1,1)
    "s5_5d_a":       [5, -1,  1, -1, -1,  1,  0],   # (2,2,1)
    "s5_5d_b":       [5,  1,  1, -1,  1, -1,  0],   # (3,2)
    "s5_6d":         [6,  0, -2,  0,  0,  0,  1],   # (3,1,1)
}
IRREPS = list(CHAR_TABLE)
DIMS = {k: v[0] for k, v in CHAR_TABLE.items()}
PARTITION = {"trivial": "(5)", "sign": "(1^5)", "standard": "(4,1)", "standard_sign": "(2,1^3)",
             "s5_5d_a": "(2,2,1)", "s5_5d_b": "(3,2)", "s5_6d": "(3,1,1)"}


class Chars:
    def __init__(self, G: S5):
        self.G = G
        self.chi = {k: np.array(v, dtype=float)[G.class_id] for k, v in CHAR_TABLE.items()}

    def inner(self, f, g):
        """(1/|G|) sum_x f(x) conj(g(x))"""
        return np.vdot(g, f) / self.G.n

    def decompose(self, chi, tol=1e-9):
        m = {k: self.inner(chi, self.chi[k]) for k in IRREPS}
        for k, v in m.items():
            assert abs(v.imag) < tol and abs(v.real - round(v.real)) < tol, (k, v)
        return {k: int(round(v.real)) for k, v in m.items() if round(v.real) != 0}

    def isotypic_projector(self, name):
        """(Pf)(g) = d/|G| sum_h chi(h) f(h^-1 * g)   =>   P[g, g'] = d/|G| chi(g * g'^-1). Basis-free, side-free."""
        G = self.G
        gg = G.table[np.arange(G.n)[:, None], G.inverse[None, :]]
        return DIMS[name] / G.n * self.chi[name][gg]

    def induced_character(self, H, lam):
        """lam: length-|G| array, a 1-d character of H extended by 0 off H.  Ind(g) = (1/|H|) sum_s lam(s*g*s^-1)."""
        return lam[self.G.conj].sum(axis=0) / len(H)

    def twisted_projector(self, H, lam, side="left"):
        """Projector onto {f : f(g*h) = conj(lam(h)) f(g) for h in H}   (side='left', blocks g*H)
                       or {f : f(h*g) = conj(lam(h)) f(g)}              (side='right', blocks H*g).
        lam trivial => projector onto functions constant on cosets.  (Pf)(g) = (1/|H|) sum_h lam(h) f(g*h)."""
        G = self.G; P = np.zeros((G.n, G.n), dtype=complex)
        for h in H:
            tgt = G.table[:, h] if side == "left" else G.table[h, :]
            P[np.arange(G.n), tgt] += lam[h]
        return P / len(H)

    def coset_projector(self, H, side="left"):
        lam = np.zeros(self.G.n); lam[H] = 1.0
        return self.twisted_projector(H, lam, side).real

    def F20_characters(self):
        """The four linear characters lam_k of F20 = C5 x| C4 via the conjugation action on C5:
        h*c*h^-1 = c^m(h), m in F5^* = <2>;  lam_k(h) = i^(k * log_2 m(h)).  lam_0 trivial, lam_2 = sign|F20."""
        G = self.G; F20 = G.F20(); c = G.perm_from_cycles((0, 1, 2, 3, 4))
        powers = {}; x = G.identity
        for a in range(5): powers[x] = a; x = G.table[x, c]
        log2 = {1: 0, 2: 1, 4: 2, 3: 3}
        lams = {k: np.zeros(G.n, dtype=complex) for k in range(4)}
        for h in F20:
            m = powers[G.table[G.table[h, c], G.inverse[h]]]
            for k in range(4): lams[k][h] = 1j ** (k * log2[m])
        return lams

    def trivial_char(self, H):
        lam = np.zeros(self.G.n, dtype=complex); lam[H] = 1; return lam

    def sign_char(self, H):
        lam = np.zeros(self.G.n, dtype=complex); lam[H] = self.G.sign[H]; return lam
