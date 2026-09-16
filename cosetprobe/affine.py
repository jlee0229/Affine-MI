"""Aff(Z_n) = {x -> a x + b : a unit, b in Z_n}.  Index of (a, b) = ui*n + b, ui = position of a in the sorted units.
mul(g, h) = g∘h (apply h, then g) = (a_g a_h, a_g b_h + b_g).  Both coset sides are always tested downstream."""
from math import gcd
import numpy as np
from .groups import Group


def phi(d): return sum(1 for k in range(1, d + 1) if gcd(k, d) == 1)


class Affine(Group):
    def __init__(self, n):
        self.nmod = n; self.units = [a for a in range(1, n) if gcd(a, n) == 1]
        self.els = [(a, b) for a in self.units for b in range(n)]; self.index = {g: i for i, g in enumerate(self.els)}
        T = np.array([[self.index[((a1 * a2) % n, (a1 * b2 + b1) % n)] for (a2, b2) in self.els] for (a1, b1) in self.els])
        super().__init__(T)
        self.divisors = [d for d in range(1, n + 1) if n % d == 0]
        self.trans = np.array([self.index[(1, b)] for b in range(n)])
        self.stab = [np.array([i for i, (a, b) in enumerate(self.els) if (a * x + b) % n == x]) for x in range(n)]
        # blocks: rho_{d,1} = the irrep common to Ind_M(1) and Ind_T(psi_k) with gcd(k, n) = n/d
        perm = self.decompose(self.induced_character(self.stab[0], self.trivial_char(self.stab[0])))
        self.block = {}; self.orbit_irreps = {}
        for d in self.divisors:
            k = n // d; S = self.decompose(self.induced_character(self.trans, self.psi(k)))
            common = [r for r in S if r in perm]; assert len(common) == 1, (d, S, perm)
            self.block[d] = common[0]; self.orbit_irreps[d] = sorted(S)
        self.irrep_block = {r: d for d, r in self.block.items()}

    def psi(self, k):
        lam = np.zeros(self.n, complex); lam[self.trans] = np.exp(2j * np.pi * k * np.arange(self.nmod) / self.nmod); return lam

    def plain_family(self):
        """Coset-constant projectors of the n point stabilizers, both sides.  Labels 'M{x}{l|r}'."""
        return [(f"M{x}{s[0]}", self.coset_projector(self.stab[x], s)) for x in range(self.nmod) for s in ("left", "right")]

    def twisted_family(self):
        """Real single-frequency subspaces of the translation subgroup: Q_{psi_k} + Q_{psi_{-k}}, both sides. Labels 'k{k}{l|r}'."""
        n = self.nmod; out = []
        for k in range(1, n // 2 + 1):
            for s in ("left", "right"):
                Q = self.twisted_projector(self.trans, self.psi(k), s)
                if k != n - k: Q = Q + self.twisted_projector(self.trans, self.psi(n - k), s)
                Q = np.asarray(Q); assert np.allclose(Q.imag, 0) if np.iscomplexobj(Q) else True
                out.append((f"k{k}{s[0]}", Q.real if np.iscomplexobj(Q) else Q))
        return out

    def sibling_blocks(self, min_dim=4):
        """Character-twisted siblings rho_{d,chi}, chi != 1, of dim >= min_dim, labelled d{d}{greek}."""
        out = {}
        for d in self.divisors:
            for i, r in enumerate(self.orbit_irreps[d]):
                if r != self.block[d] and self.dims[r] >= min_dim: out[f"{d}{chr(945 + i)}"] = r
        return out

    def plainchar_family(self):
        """Character-twisted point-stabilizer projectors (M_x, lambda) for every non-trivial character lambda of the units,
        transported to each conjugate M_x by the translation (1, x); both sides. Labels 'M{x}c{i}{l|r}'."""
        n = self.nmod; M0 = self.stab[0]; lams = self.linear_characters(M0); out = []
        for li, lam in enumerate(lams):
            if np.allclose(lam[M0], 1): continue
            for x in range(n):
                g = self.index[(1, x)]; Hc = self.stab[x]; lamc = np.zeros(self.n, complex); lamc[self.T[self.T[g, M0], self.inverse[g]]] = lam[M0]
                assert set(self.T[self.T[g, M0], self.inverse[g]].tolist()) == set(Hc.tolist())
                for s in ("left", "right"):
                    out.append((f"M{x}c{li}{s[0]}", np.asarray(self.twisted_projector(Hc, lamc, s))))
        return out

    def blocks_under_test(self, min_dim=4):
        return {d: r for d, r in self.block.items() if self.dims[r] >= min_dim}

    def selftest(self):
        n = self.nmod; ok = True
        def chk(c, m): nonlocal ok; ok &= c; print(("PASS " if c else "FAIL ") + m)
        chk(int((self.dims ** 2).sum()) == self.n, f"n={n}: sum d^2 = |G| = {self.n}")
        chk(all(self.dims[self.block[d]] == phi(d) for d in self.divisors), f"n={n}: dim rho_d = phi(d) for d in {self.divisors}")
        perm = self.decompose(self.induced_character(self.stab[0], self.trivial_char(self.stab[0])))
        chk(sorted(perm) == sorted(self.block.values()) and all(v == 1 for v in perm.values()), f"n={n}: Ind_M(1) = ⊕_d rho_d, multiplicity-free")
        chk(self.decompose(self.induced_character(self.trans, self.psi(1))) == {self.block[n]: 1}, f"n={n}: Ind_T(psi_1) = rho_n exactly (monomial, zero slack)")
        for d in self.divisors:
            S = self.orbit_irreps[d]; chk(sum(self.dims[r] for r in S) == phi(n) and all(self.dims[r] == phi(d) for r in S), f"n={n}: Ind_T(psi_{n//d}) = {len(S)} irreps of dim phi({d})={phi(d)}, total phi(n)={phi(n)}")
        for d, r in self.blocks_under_test().items():
            P = self.isotypic_projector(r); dd = self.dims[r]
            pl = self.plain_family(); tw = self.twisted_family()
            chk(all(abs(np.trace(P @ Q @ P) - dd) < 1e-6 for _, Q in pl), f"n={n} block d={d}: every plain subspace meets V_rho in dim {dd}")
            ranks = [round(np.trace(P @ Q @ P).real) for _, Q in tw]
            chk(sorted(set(ranks)) in ([0, 2 * dd], [2 * dd]) or sorted(set(ranks)) == [0, dd, 2 * dd], f"n={n} block d={d}: twisted subspaces meet V_rho in dim 0 or {2*dd} (pair) — got {sorted(set(ranks))}")
            for s in ("l", "r"):
                tot = sum(P @ Q @ P for lab, Q in tw if lab.endswith(s)); chk(np.allclose(tot, P), f"n={n} block d={d}: twisted subspaces (side {s}) tile V_rho")
        print("ALL PASS" if ok else "SOME FAILURES"); return ok
