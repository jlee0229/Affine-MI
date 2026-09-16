"""Group theory on an arbitrary multiplication table.  T[x, y] = index of x*y.
Character table by Burnside–Dixon; isotypic and twisted-coset projectors; induced characters; cosets; closure.
Functions on G are length-|G| arrays indexed by element index."""
import numpy as np


class Group:
    def __init__(self, table, names=None):
        self.T = np.asarray(table); self.n = len(self.T); self.names = names
        ar = np.arange(self.n)
        self.identity = int([i for i in range(self.n) if np.array_equal(self.T[i], ar)][0])
        self.inverse = np.array([int(np.where(self.T[i] == self.identity)[0][0]) for i in range(self.n)])
        self.conj = self.T[self.T[ar[:, None], ar[None, :]], self.inverse[:, None]]   # conj[s, g] = s*g*s^-1
        cls = -np.ones(self.n, int); self.classes = []
        for g in [self.identity] + [g for g in range(self.n) if g != self.identity]:   # identity class first (Burnside–Dixon normalizes on it)
            if cls[g] < 0:
                m = np.unique(self.conj[:, g]); cls[m] = len(self.classes); self.classes.append(m)
        self.class_id = cls; self.class_sizes = np.array([len(c) for c in self.classes])
        self.chars = self._character_table()                       # (K, K) complex, rows = irreps, cols = classes
        self.dims = np.round(self.chars[:, 0].real).astype(int)
        self.chi = [c[self.class_id] for c in self.chars]          # each irrep's character as a function on G

    # ---- character table (Burnside–Dixon) ----
    def _character_table(self):
        K = len(self.classes); N = self.n; sizes = self.class_sizes
        M = np.zeros((K, K, K))
        for i, Ci in enumerate(self.classes):
            for j, Cj in enumerate(self.classes):
                M[i, j] = np.bincount(self.class_id[self.T[np.ix_(Ci, Cj)].ravel()], minlength=K) / sizes
        rng = np.random.default_rng(0)
        for attempt in range(20):
            A = sum(rng.standard_normal() * M[i] for i in range(K))
            w, V = np.linalg.eig(A)
            if len(np.unique(np.round(w, 6))) == K: break
        chars = []
        for v in V.T:
            omega = v / v[0]; cod = omega / sizes
            d = np.sqrt(N / (sizes * np.abs(cod) ** 2).sum()); chars.append(d * cod)
        chars = np.array(sorted(chars, key=lambda c: (round(c[0].real, 6), tuple((-round(x.real, 6), -round(x.imag, 6)) for x in c))))
        G = (chars * sizes) @ chars.conj().T / N
        assert np.allclose(G, np.eye(K), atol=1e-6), "character table: orthogonality failed"
        assert abs((chars[:, 0].real ** 2).sum() - N) < 1e-6, "character table: sum d^2 != |G|"
        return chars

    # ---- projectors on functions ----
    def isotypic_projector(self, r):
        gg = self.T[np.arange(self.n)[:, None], self.inverse[None, :]]
        P = self.dims[r] / self.n * self.chi[r][gg]
        return P.real if np.allclose(P.imag, 0) else P

    def twisted_projector(self, H, lam, side="left"):
        """Projector onto {f : f(g*h) = conj(lam(h)) f(g), h in H} (left) or {f(h*g) = conj(lam(h)) f(g)} (right).
        lam: length-|G| array, a 1-d character of H (zero off H). lam == 1 on H gives the coset-constant projector."""
        P = np.zeros((self.n, self.n), complex)
        for h in H:
            tgt = self.T[:, h] if side == "left" else self.T[h, :]
            P[np.arange(self.n), tgt] += lam[h]
        P /= len(H)
        return P.real if np.allclose(P.imag, 0) else P

    def coset_projector(self, H, side="left"):
        lam = np.zeros(self.n); lam[H] = 1; return self.twisted_projector(H, lam, side)

    # ---- induced characters ----
    def induced_character(self, H, lam):
        return lam[self.conj].sum(axis=0) / len(H)

    def decompose(self, f, tol=1e-6):
        """Multiplicities of irreps in a class function f (length |G|)."""
        fc = np.array([f[c[0]] for c in self.classes])
        m = (self.chars.conj() * self.class_sizes) @ fc / self.n
        out = {}
        for i, x in enumerate(m):
            if abs(x) > tol:
                assert abs(x.imag) < 1e-6 and abs(x.real - round(x.real)) < 1e-6, (i, x); out[i] = int(round(x.real))
        return out

    def trivial_char(self, H):
        lam = np.zeros(self.n, complex); lam[H] = 1; return lam

    # ---- subgroups & cosets ----
    def closure(self, gens):
        S = set(int(g) for g in gens) | {self.identity}; frontier = list(S)
        while frontier:
            new = []
            for a in frontier:
                for g in gens:
                    b = int(self.T[a, g])
                    if b not in S: S.add(b); new.append(b)
            frontier = new
        return np.array(sorted(S))

    def cosets(self, H, side="left"):
        cid = -np.ones(self.n, int); blocks = []
        for g in range(self.n):
            if cid[g] >= 0: continue
            blk = self.T[g, H] if side == "left" else self.T[H, g]
            cid[blk] = len(blocks); blocks.append(np.array(sorted(blk)))
        return cid, blocks

    def conjugate(self, H, g): return np.array(sorted(self.T[self.T[g, H], self.inverse[g]]))
    def conjugates(self, H): return sorted({tuple(self.conjugate(H, g)) for g in range(self.n)})
    def is_subgroup(self, H): return set(self.T[np.ix_(H, H)].ravel().tolist()) == set(int(h) for h in H)

    def linear_characters(self, H):
        """All 1-d characters of the subgroup H (as length-|G| arrays, zero off H), via the character table of H itself."""
        H = np.asarray(sorted(H)); loc = {int(h): i for i, h in enumerate(H)}
        sub = Group(np.array([[loc[int(self.T[a, b])] for b in H] for a in H]))
        out = []
        for r in range(len(sub.chars)):
            if sub.dims[r] == 1:
                lam = np.zeros(self.n, complex); lam[H] = sub.chi[r]; out.append(lam)
        return out
