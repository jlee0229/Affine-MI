"""S5 in Chughtai's indexing.

Element i is sympy-1.11.1 `SymmetricGroup(5)._elements[i]` (pinned in data/s5_elements_sympy1111.json).
The trained models predict table[x, y] = idx(perm_x * perm_y) with sympy's convention (apply x, then y).
Throughout this package `*` denotes THAT operation (the model's operation); it is the opposite of textbook
composition, so "g*H" here is the textbook right coset H∘g. We never use textbook composition.
"""
import json
import numpy as np
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
CLASS_NAMES = ["1^5", "2", "2^2", "3", "3.2", "4", "5"]
CLASS_SIZES = [1, 10, 15, 20, 20, 30, 24]
_CT_TO_CLASS = {(1, 1, 1, 1, 1): 0, (2, 1, 1, 1): 1, (2, 2, 1): 2, (3, 1, 1): 3, (3, 2): 4, (4, 1): 5, (5,): 6}


def _cycle_type(p):
    n = len(p); seen = [False] * n; ct = []
    for i in range(n):
        if not seen[i]:
            L = 0; j = i
            while not seen[j]:
                seen[j] = True; j = p[j]; L += 1
            ct.append(L)
    return tuple(sorted(ct, reverse=True))


class S5:
    def __init__(self):
        self.elements = [tuple(a) for a in json.load(open(DATA / "s5_elements_sympy1111.json"))]
        self.n = len(self.elements)
        self.index = {p: i for i, p in enumerate(self.elements)}
        self.table = np.load(DATA / "s5_mult_table.npy")
        self.identity = self.index[tuple(range(5))]
        self.inverse = np.array([int(np.where(self.table[i] == self.identity)[0][0]) for i in range(self.n)])
        self.cycle_types = [_cycle_type(p) for p in self.elements]
        self.class_id = np.array([_CT_TO_CLASS[ct] for ct in self.cycle_types])
        self.sign = np.array([(-1) ** (5 - len(ct)) for ct in self.cycle_types])
        ar = np.arange(self.n)
        self.conj = self.table[self.table[ar[:, None], ar[None, :]], self.inverse[:, None]]  # conj[s, g] = s*g*s^-1

    def mul(self, x, y): return self.table[x, y]

    def perm_from_cycles(self, *cycles):
        p = list(range(5))
        for c in cycles:
            for a, b in zip(c, c[1:] + c[:1]): p[a] = b
        return self.index[tuple(p)]

    def closure(self, gens):
        S = set(gens) | {self.identity}; frontier = list(S)
        while frontier:
            new = []
            for a in frontier:
                for g in gens:
                    b = int(self.table[a, g])
                    if b not in S: S.add(b); new.append(b)
            frontier = new
        return np.array(sorted(S))

    def cosets(self, H, side="left"):
        """side='left': blocks g*H ; side='right': blocks H*g.  Returns (coset id per element, list of blocks)."""
        cid = -np.ones(self.n, dtype=int); blocks = []
        for g in range(self.n):
            if cid[g] >= 0: continue
            blk = self.table[g, H] if side == "left" else self.table[H, g]
            cid[blk] = len(blocks); blocks.append(np.array(sorted(blk)))
        return cid, blocks

    def conjugate(self, H, g): return np.array(sorted(self.table[self.table[g, H], self.inverse[g]]))  # g*H*g^-1

    def normalizer(self, H):
        Hs = set(H.tolist())
        return np.array([g for g in range(self.n) if set(self.conjugate(H, g).tolist()) == Hs])

    def conjugates(self, H):
        """All distinct conjugates of H (as sorted tuples)."""
        return sorted({tuple(self.conjugate(H, g)) for g in range(self.n)})

    def stabilizer(self, pts): return np.array([i for i, p in enumerate(self.elements) if all(p[a] == a for a in pts)])

    def setwise_stabilizer(self, S):
        S = set(S); return np.array([i for i, p in enumerate(self.elements) if {p[a] for a in S} == S])

    # named subgroups
    def A5(self): return np.where(self.sign == 1)[0]
    def S4(self, i=4): return self.stabilizer([i])
    def C5(self): return self.closure([self.perm_from_cycles((0, 1, 2, 3, 4))])
    def F20(self): return self.normalizer(self.C5())
    def D10(self): return self.closure([self.perm_from_cycles((0, 1, 2, 3, 4)), self.perm_from_cycles((1, 4), (2, 3))])
    def S3xS2(self): return self.setwise_stabilizer([3, 4])
    def S3(self): return self.stabilizer([3, 4])
    def C4(self): return self.closure([self.perm_from_cycles((1, 2, 4, 3))])
    def S2(self): return self.closure([self.perm_from_cycles((0, 1))])
