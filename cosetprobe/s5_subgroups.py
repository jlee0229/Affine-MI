"""Enumerate all subgroups of S5 (every subgroup of S5 is 2-generated) and group them into conjugacy classes."""
import numpy as np
from .s5 import S5

NAMES_BY_ORDER = {1: "1", 2: "C2", 3: "C3", 4: "C4/V4", 5: "C5", 6: "S3/C6", 8: "D8", 10: "D10", 12: "A4/D12", 20: "F20", 24: "S4", 60: "A5", 120: "S5"}


def all_subgroups(G: S5):
    seen = {}
    for a in range(G.n):
        for b in range(a, G.n):
            H = tuple(G.closure([a, b]))
            if H not in seen: seen[H] = np.array(H)
    return list(seen.values())


def conjugacy_classes_of_subgroups(G: S5):
    subs = all_subgroups(G); classes = []; done = set()
    for H in sorted(subs, key=lambda h: (len(h), tuple(h))):
        key = tuple(H)
        if key in done: continue
        conj = G.conjugates(H)
        for c in conj: done.add(c)
        classes.append({"rep": H, "order": len(H), "n_conj": len(conj), "conjugates": [np.array(c) for c in conj],
                        "normal": len(conj) == 1})
    return classes


def name_class(G: S5, cl):
    H = cl["rep"]; o = cl["order"]; Hs = set(H.tolist())
    named = {"A5": G.A5(), "S4": G.S4(), "F20": G.F20(), "D10": G.D10(), "C5": G.C5(), "S3xS2": G.S3xS2(), "S3": G.S3(), "C4": G.C4()}
    for k, v in named.items():
        if set(v.tolist()) in [set(c.tolist()) for c in cl["conjugates"]]: return k
    # structural hints
    cts = sorted({G.cycle_types[h] for h in H})
    if o == 2: return "C2<(2)>" if (2, 1, 1, 1) in cts else "C2<(2,2)>"
    if o == 3: return "C3"
    if o == 4: return "V4" if (4, 1) not in cts else "C4"
    if o == 6: return "C6" if (3, 2) in cts else "S3'"
    if o == 8: return "D8"
    if o == 12: return "A4" if (2, 1, 1, 1) not in cts else "D12"
    if o == 1: return "1"
    if o == 120: return "S5"
    return f"order{o}"
