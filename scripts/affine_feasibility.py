"""Feasibility for plan (c): build Aff(Z_n), its character table (Burnside–Dixon), the divisor-block irreps,
and the nulls for the two competing descriptions of a neuron in the top block:
  PLAIN   : constant on cosets of a point-stabilizer M_x = {g : g(x)=x} (either side)     -> 'function of the translation only'
  TWISTED : f(g*h) = conj(psi_k(h)) f(g) for h in the translation subgroup T (either side) -> 'single Fourier frequency in b'
"""
import numpy as np, sys
from math import gcd

def affine(n):
    units = [a for a in range(1, n) if gcd(a, n) == 1]
    els = [(a, b) for a in units for b in range(n)]; idx = {g: i for i, g in enumerate(els)}; N = len(els)
    # g = (a,b) : x -> a x + b ;  mul(g,h) = g∘h (apply h, then g) = (a_g a_h, a_g b_h + b_g)
    T = np.array([[idx[((a1 * a2) % n, (a1 * b2 + b1) % n)] for (a2, b2) in els] for (a1, b1) in els])
    return els, idx, T, units

def group_data(T):
    N = len(T); e = [i for i in range(N) if all(T[i, j] == j for j in range(N))][0]
    inv = np.array([int(np.where(T[i] == e)[0][0]) for i in range(N)])
    conj = T[T[np.arange(N)[:, None], np.arange(N)[None, :]], inv[:, None]]       # conj[s,g] = s g s^-1
    cls = -np.ones(N, int); classes = []
    for g in range(N):
        if cls[g] < 0:
            members = np.unique(conj[:, g]); cls[members] = len(classes); classes.append(members)
    return e, inv, conj, cls, classes

def character_table(T, cls, classes):
    """Burnside–Dixon via simultaneous diagonalization of class-multiplication matrices."""
    N = len(T); K = len(classes); sizes = np.array([len(c) for c in classes])
    M = np.zeros((K, K, K))
    for i, Ci in enumerate(classes):
        for j, Cj in enumerate(classes):
            prod = T[np.ix_(Ci, Cj)].ravel(); counts = np.bincount(cls[prod], minlength=K)
            M[i, j] = counts / sizes                                            # a_ijk: C_i C_j = sum_k a_ijk C_k
    rng = np.random.default_rng(0); A = sum(rng.standard_normal() * M[i] for i in range(K))
    w, V = np.linalg.eig(A)                                                     # right eigenvectors: omega(C_i) omega(C_j) = sum_k a_ijk omega(C_k)
    chars = []
    for v in V.T:
        v = v / v[0]                                                            # omega(C_0)=1 on the identity class
        omega = v                                                               # omega_chi(C_i) = |C_i| chi(C_i)/chi(1)
        chi_over_d = omega / sizes
        d = np.sqrt(N / (sizes * np.abs(chi_over_d) ** 2).sum())
        chars.append(np.round(d * chi_over_d, 8))
    chars = np.array(sorted(chars, key=lambda c: (c[0].real, [(-x.real, -x.imag) for x in c])))
    G = (chars * sizes) @ chars.conj().T / N
    assert np.allclose(G, np.eye(K), atol=1e-6), "character table failed orthogonality"
    assert abs((chars[:, 0].real ** 2).sum() - N) < 1e-6
    return chars

def projector_isotypic(chi_class, cls, T, inv):
    N = len(T); d = chi_class[0].real; chi = chi_class[cls]
    return d / N * chi[T[np.arange(N)[:, None], inv[None, :]]]

def twisted_projector(T, H, lam, side):
    N = len(T); P = np.zeros((N, N), complex)
    for h in H:
        tgt = T[:, h] if side == "left" else T[h, :]
        P[np.arange(N), tgt] += lam[h]
    return P / len(H)

def run(n, Nmc=4000):
    els, idx, T, units = affine(n); N = len(els); e, inv, conj, cls, classes = group_data(T)
    chars = character_table(T, cls, classes)
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    print(f"\n===== Aff(Z_{n}): order {N}, |units|={len(units)}, {len(classes)} classes/irreps, dims {sorted(int(round(c[0].real)) for c in chars)} =====")
    trans = np.array([idx[(1, b)] for b in range(n)])                            # translation subgroup T
    M0 = np.array([idx[(a, 0)] for a in units])                                  # stabilizer of x=0
    stabs = [np.array([i for i, (a, b) in enumerate(els) if (a * x + b) % n == x]) for x in range(n)]
    # induced-from-trivial of M0 = permutation rep on Z_n ; decompose
    def ind_char(H, lam):
        return lam[conj].sum(0) / len(H)
    def decompose(f):
        f_cls = np.array([f[c[0]] for c in classes]); sizes = np.array([len(c) for c in classes])
        m = (chars.conj() * sizes) @ f_cls / N; return {i: int(round(x.real)) for i, x in enumerate(m) if abs(x) > 1e-6}
    lam1 = np.zeros(N, complex); lam1[M0] = 1
    perm = decompose(ind_char(M0, lam1)); print("Ind_M(1) = permutation rep on Z_n decomposes as:", {f"irrep{i}(d={int(round(chars[i][0].real))})": m for i, m in perm.items()}, "| #divisors =", len(divisors))
    # twisted from translations: psi_k
    psi = lambda k: np.array([np.exp(2j * np.pi * k * b / n) if a == 1 else 0 for (a, b) in els])
    for k in [1, n // divisors[1] if len(divisors) > 2 else 1]:
        print(f"Ind_T(psi_{k}) =", {f"irrep{i}(d={int(round(chars[i][0].real))})": m for i, m in decompose(ind_char(trans, psi(k))).items()})
    # top block = irrep of dim phi(n) that appears in Ind_T(psi_1)
    top = [i for i, m in decompose(ind_char(trans, psi(1))).items()][0]; d = int(round(chars[top][0].real))
    P = projector_isotypic(chars[top], cls, T, inv).real; assert abs(np.trace(P) - d * d) < 1e-6
    print(f"top block: irrep{top}, dim {d}, V_rho dim {d*d}")
    plain = [twisted_projector(T, S, np.where(np.isin(np.arange(N), S), 1, 0).astype(complex), s).real for S in stabs for s in ["left", "right"]]
    korbit = [k for k in range(1, n) if gcd(k, n) == 1]
    twist = [twisted_projector(T, trans, psi(k), s) for k in korbit for s in ["left", "right"]]
    for name, Qs in [("PLAIN (stabilizer cosets)", plain), ("TWISTED (single frequency)", twist)]:
        dims = sorted({int(round(np.trace(Q @ P).real)) for Q in Qs})
        rng = np.random.default_rng(1); U = P @ rng.standard_normal((N, Nmc)); U /= np.linalg.norm(U, axis=0)
        best = np.max([((np.abs(Q @ U)) ** 2).sum(0) for Q in Qs], axis=0)
        print(f"  {name:28s}: {len(Qs)} subspaces of dim {dims} inside V_rho ({d*d}) | null mean {best.mean():.3f} p95 {np.percentile(best,95):.3f} p99 {np.percentile(best,99):.3f} P(>=0.9) {(best>=0.9).mean():.4f}")
    # are the two families genuinely different? overlap of a plain subspace with the best twisted one
    Qp = plain[0] @ P; Qt = [Q @ P for Q in twist]
    ov = max(np.linalg.norm(Q @ Qp) ** 2 / max(np.linalg.norm(Qp) ** 2, 1e-12) for Q in Qt)
    print(f"  max overlap (Frobenius) of a plain subspace with any twisted subspace: {ov:.3f}  (1 = identical, ~1/|family| = generic)")

for n in [int(x) for x in (sys.argv[1:] or ["15", "13", "16", "21"])]:
    run(n)
