"""Dihedral group D_n in Chughtai's indexing: element i = r^(i mod n) s^(i div n); and the orthogonal 2-d irreps rho_k."""
import numpy as np


def dihedral_table(n):
    N = 2 * n; T = np.zeros((N, N), int)
    for x in range(N):
        xr, xs = x % n, x // n
        for y in range(N):
            yr, ys = y % n, y // n
            zr, zs = ((xr + yr) % n, ys) if xs == 0 else ((xr - yr) % n, (1 + ys) % 2)
            T[x, y] = zr + zs * n
    return T

def rho(n, k):
    R = np.zeros((2 * n, 2, 2))
    for i in range(n):
        c, s = np.cos(2 * np.pi * i * k / n), np.sin(2 * np.pi * i * k / n)
        R[i] = [[c, -s], [s, c]]; R[n + i] = [[c, s], [s, -c]]
    return R
