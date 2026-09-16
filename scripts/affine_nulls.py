from cosetprobe.affine import Affine
lines = ["# Nulls for the affine pre-registration (attached before training; Monte Carlo, no model data)", "",
         "u = random unit vector in V_rho (block under test). alpha_plain = max over 2n stabilizer-coset subspaces; alpha_twist = max over real single-frequency pair subspaces (both sides). N = 4000.", "",
         "| n | block d | dim rho | dim V_rho | plain: mean / p95 / p99 / P(>=0.9) | twisted: mean / p95 / p99 / P(>=0.9) | max plain–twisted overlap |", "|---|---|---|---|---|---|---|"]
for n in [11, 13, 15, 16, 21]:
    t0 = time.time(); A = Affine(n); A.selftest(); pl = A.plain_family(); tw = A.twisted_family(); rng = np.random.default_rng(1)
    for d, r in A.blocks_under_test().items():
        P = A.isotypic_projector(r); U = P @ rng.standard_normal((A.n, 4000)); U /= np.linalg.norm(U, axis=0)
        ap = np.max([((Q @ U) ** 2).sum(0) for _, Q in pl], axis=0); at = np.max([((Q @ U) ** 2).sum(0) for _, Q in tw], axis=0)
        Qp = pl[0][1] @ P; ov = max(np.linalg.norm(Q @ Qp) ** 2 / np.linalg.norm(Qp) ** 2 for _, Q in tw)
        lines.append(f"| {n} | {d} | {A.dims[r]} | {A.dims[r]**2} | {ap.mean():.3f} / {np.percentile(ap,95):.3f} / {np.percentile(ap,99):.3f} / {(ap>=0.9).mean():.4f} | {at.mean():.3f} / {np.percentile(at,95):.3f} / {np.percentile(at,99):.3f} / {(at>=0.9).mean():.4f} | {ov:.3f} |")
    print(f"n={n} done ({time.time()-t0:.1f}s)")
open("results/affine_nulls.md", "w").write("\n".join(lines)); print("\n".join(lines[5:]))
