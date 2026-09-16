import numpy as np
from cosetprobe import S5, Chars, CHAR_TABLE, IRREPS, DIMS, CLASS_SIZES, load_run, hidden_acts, percent_hidden

G = S5(); C = Chars(G)
ok = lambda cond, msg: print(("PASS " if cond else "FAIL ") + msg) or cond
allok = True

# --- group ---
sizes = np.bincount(G.class_id, minlength=7).tolist()
allok &= ok(sizes == CLASS_SIZES, f"class sizes {sizes}")
allok &= ok(all(G.table[G.table[a, b], c] == G.table[a, G.table[b, c]] for a in range(0, 120, 7) for b in range(0, 120, 11) for c in range(0, 120, 13)), "associativity (sampled)")
allok &= ok(all(G.table[i, G.inverse[i]] == G.identity for i in range(120)), "inverses")
subs = {"A5": G.A5(), "S4": G.S4(), "F20": G.F20(), "D10": G.D10(), "C5": G.C5(), "S3xS2": G.S3xS2(), "S3": G.S3(), "C4": G.C4()}
for k, v in subs.items():
    closed = set(G.table[np.ix_(v, v)].ravel().tolist()) == set(v.tolist())
    allok &= ok(closed, f"{k} closed, order {len(v)}")
allok &= ok({len(subs[k]) for k in subs} == {60, 24, 20, 10, 5, 12, 6, 4}, "subgroup orders {60,24,20,10,5,12,6,4}")
allok &= ok(len(G.conjugates(G.F20())) == 6 and len(G.conjugates(G.S4())) == 5 and len(G.conjugates(G.C5())) == 6, "conjugate counts F20:6 S4:5 C5:6")
cid, blocks = G.cosets(G.F20(), "left"); allok &= ok(len(blocks) == 6 and all(len(b) == 20 for b in blocks), "F20 left cosets 6x20")

# --- characters ---
T = np.array([CHAR_TABLE[k] for k in IRREPS], dtype=float); w = np.array(CLASS_SIZES, dtype=float)
gram = (T * w) @ T.T / 120
allok &= ok(np.allclose(gram, np.eye(7)), "character table row orthogonality")
allok &= ok(sum(d * d for d in DIMS.values()) == 120, "sum d^2 = 120")
Ps = {r: C.isotypic_projector(r) for r in IRREPS}
allok &= ok(all(np.allclose(P @ P, P) and np.allclose(P, P.T) for P in Ps.values()), "isotypic projectors idempotent+symmetric")
allok &= ok(np.allclose(sum(Ps.values()), np.eye(120)), "isotypic projectors sum to identity")
allok &= ok(all(abs(np.trace(Ps[r]) - DIMS[r] ** 2) < 1e-9 for r in IRREPS), "isotypic ranks = d^2")

# --- induced characters (the verify_s5 claims) ---
lam = C.F20_characters(); F20 = G.F20()
allok &= ok(all(np.allclose(lam[k][F20] * lam[k][F20].conj(), 1) for k in range(4)), "F20 characters unimodular on F20")
# homomorphism check for lam_1 wrt the table op
h1 = all(abs(lam[1][G.table[a, b]] - lam[1][a] * lam[1][b]) < 1e-12 for a in F20 for b in F20)
allok &= ok(h1, "lam_1 is a homomorphism of F20 under the model's operation")
allok &= ok(np.allclose(lam[2][F20], G.sign[F20]), "lam_2 = sign restricted to F20")
def ind(H, lamv): return C.decompose(C.induced_character(H, lamv))
tests = {
    "Ind_A5(1)": (ind(G.A5(), C.trivial_char(G.A5())), {"trivial": 1, "sign": 1}),
    "Ind_S4(1)": (ind(G.S4(), C.trivial_char(G.S4())), {"trivial": 1, "standard": 1}),
    "Ind_S4(sgn)": (ind(G.S4(), C.sign_char(G.S4())), {"sign": 1, "standard_sign": 1}),
    "Ind_F20(lam0)": (ind(F20, lam[0]), {"trivial": 1, "s5_5d_a": 1}),
    "Ind_F20(lam1)": (ind(F20, lam[1]), {"s5_6d": 1}),
    "Ind_F20(lam2)": (ind(F20, lam[2]), {"sign": 1, "s5_5d_b": 1}),
    "Ind_F20(lam3)": (ind(F20, lam[3]), {"s5_6d": 1}),
    "Ind_C5(1)": (ind(G.C5(), C.trivial_char(G.C5())), {"trivial": 1, "sign": 1, "s5_5d_a": 1, "s5_5d_b": 1, "s5_6d": 2}),
    "Ind_S3xS2(1)": (ind(G.S3xS2(), C.trivial_char(G.S3xS2())), {"trivial": 1, "standard": 1, "s5_5d_b": 1}),
    "Ind_D10(1)": (ind(G.D10(), C.trivial_char(G.D10())), None),
}
for k, (got, exp) in tests.items():
    if exp is None: print(f"INFO {k} = {got}"); continue
    allok &= ok(got == exp, f"{k} = {got}")

# --- twisted projectors: the 6d 'clock' claim ---
for side in ["left", "right"]:
    Q0 = C.coset_projector(F20, side); Q1 = C.twisted_projector(F20, lam[1], side)
    allok &= ok(np.allclose(Q0 @ Q0, Q0) and abs(np.trace(Q0) - 6) < 1e-9, f"[{side}] F20-coset projector rank 6")
    allok &= ok(np.allclose(Q1 @ Q1, Q1) and abs(np.trace(Q1) - 6) < 1e-9, f"[{side}] F20 lam_1-twisted projector rank 6")
    allok &= ok(np.allclose(Ps["s5_6d"] @ Q0, 0), f"[{side}] F20-coset-constant functions carry ZERO 6d weight")
    allok &= ok(np.allclose((Ps["trivial"] + Ps["s5_5d_a"]) @ Q0, Q0), f"[{side}] F20-coset-constant functions live in trivial + (2,2,1)")
    allok &= ok(np.allclose(Ps["s5_6d"] @ Q1, Q1), f"[{side}] lam_1-twisted F20 functions live ENTIRELY in 6d")
    QC5 = C.coset_projector(G.C5(), side)
    allok &= ok(np.allclose(QC5 @ Q1, Q1), f"[{side}] twisted functions are constant on C5-cosets (clock on C5)")
    allok &= ok(np.allclose(Ps["standard"] @ QC5, 0) and np.allclose(Ps["standard_sign"] @ QC5, 0), f"[{side}] C5-coset-constant functions carry zero standard/standard_sign weight")

# --- pipeline validation vs Chughtai's own summary_metrics.json ---
print("\n--- percent_hidden replication (ours vs theirs) ---")
for run in ["S5_MLP_seed2", "S5_MLP_seed1", "S5_MLP_seed23", "S5_MLP_seed28", "S5_MLP_hidden_dim_64_seed1"]:
    R = load_run(run); H = hidden_acts(R["sd"]); ours = percent_hidden(H, G, C)
    maxdiff = 0.0
    for k, v in ours.items():
        theirs = R["summary"].get(k); maxdiff = max(maxdiff, abs(v - theirs))
    line = " ".join(f"{r.split('_')[-1] if r.startswith('s5') else r[:4]}={ours[f'total_percent_hidden_{r}_rep']:.3f}/{R['summary'][f'total_percent_hidden_{r}_rep']:.3f}" for r in IRREPS)
    allok &= ok(maxdiff < 1e-2, f"{run:28s} max|diff|={maxdiff:.1e}  explained={ours['percent_hidden_explained']:.3f}/{R['summary']['percent_hidden_explained']:.3f}\n      {line}")

print("\nALL PASS" if allok else "\nSOME FAILURES")
