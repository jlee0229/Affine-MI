"""Pre-registered statistics S1–S3 and P-A verdicts for Aff(Z_n) models (preregistration/affine_family.md).
usage: scripts/affine_probe.py --runs DIR [DIR ...] --out PREFIX      (each DIR has model.pt + cfg.json from train.py)
       scripts/affine_probe.py --randinit 11 13 15 16 21 --seeds 10 --out PREFIX"""
import json, argparse, numpy as np, pandas as pd, torch
from pathlib import Path
from cosetprobe.affine import Affine
from cosetprobe.probe import preacts
BAR = 0.90; _cache = {}

def support_orbits(A):
    """For each block d: list of (orbit O ∪ -O as tuple, projector = sum of fixed-frequency (right) twisted projectors over it),
    for O ranging over orbits of subgroups K of the units acting on the block's frequencies. Descriptive only."""
    n = A.nmod; units = A.units
    subs = {tuple(sorted({1}))}
    for a in units:
        for b in units:
            K = {1}; frontier = [1]
            while frontier:
                new = []
                for k in frontier:
                    for g in (a, b):
                        h = (k * g) % n
                        if h not in K: K.add(h); new.append(h)
                frontier = new
            subs.add(tuple(sorted(K)))
    out = {}
    for d, r in A.blocks_under_test().items():
        F = [k for k in range(1, n) if np.gcd(k, n) == n // d]; seen = {}
        for K in subs:
            done = set()
            for k in F:
                if k in done: continue
                O = sorted({(k * g) % n for g in K} | {(-k * g) % n for g in K}); done.update(O)
                if tuple(O) in seen: continue
                Q = sum(A.twisted_projector(A.trans, A.psi(kk), "right") for kk in O); seen[tuple(O)] = np.asarray(Q).real
        out[d] = sorted(seen.items(), key=lambda kv: len(kv[0]))
    return out

def group(n):
    if n not in _cache:
        A = Affine(n); _cache[n] = dict(A=A, P={r: A.isotypic_projector(r) for r in range(len(A.dims))}, plain=A.plain_family(), twist=A.twisted_family(), orbits=support_orbits(A), plainchar=A.plainchar_family())
    return _cache[n]

def analyze(sd, n, run):
    g = group(n); A = g["A"]; Ux, Uy = preacts(sd); rows = []
    for side, U in [("x", Ux), ("y", Uy)]:
        n2 = (U ** 2).sum(0); live = n2 > 1e-8 * n2.max(); m = U.shape[1]; den = np.maximum(n2, 1e-300)
        E = np.stack([(np.abs(g["P"][r] @ U) ** 2).sum(0) for r in range(len(A.dims))]).real / den
        Sp = np.stack([((Q @ U) ** 2).sum(0) for _, Q in g["plain"]]) / den; St = np.stack([((Q @ U) ** 2).sum(0) for _, Q in g["twist"]]) / den
        dom, ip, it = E.argmax(0), Sp.argmax(0), St.argmax(0)
        # descriptive: smallest frequency-support orbit capturing >= 0.9 (fixed-frequency side), per block
        supp = np.zeros(m, int); supp_lab = [""] * m
        for d, orbs in g["orbits"].items():
            r = A.block[d]; cols = np.where(dom == r)[0]
            if len(cols) == 0: continue
            for O, Q in orbs:
                sc = ((Q @ U[:, cols]) ** 2).sum(0) / den[cols]
                for c, v in zip(cols, sc):
                    if supp[c] == 0 and v >= 0.9: supp[c] = len(O); supp_lab[c] = ",".join(map(str, O))
        for j in range(m):
            r = int(dom[j])
            rows.append(dict(run=run, n=n, side=side, neuron=j, energy=n2[j], live=bool(live[j]), dom_irrep=r, dom_dim=int(A.dims[r]),
                             dom_block=A.irrep_block.get(r, 0), e_dom=E[r, j], block_dominant=bool(E[r, j] >= 0.9),
                             alpha_plain=Sp[ip[j], j], plain_label=g["plain"][ip[j]][0], alpha_twist=St[it[j], j], twist_label=g["twist"][it[j]][0],
                             aligned_plain=bool(live[j] and Sp[ip[j], j] >= BAR), aligned_twist=bool(live[j] and St[it[j], j] >= BAR),
                             support_orbit=int(supp[j]), support_freqs=supp_lab[j]))
    return pd.DataFrame(rows)

def verdict(fp, ft):
    if np.isnan(fp): return "empty"
    if fp >= 0.5 and ft < 0.5: return "plain"
    if ft >= 0.5 and fp < 0.5: return "twist"
    if fp >= 0.5 and ft >= 0.5: return "both"
    if fp < 0.3 and ft < 0.3: return "neither"
    return "unclear"

def component_level(sd, n, run):
    """DESCRIPTIVE: for each block under test, project every live column onto V_rho and score the component (weight = component energy)."""
    g = group(n); A = g["A"]; Ux, Uy = preacts(sd); out = []
    blocks = {str(d): r for d, r in A.blocks_under_test().items()} | A.sibling_blocks()
    for lab, r in blocks.items():
        P = g["P"][r]; tot = 0.0; ep = 0.0; epc = 0.0; et = 0.0; wsum_p = 0.0; wsum_pc = 0.0; wsum_t = 0.0
        for side, U in [("x", Ux), ("y", Uy)]:
            n2 = (U ** 2).sum(0); live = n2 > 1e-8 * n2.max(); V = np.asarray(P @ U).real; c2 = (V ** 2).sum(0)
            keep = live & (c2 > 1e-4 * c2.max())
            if keep.sum() == 0: continue
            V = V[:, keep]; c2 = c2[keep]; den = np.maximum(c2, 1e-300)
            ap = np.max([(np.abs(Q @ V) ** 2).sum(0) for _, Q in g["plain"]], axis=0) / den
            apc = np.max([(np.abs(Q @ V) ** 2).sum(0) for _, Q in g["plainchar"]], axis=0) / den
            at = np.max([(np.abs(Q @ V) ** 2).sum(0) for _, Q in g["twist"]], axis=0) / den
            tot += c2.sum(); ep += c2[ap >= BAR].sum(); epc += c2[apc >= BAR].sum(); et += c2[at >= BAR].sum(); wsum_p += (c2 * ap).sum(); wsum_pc += (c2 * apc).sum(); wsum_t += (c2 * at).sum()
        f = lambda v: v / tot if tot > 0 else np.nan
        rec = dict(run=run, block=lab, comp_energy=tot, C_plain=f(ep), C_plainchar=f(epc), C_twist=f(et), wmean_alpha_plain=f(wsum_p), wmean_alpha_plainchar=f(wsum_pc), wmean_alpha_twist=f(wsum_t))
        rec["verdict_component"] = verdict(max(rec["C_plain"], rec["C_plainchar"]) if tot > 0 else np.nan, rec["C_twist"]); out.append(rec)
    return pd.DataFrame(out)

def summarize(df):
    out = []
    for run, g in df.groupby("run", sort=False):
        n = int(g.n.iloc[0]); A = group(n)["A"]; live = g[g.live]
        for d, r in A.blocks_under_test().items():
            b = live[live.block_dominant & (live.dom_irrep == r)]; tot = b.energy.sum()
            f = lambda mask: b.energy[mask].sum() / tot if tot > 0 else np.nan
            wmed = lambda col: (np.interp(0.5, np.cumsum(b.energy.values[np.argsort(b[col].values)]) / tot, np.sort(b[col].values)) if tot > 0 else np.nan)
            side = lambda col: " ".join(f"{k}:{v:.2f}" for k, v in (b.groupby(b[col].str[-1]).energy.sum() / tot).items()) if tot > 0 else ""
            rec = dict(run=run, n=n, block=d, dim=int(A.dims[r]), n_cols=len(b), energy_share=tot / live.energy.sum(),
                       F_plain=f(b.aligned_plain), F_twist=f(b.aligned_twist), F_both=f(b.aligned_plain & b.aligned_twist),
                       med_alpha_plain=wmed("alpha_plain"), med_alpha_twist=wmed("alpha_twist"),
                       plain_side=side("plain_label"), twist_side=side("twist_label"))
            rec["verdict"] = verdict(rec["F_plain"], rec["F_twist"])
            rec["support_orbit_dist"] = " ".join(f"{k}:{v:.2f}" for k, v in (b.groupby("support_orbit").energy.sum() / tot).sort_index().items()) if tot > 0 else ""
            out.append(rec)
        # whole-model descriptive: energy by block (all irreps), fraction block-dominant
        eb = live.groupby("dom_block").energy.sum() / live.energy.sum()
        out[-1]["energy_by_block"] = " ".join(f"d{k}:{v:.2f}" for k, v in eb.sort_index().items()); out[-1]["frac_block_dominant"] = live.energy[live.block_dominant].sum() / live.energy.sum()
    return pd.DataFrame(out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="*", default=[]); ap.add_argument("--randinit", nargs="*", type=int, default=[]); ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--out", required=True)
    a = ap.parse_args(); dfs = []
    for rd in a.runs:
        rd = Path(rd); cfg = json.load(open(rd / "cfg.json")); n = int(cfg["group"][3:])
        sd = torch.load(rd / "model.pt", map_location="cpu", weights_only=True); df = analyze(sd, n, rd.name); df["test_acc"] = cfg["final"]["test_acc"]; dfs.append(df)
    for n in a.randinit:
        N = Affine(n).n
        for s in range(1, a.seeds + 1):
            torch.manual_seed(s); E, m = 256, 128
            sd = {"W_x": torch.randn(N, E) / np.sqrt(E), "W_y": torch.randn(N, E) / np.sqrt(E), "W": torch.randn(2 * E, m) / np.sqrt(2 * E), "W_U": torch.randn(m, N) / np.sqrt(m)}
            df = analyze(sd, n, f"randinit_aff{n}_seed{s}"); df["test_acc"] = np.nan; dfs.append(df)
    D = pd.concat(dfs, ignore_index=True); D.to_csv(f"{a.out}_per_column.csv", index=False)
    S = summarize(D)
    comps = []
    for rd in a.runs:
        rd = Path(rd); cfg = json.load(open(rd / "cfg.json")); n = int(cfg["group"][3:])
        comps.append(component_level(torch.load(rd / "model.pt", map_location="cpu", weights_only=True), n, rd.name))
    if comps:
        Cc = pd.concat(comps, ignore_index=True); S["block"] = S["block"].astype(str); S = S.merge(Cc, on=["run", "block"], how="outer").sort_values(["run", "block"])
    S.to_csv(f"{a.out}_per_model.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    print(S[["run", "block", "dim", "n_cols", "energy_share", "F_plain", "F_twist", "F_both", "med_alpha_plain", "med_alpha_twist", "verdict", "support_orbit_dist"]].round(3).to_string(index=False))
    if "C_plain" in S: print("\n--- DESCRIPTIVE component-level (every live column projected onto the block; weight = component energy) ---"); print(S[S.comp_energy > 1][["run", "block", "comp_energy", "C_plain", "C_plainchar", "C_twist", "wmean_alpha_plain", "wmean_alpha_plainchar", "wmean_alpha_twist", "verdict_component"]].round(3).to_string(index=False))
