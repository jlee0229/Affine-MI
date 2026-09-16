"""Final tally of the frozen predictions (preregistration/affine_family.md) over all trained affine models.
Runs the probe on every runs/affine/* model, then evaluates P-A, P-B, P-C, P-D exactly as pre-registered."""
import sys, json, glob, subprocess, numpy as np, pandas as pd
from pathlib import Path
runs = sorted(d for d in glob.glob("runs/affine/*") if Path(d, "cfg.json").exists())
meta = []
for d in runs:
    c = json.load(open(Path(d) / "cfg.json")); name = Path(d).name
    meta.append(dict(run=name, group=c["group"], n=c["n"], nmod=int(c["group"][3:]), recipe=c["recipe"], seed=c["seed"], hidden=c["hidden"], epochs=c["epochs"], test_acc=c["final"]["test_acc"], included=c["final"]["test_acc"] >= 0.99))
M = pd.DataFrame(meta); M.to_csv("results/affine/final_models.csv", index=False)
inc = M[M.included]
print(f"{len(M)} trained models; {len(inc)} included (test acc >= 0.99); excluded:"); print(M[~M.included][["run", "test_acc"]].to_string(index=False) if (~M.included).any() else "  none")
subprocess.run([sys.executable, "scripts/affine_probe.py", "--out", "results/affine/final", "--runs"] + [f"runs/affine/{r}" for r in inc.run], check=True, capture_output=True)
S = pd.read_csv("results/affine/final_per_model.csv"); S["block"] = S["block"].astype(str); S = S.merge(M, on="run")
S["phase"] = np.where(S.seed <= 2, "dev", "confirm"); S["cell"] = S.recipe + "_m" + S.hidden.astype(str) + "_ep" + S.epochs.astype(str)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 40)
main = S[(S.recipe == "wu") & (S.hidden == 128) & S.block.str.isdigit()]
print("\n=== P-A: verdicts per (n, block) — block-dominant columns, frozen families, bar 0.90 ===")
tab = main.pivot_table(index=["nmod", "block"], columns="phase", values="verdict", aggfunc=lambda v: " ".join(f"{k}:{c}" for k, c in v.value_counts().items()))
print(tab.to_string())
conf = main[main.phase == "confirm"]
print("\nP-A test (confirm seeds; per block: majority both<0.3 -> falsified-neither; majority both>=0.5 -> falsified-both):")
for (nm, blk), g in conf.groupby(["nmod", "block"]):
    if g.verdict.eq("empty").all(): print(f"  n={nm:2d} block {blk:>2s}: all EMPTY (energy in sibling / factor blocks) — no test"); continue
    g2 = g[g.verdict != "empty"]; neither = (g2.verdict == "neither").mean(); both = (g2.verdict == "both").mean(); one = g2.verdict.isin(["plain", "twist"]).mean()
    status = "FALSIFIED-neither" if neither > 0.5 else ("FALSIFIED-both" if both > 0.5 else ("PASS" if one > 0.5 else "unclear"))
    print(f"  n={nm:2d} block {blk:>2s}: {len(g2)} non-empty confirm models — plain {(g2.verdict=='plain').sum()}, twist {(g2.verdict=='twist').sum()}, neither {(g2.verdict=='neither').sum()}, both {(g2.verdict=='both').sum()} -> {status}")
print("\n=== P-B: median F_plain(top block) prime vs composite (confirm seeds; non-empty) ===")
top = conf[conf.apply(lambda r: r.block == str(r.nmod), axis=1) & (conf.verdict != "empty")]
med = top.groupby("nmod").F_plain.median(); print(med.round(3).to_string())
primes = [p for p in [11, 13] if p in med.index]; comps = [c for c in [15, 16, 21] if c in med.index]
if primes and comps:
    gap = np.median([med[p] for p in primes]) - np.median([med[c] for c in comps]); print(f"  prime median {np.median([med[p] for p in primes]):.3f} vs composite median {np.median([med[c] for c in comps]):.3f} -> gap {gap:.3f} -> {'PASS' if gap >= 0.2 else ('FALSIFIED' if gap < 0.05 else 'inconclusive')}")
else: print("  P-B untestable with current cells (top blocks empty for factored n; see amendments)")
print("\n=== P-C: R-Wu vs R-Chughtai on n=15, per seed and block ===")
c15 = S[(S.nmod == 15) & (S.hidden == 128) & S.block.str.isdigit()].pivot_table(index=["seed", "block"], columns="recipe", values="verdict", aggfunc="first"); print(c15.to_string())
if "chughtai" in c15 and "wu" in c15:
    both = c15.dropna(); agree = (both.chughtai == both.wu).groupby(level="seed").all(); ns = len(agree)
    print(f"  seeds with both recipes: {ns}; agreeing on all blocks: {int(agree.sum())}/{ns} -> " + ("PASS" if agree.sum() >= 4 else ("FALSIFIED" if (ns >= 5 and agree.sum() <= 2) else "pending/inconclusive")))
    cc = S[(S.nmod == 15) & (S.hidden == 128) & (S.comp_energy > 1)].copy(); cc["blk"] = cc.block.str.rstrip("αβγδεζηθ")
    cc = cc.pivot_table(index=["seed", "blk"], columns="recipe", values="verdict_component", aggfunc="first").dropna()
    print("  DESCRIPTIVE component-level (sibling-agnostic) agreement:", f"{int((cc.chughtai == cc.wu).groupby(level='seed').all().sum())}/{cc.index.get_level_values('seed').nunique()} seeds")
print("\n=== P-D: m=128 vs m=512 on n=15 (R-Wu) ===")
d15 = S[(S.nmod == 15) & (S.recipe == "wu") & S.block.str.isdigit()].pivot_table(index=["seed", "block"], columns="hidden", values="verdict", aggfunc="first"); print(d15.to_string())
if 512 in d15 and 128 in d15:
    both = d15.dropna(); agree = (both[512] == both[128]).groupby(level="seed").all(); print(f"  seeds with both widths: {len(agree)}; agreeing: {int(agree.sum())}/{len(agree)} -> {'PASS' if agree.sum() >= 4 else 'reported'}")
    dd = S[(S.nmod == 15) & (S.recipe == "wu") & (S.comp_energy > 1)].copy(); dd["blk"] = dd.block.str.rstrip("αβγδεζηθ")
    dd = dd.pivot_table(index=["seed", "blk"], columns="hidden", values="verdict_component", aggfunc="first").dropna()
    print("  DESCRIPTIVE component-level (sibling-agnostic) agreement:", f"{int((dd[512] == dd[128]).groupby(level='seed').all().sum())}/{dd.index.get_level_values('seed').nunique()} seeds")
print("\n=== DESCRIPTIVE component-level verdicts (all irreps dim>=4 incl. siblings; plain ∪ plain-with-character vs twisted) ===")
print(S[S.comp_energy > 1][["run", "block", "C_plain", "C_plainchar", "C_twist", "verdict_component"]].round(2).to_string(index=False))
