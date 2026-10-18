# Re-run cell 8 of indiana_check.ipynb on the uploaded files and compare with indiana_results.json
import numpy as np, pandas as pd, json
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root
RES = os.path.join(ROOT, "results")

# usage: python recheck_indiana.py path/to/cohort.csv  (the file written by notebooks/03_indiana.ipynb)
cohort = pd.read_csv(sys.argv[1])
I = f'{RES}/indiana'
idx = pd.read_csv(f'{I}/image_index.csv')
assert (cohort.uid.values == idx.uid.values).all() and (cohort.filename.values == idx.filename.values).all(), 'row order differs from image_index.csv'
ref = json.load(open(f'{I}/indiana_results.json'))
FINDINGS = list(ref["summary"]["all_images"]["per_finding"].keys())
assert list(cohort.columns[4:4+51]) == FINDINGS and len(FINDINGS) == 51
S = {"chexzero": np.load(f'{I}/scores_chexzero.npy'), "clear_concept": np.load(f'{I}/scores_clear_concept.npy'),
     "clear_concept_instr": np.load(f'{I}/scores_clear_concept_instr.npy'), "clear_direct": np.load(f'{I}/scores_clear_direct.npy'),
     "clear_concept_squash": np.load(f'{I}/scores_clear_concept_squash.npy')}
SEED = 2026
Y = cohort[FINDINGS].values.astype(np.float64)
uid_codes, uid_idx = np.unique(cohort.uid.values, return_inverse=True); NU = len(uid_codes)
def wauc(s, y, w):
    o = np.argsort(s, kind="mergesort"); s, y, w = s[o], y[o], w[o]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    bp, bn = np.add.reduceat(w * y, starts), np.add.reduceat(w * (1 - y), starts)
    P, N = bp.sum(), bn.sum()
    if P == 0 or N == 0: return np.nan
    return float((bp * (np.cumsum(bn) - bn + 0.5 * bn)).sum() / (P * N))
rng = np.random.default_rng(SEED)
W = [np.bincount(rng.integers(0, NU, NU), minlength=NU).astype(np.float64) for _ in range(1000)]
def q(a): a = a[~np.isnan(a)]; return float(np.quantile(a, .025)), float(np.quantile(a, .975))
worst = 0
for name, mask in [("all_images", np.ones(len(cohort), bool)), ("frontal_only", (cohort.projection == "Frontal").values)]:
    ev = {}
    for k, s in S.items():
        pt = np.array([wauc(s[mask, j], Y[mask, j], np.ones(mask.sum())) for j in range(51)])
        bt = np.array([[wauc(s[mask, j], Y[mask, j], w[uid_idx[mask]]) for j in range(51)] for w in W])
        ev[k] = (pt, bt)
    R = ref["summary"][name]; pc, bc = ev["chexzero"]
    for k, (pk, bk) in ev.items():
        vals = [(np.nanmean(pk), R[k]["macro"]), *zip(q(np.nanmean(bk, axis=1)), R[k]["macro_ci95"])]
        if k != "chexzero":
            vals += [(np.nanmean(pk) - np.nanmean(pc), R[k]["delta_vs_chexzero"]), *zip(q(np.nanmean(bk, axis=1) - np.nanmean(bc, axis=1)), R[k]["delta_ci95"])]
            assert [int((pk > pc).sum()), int((pk < pc).sum())] == R[k]["wins_losses_vs_chexzero"]
        for j, f in enumerate(FINDINGS): assert round(float(pk[j]), 3) == R["per_finding"][f][k], (name, k, f)
        d = max(abs(a - b) for a, b in vals); worst = max(worst, d)
        print(f"{name:13s} {k:22s} macro {np.nanmean(pk):.4f}  max |diff| = {d:.1e}")
ri = ref["run_info"]
assert len(cohort) == ri["n_images"] and cohort.uid.nunique() == ri["n_reports"] and int((cohort.projection == "Frontal").sum()) == ri["n_frontal"]
assert {f: int(cohort.groupby("uid")[f].max().sum()) for f in FINDINGS} == ri["findings_positive_reports"]
pf = pd.read_csv(f'{I}/per_finding_auroc.csv', index_col=0)
assert all(pf.loc[f, k] == ref["summary"]["all_images"]["per_finding"][f][k] for f in FINDINGS for k in S)
print("counts, positive reports and per_finding_auroc.csv all match; worst difference", worst)
