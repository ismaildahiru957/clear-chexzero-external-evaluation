# Re-run the registered VinDr analysis (cell 8 of vinbigdata_check.ipynb) on the uploaded files
import numpy as np, pandas as pd, json
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root
RES = os.path.join(ROOT, "results")

# usage: python recheck_vindr.py path/to/cohort.csv  (the file written by notebooks/05_amendment2_vindr.ipynb)
cohort = pd.read_csv(sys.argv[1])
V = f'{RES}/amendment2_vindr'
assert (cohort.image_id.values == pd.read_csv(f'{V}/image_index.csv').image_id.values).all(), 'row order differs from image_index.csv'
S = {"chexzero": np.load(f'{V}/scores_chexzero.npy'), "clear_concept": np.load(f'{V}/scores_clear_concept.npy'),
     "clear_concept_instr": np.load(f'{V}/scores_clear_concept_instr.npy'), "clear_direct": np.load(f'{V}/scores_clear_direct.npy')}
ref = json.load(open(f'{V}/vin_results.json'))
FINDINGS = ref["accuracy"]["majority"]["eligible"]
assert list(cohort.columns[1:27:2]) == FINDINGS
SEED = 2026; C = cohort; n = len(C); pidx = np.arange(n)
rng = np.random.default_rng(SEED)
WB = [np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64) for _ in range(1000)]
class Cell:
    def __init__(self, s, y, p):
        o = np.argsort(s, kind="mergesort"); s = s[o]; self.y, self.p = y[o].astype(np.float64), p[o]
        self.starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    def auc(self, w=None):
        wi = 1.0 if w is None else w[self.p]
        bp = np.add.reduceat(wi * self.y, self.starts); bn = np.add.reduceat(wi * (1 - self.y), self.starts)
        P, N = bp.sum(), bn.sum()
        return np.nan if P == 0 or N == 0 else float((bp * (np.cumsum(bn) - bn + 0.5 * bn)).sum() / (P * N))
def ci(a, lvl=0.95):
    a = a[~np.isnan(a)]; return (float(np.quantile(a, (1-lvl)/2)), float(np.quantile(a, 1-(1-lvl)/2)))
def table(s, Y, mask):
    idx = np.flatnonzero(mask); pt, bt = [], []
    for j in range(len(FINDINGS)):
        c = Cell(s[idx, j], Y[idx, j], pidx[idx]); pt.append(c.auc()); bt.append([c.auc(w) for w in WB])
    return np.array(pt), np.array(bt).T
worst = 0.0
for labname, Y in [("majority", C[FINDINGS].values), ("any", C[["any_" + f for f in FINDINGS]].values)]:
    E = [j for j in range(len(FINDINGS)) if Y[:, j].sum() >= 30]
    tabs = {k: table(S[k], Y, np.ones(n, bool)) for k in S}
    pc, bc = tabs["chexzero"]
    for k, (pk, bk) in tabs.items():
        R = ref["accuracy"][labname][k]
        vals = [(np.nanmean(pk[E]), R["macro"]), *zip(ci(np.nanmean(bk[:, E], axis=1)), R["macro_ci95"])]
        if k != "chexzero":
            vals += [(np.nanmean(pk[E]) - np.nanmean(pc[E]), R["delta_vs_chexzero"]), *zip(ci(np.nanmean(bk[:, E], axis=1) - np.nanmean(bc[:, E], axis=1)), R["delta_ci95"])]
            assert [int((pk[E] > pc[E]).sum()), int((pk[E] < pc[E]).sum())] == R["wins_losses"]
        d = max(abs(a - b) for a, b in vals); worst = max(worst, d)
        print(f"{labname:9s} {k:20s} macro {np.nanmean(pk[E]):.4f}  max |diff| vs vin_results.json = {d:.2e}")
# positives and demographic counts against run_info
ri = ref["run_info"]
assert {f: int(C[f].sum()) for f in FINDINGS} == ri["positives_majority"]
assert {f: int(C["any_" + f].sum()) for f in FINDINGS} == ri["positives_any"]
assert int(C.sex.isna().sum()) == ri["sex_counts"]["None"] and int((C.sex == "F").sum()) == ri["sex_counts"]["F"]
assert int(C.age.isna().sum()) == ri["age_missing"]
print("positives and demographic counts match run_info; worst difference", worst)
