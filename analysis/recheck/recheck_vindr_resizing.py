# Re-run cell 6 of vindr_squash_check.ipynb on the uploaded files and compare with vin_squash_results.json
import numpy as np, pandas as pd, json
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root
RES = os.path.join(ROOT, "results")

# usage: python recheck_vindr_resizing.py path/to/cohort.csv  (the file written by notebooks/05_amendment2_vindr.ipynb)
vc = pd.read_csv(sys.argv[1])
V, X = f'{RES}/amendment2_vindr', f'{RES}/posthoc_vindr_resizing'
assert (vc.image_id.values == pd.read_csv(f'{V}/image_index.csv').image_id.values).all(), 'row order differs from image_index.csv'
ref = json.load(open(f'{X}/vin_squash_results.json'))
FINDINGS = ref["results"]["all"]["eligible"]
S_prev = {"chexzero": np.load(f'{V}/scores_chexzero.npy'), "clear_concept": np.load(f'{V}/scores_clear_concept.npy'), "clear_direct": np.load(f'{V}/scores_clear_direct.npy')}
S_prev = {k: v.astype(np.float64) for k, v in S_prev.items()}
c_sq = np.load(f'{X}/scores_clear_concept_squash.npy').astype(np.float64); d_sq = np.load(f'{X}/scores_clear_direct_squash.npy').astype(np.float64)
assert c_sq.shape == d_sq.shape == (15000, 13)
SEED = 2026
class Cell:
    def __init__(self, s, y, pidx):
        o = np.argsort(s, kind="mergesort"); s = s[o]; self.y, self.p = y[o].astype(np.float64), pidx[o]
        self.starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    def auc(self, w=None):
        wi = 1.0 if w is None else w[self.p]
        bp = np.add.reduceat(wi * self.y, self.starts); bn = np.add.reduceat(wi * (1 - self.y), self.starts)
        P, N = bp.sum(), bn.sum()
        return np.nan if P == 0 or N == 0 else float((bp * (np.cumsum(bn) - bn + 0.5 * bn)).sum() / (P * N))
def ci(a, lvl=0.95):
    a = a[~np.isnan(a)]; return (float(np.quantile(a, (1-lvl)/2)), float(np.quantile(a, 1-(1-lvl)/2)))
def table(s, Y, mask, pidx, WB):
    idx = np.flatnonzero(mask); pt, bt = [], []
    for j in range(Y.shape[1]):
        c = Cell(s[idx, j], Y[idx, j], pidx[idx]); pt.append(c.auc()); bt.append([c.auc(w) for w in WB])
    return np.array(pt), np.array(bt).T
def mac(tab, E): pt, bt = tab; return float(np.nanmean(pt[E])), np.nanmean(bt[:, E], axis=1)
ALL13 = list(vc.columns[1:27:2])
Y = vc[ALL13].values; n = len(vc); iidx = np.arange(n)
rng = np.random.default_rng(SEED); WB = [np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64) for _ in range(1000)]
SC = {"chexzero": S_prev["chexzero"], "concept_pad": S_prev["clear_concept"], "concept_squash": c_sq, "direct_pad": S_prev["clear_direct"], "direct_squash": d_sq}
sk = vc.sex.isin(["M", "F"]).values; ak = vc.age.notna().values
subsets = {"all": np.ones(n, bool), "sex_known": sk, "sex_missing": ~sk, "age_known": ak, "age_missing": ~ak}
bad = 0; checked = 0
def same(a, b):
    global bad, checked; checked += 1
    if a != b: bad += 1; print("MISMATCH", a, b)
for g, m in subsets.items():
    E = [j for j in range(13) if Y[m, j].sum() >= 30]; R = ref["results"][g]
    same(R["n"], int(m.sum())); same(R["eligible"], [ALL13[j] for j in E])
    T = {k: table(v, Y, m, iidx, WB) for k, v in SC.items()}
    pc, bc = mac(T["chexzero"], E); same(R["chexzero"], round(pc, 4))
    for k in ["concept_pad", "concept_squash", "direct_pad", "direct_squash"]:
        pk, bk = mac(T[k], E)
        same(R[k]["macro"], round(pk, 4)); same(R[k]["minus_chexzero"], [round(pk - pc, 4)] + [round(x, 4) for x in ci(bk - bc)])
        same(R[k]["wins_losses"], [int((T[k][0][E] > T["chexzero"][0][E]).sum()), int((T[k][0][E] < T["chexzero"][0][E]).sum())])
    for a, b in [("concept_squash", "concept_pad"), ("direct_squash", "direct_pad")]:
        pa, ba = mac(T[a], E); pb, bb = mac(T[b], E)
        same(R[f"{a}_minus_{b.split('_')[1]}"], [round(pa - pb, 4)] + [round(x, 4) for x in ci(ba - bb)])
    same(R["per_finding"], {ALL13[j]: {k: round(float(T[k][0][j]), 3) for k in SC} for j in E})
    print(g, "n", int(m.sum()), "concept_squash minus CheXzero", R["concept_squash"]["minus_chexzero"], "| squash minus pad", R["concept_squash_minus_pad"])
print(f"checked {checked} items, mismatches {bad}")
