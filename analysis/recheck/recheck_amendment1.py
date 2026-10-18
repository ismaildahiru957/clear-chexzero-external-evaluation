# Re-run the A2 prompt-robustness analysis (cells 9-11 of study1_audit_prompts.ipynb) and the held-out
# thresholds of A1 from scores_all_templates.npz, and compare with amend1_results.json.
import numpy as np, pandas as pd, json, time
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root
RES = os.path.join(ROOT, "results")

from sklearn.metrics import roc_curve
SC = dict(np.load(f'{RES}/amendment1/scores_all_templates.npz'))
SC = {tuple(k.split('__')): v.astype(np.float64) for k, v in SC.items()}        # notebook: torch .double() of float32 scores
cohort = pd.read_csv(f'{RES}/nih_registered/cohort.csv')   # identical to the Amendment 1 run's cohort.csv
ref = json.load(open(f'{RES}/amendment1/amend1_results.json'))
FINDINGS = ["Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass","Nodule","Pneumonia","Pneumothorax","Consolidation","Edema","Emphysema","Fibrosis","Pleural_Thickening","Hernia"]
TEMPLATES = ["T1", "T2", "T3", "T4"]; SEED = 2026
test = cohort[cohort.set == "test"]
def eligible(data, col, groups):
    out = []
    for f in FINDINGS:
        if all(len(s := data[(data[col] == g) & (data[f] == 1)]) >= 30 and s.patient.nunique() >= 10 for g in groups): out.append(f)
    return out
SEX_ELIG = eligible(test, "sex", ["F", "M"]); AGE_ELIG = eligible(test, "age_band4", ["0-18", "18-40", "40-60", "60+"])
print("sex eligible", len(SEX_ELIG), "| age eligible", len(AGE_ELIG))
is_test = (cohort.set == "test").values; is_held = (cohort.set == "heldout").values
T = cohort[is_test].reset_index(drop=True); Y = T[FINDINGS].values.astype(np.float64)
pat_codes, pat_idx = np.unique(T.patient.values, return_inverse=True); NP = len(pat_codes)
rng = np.random.default_rng(SEED)
WB = [np.bincount(rng.integers(0, NP, NP), minlength=NP).astype(np.float64) for _ in range(1000)]
G = {"sex_F": (T.sex == "F").values, "sex_M": (T.sex == "M").values}
for a in ["0-18", "18-40", "40-60", "60+"]: G[f"age_{a}"] = (T.age_band4 == a).values
class Cell:
    def __init__(self, s, y, pidx):
        o = np.argsort(s, kind="mergesort"); s = s[o]; self.y, self.pidx = y[o], pidx[o]
        self.starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    def auc(self, w=None):
        wi = 1.0 if w is None else w[self.pidx]
        bp = np.add.reduceat(wi * self.y, self.starts); bn = np.add.reduceat(wi * (1 - self.y), self.starts)
        P, N = bp.sum(), bn.sum()
        if P == 0 or N == 0: return np.nan
        return float((bp * (np.cumsum(bn) - bn + 0.5 * bn)).sum() / (P * N))
def ci(a, lvl=0.95):
    a = a[~np.isnan(a)]; return float(np.quantile(a, (1-lvl)/2)), float(np.quantile(a, 1-(1-lvl)/2))
def auc_table(s_test):
    groups = {"all": np.ones(len(T), bool), **G}; pt, bt = {}, {}
    for g, m in groups.items():
        idx = np.flatnonzero(m)
        for j, f in enumerate(FINDINGS):
            c = Cell(s_test[idx, j], Y[idx, j], pat_idx[idx]); pt[(g, f)] = c.auc(); bt[(g, f)] = np.array([c.auc(w) for w in WB])
    return pt, bt
def macro(pt, bt, g, flist):
    return float(np.nanmean([pt[(g, f)] for f in flist])), np.nanmean(np.array([bt[(g, f)] for f in flist]), axis=0)
def gaps(pt, bt):
    out = {}
    pF, bF = macro(pt, bt, "sex_F", SEX_ELIG); pM, bM = macro(pt, bt, "sex_M", SEX_ELIG)
    out["sex_F_minus_M"] = (pF - pM, ci(bF - bM, 0.95))
    pR, bR = macro(pt, bt, "age_40-60", AGE_ELIG)
    for a in ["0-18", "18-40", "60+"]:
        p, b = macro(pt, bt, f"age_{a}", AGE_ELIG); out[f"age_{a}_minus_40-60"] = (p - pR, ci(b - bR, 1 - 0.05/3))
    return out
worst = 0.0; n_vals = 0
def cmp(a, b):
    global worst, n_vals; n_vals += 1; worst = max(worst, abs(a - b))
for (m, t), s in SC.items():
    if m == "clear_concept_instr": continue
    t0 = time.time(); pt, bt = auc_table(s[is_test]); p, b = macro(pt, bt, "all", FINDINGS)
    r = ref["A2"][f"{m}|{t}"]
    cmp(p, r["macro_all"][0]); [cmp(x, y) for x, y in zip(ci(b), r["macro_all"][1])]
    for k, (gp, (lo, hi)) in gaps(pt, bt).items():
        cmp(gp, r["gaps"][k][0]); cmp(lo, r["gaps"][k][1][0]); cmp(hi, r["gaps"][k][1][1])
    assert {f: round(pt[("all", f)], 3) for f in FINDINGS} == r["per_finding"], (m, t)
    print(f"{m:14s} {t} macro {p:.4f}  ({time.time()-t0:.0f}s)  worst so far {worst:.1e}", flush=True)
# held-out Youden thresholds for the concept route (A1)
s_cc = SC[("clear_concept", "T1")]
for j, f in enumerate(FINDINGS):
    yh = cohort.loc[is_held, f].values; fpr, tpr, th = roc_curve(yh, s_cc[is_held, j])
    cmp(float(th[np.argmax(tpr - fpr)]), ref["thresholds"][f])
# T1 scores against the registered run's per-image scores (test images)
reg = {"chexzero": "scores_chexzero_ensemble.csv", "clear_concept": "scores_clear_concept.csv", "clear_direct": "scores_clear_direct.csv"}
for m, fn in reg.items():
    d = pd.read_csv(f'{RES}/nih_registered/' + fn).set_index("image")
    a = pd.DataFrame(SC[(m, "T1")], index=cohort.image, columns=FINDINGS)
    common = a.index.intersection(d.index)
    print(f"T1 {m}: {len(common)} images shared with registered run, max |diff| {np.abs(a.loc[common, FINDINGS].values - d.loc[common, FINDINGS].values).max():.2e}")
print(f"compared {n_vals} values; worst absolute difference {worst:.2e}")
