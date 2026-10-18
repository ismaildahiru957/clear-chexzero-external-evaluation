"""Study 1 preregistered analysis (OSF, 1 Oct 2026).

Inputs: cohort.csv, eligibility.json and four score files from study1_full_scoring.ipynb.
Patient-level bootstrap, 1,000 resamples, seed 2026, paired across subgroups and models.
"""
import json, time
import numpy as np, pandas as pd

import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
D = os.environ.get("NIH_DIR", os.path.join(ROOT, "results", "nih_registered"))   # scores, cohort, eligibility
OUT = os.environ.get("NIH_OUT", D)                                                  # analysis outputs
import os; os.makedirs(OUT, exist_ok=True)
FINDINGS = ["Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass","Nodule","Pneumonia",
            "Pneumothorax","Consolidation","Edema","Emphysema","Fibrosis","Pleural_Thickening","Hernia"]
SETS = {"chexzero": "scores_chexzero_ensemble.csv", "clear_concept": "scores_clear_concept.csv",
        "clear_direct": "scores_clear_direct.csv", "clear_concept_instr": "scores_clear_concept_instructprompts.csv"}
B, SEED = 1000, 2026

cohort = pd.read_csv(f"{D}/cohort.csv")
elig = json.load(open(f"{D}/eligibility.json"))
scores = {}
for k, f in SETS.items():
    s = pd.read_csv(f"{D}/{f}")
    assert (s.image.values == cohort.image.values).all()
    scores[k] = s[FINDINGS].values.astype(np.float64)

test_mask = (cohort.set == "test").values
held_mask = (cohort.set == "heldout").values
T = cohort[test_mask].reset_index(drop=True)
Y = T[FINDINGS].values.astype(np.int8)
S_test = {k: v[test_mask] for k, v in scores.items()}
S_held = {k: v[held_mask] for k, v in scores.items()}
Y_held = cohort.loc[held_mask, FINDINGS].values.astype(np.int8)

pat_codes, pat_idx = np.unique(T.patient.values, return_inverse=True)
NP = len(pat_codes)

# ---------- subgroups ----------
def masks(df):
    m = {"all": np.ones(len(df), bool)}
    for s in ["F", "M"]: m[f"sex_{s}"] = (df.sex == s).values
    for a in ["0-18", "18-40", "40-60", "60+"]: m[f"age_{a}"] = (df.age_band4 == a).values
    for a in ["0-18", "18-40", "40-60", "60-80", "80+"]: m[f"age5_{a}"] = (df.age_band5 == a).values
    return m
PA = (T.view == "PA").values
G = masks(T)
GPA = {k: v & PA for k, v in masks(T).items()}

# ---------- weighted AUROC machinery ----------
class Cell:
    """Pre-sorted data for one (score set, finding, subgroup)."""
    __slots__ = ("pidx", "y", "starts", "ok")
    def __init__(self, s, y, pidx):
        o = np.argsort(s, kind="mergesort")
        s, self.y, self.pidx = s[o], y[o].astype(np.float64), pidx[o]
        self.starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
        self.ok = (self.y.sum() > 0) and (self.y.sum() < len(self.y))
    def auc(self, w=None):
        y = self.y
        if w is None:
            pos, neg = y, 1 - y
        else:
            wi = w[self.pidx]; pos, neg = wi * y, wi * (1 - y)
        bp, bn = np.add.reduceat(pos, self.starts), np.add.reduceat(neg, self.starts)
        P, N = bp.sum(), bn.sum()
        if P == 0 or N == 0: return np.nan
        before = np.cumsum(bn) - bn
        return float((bp * (before + 0.5 * bn)).sum() / (P * N))

def build(group_masks, which_sets):
    cells = {}
    for k in which_sets:
        for gname, gm in group_masks.items():
            idx = np.flatnonzero(gm)
            for j, f in enumerate(FINDINGS):
                cells[(k, gname, f)] = Cell(S_test[k][idx, j], Y[idx, j], pat_idx[idx])
    return cells

t0 = time.time()
ALL_SETS = list(SETS)
cells = build(G, ALL_SETS)
cells_pa = build({k: GPA[k] for k in ["all", "sex_F", "sex_M", "age_0-18", "age_18-40", "age_40-60", "age_60+"]}, ALL_SETS)
print("built cells", round(time.time() - t0, 1), "s")

# ---------- bootstrap: AUROC for every cell, point estimate + B resamples ----------
rng = np.random.default_rng(SEED)
W = [np.bincount(rng.integers(0, NP, NP), minlength=NP).astype(np.float64) for _ in range(B)]

def run(cellset):
    keys = list(cellset)
    point = np.array([cellset[c].auc() for c in keys])
    boot = np.empty((B, len(keys)))
    for b in range(B):
        w = W[b]
        boot[b] = [cellset[c].auc(w) for c in keys]
        if b % 200 == 0: print("  resample", b, round(time.time() - t0, 1), "s", flush=True)
    return keys, point, boot

keys, point, boot = run(cells)
np.savez_compressed(f"{OUT}/boot_main.npz", point=point, boot=boot, keys=np.array(["|".join(k) for k in keys]))
keys_pa, point_pa, boot_pa = run(cells_pa)
np.savez_compressed(f"{OUT}/boot_pa.npz", point=point_pa, boot=boot_pa, keys=np.array(["|".join(k) for k in keys_pa]))
print("bootstrap done", round(time.time() - t0, 1), "s")
