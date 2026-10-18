"""Fixed draw of one true positive per finding for the example-image figure (post hoc, illustrative).
Pool: NIH test images positive for the finding and scored at or above the held-out Youden threshold by CLEAR's concept route (T1).
Draw: candidates sorted by file name; numpy default_rng(2026), one draw per finding in the order below."""
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_curve
co = pd.read_csv("data/cohort.csv"); sc = pd.read_csv("data/scores_clear_concept.csv")
assert (co.image.values == sc.image.values).all() and (co.set.values == sc.set.values).all()
h, t = co.set == "heldout", co.set == "test"
rng = np.random.default_rng(2026); sel = {}
for f in ["Cardiomegaly", "Emphysema", "Hernia"]:
    fpr, tpr, th = roc_curve(co.loc[h, f], sc.loc[h, f]); thr = float(th[np.argmax(tpr - fpr)])
    cand = sorted(co.loc[t & (co[f] == 1) & (sc[f] >= thr), "image"])
    pick = cand[rng.integers(len(cand))]; r = co[co.image == pick].iloc[0]
    sel[f] = dict(image=pick, n_tp=len(cand), n_pos=int((t & (co[f] == 1)).sum()), thr=thr,
                  score=float(sc.loc[co.image == pick, f].iloc[0]), labels=r.labels, age=int(r.age), sex=r.sex, view=r.view)
json.dump(sel, open("fig7_selection.json", "w"), indent=1); print(sel)
