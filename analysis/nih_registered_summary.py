"""Summaries for Study 1 from the bootstrap arrays (preregistered estimands)."""
import json
import numpy as np, pandas as pd
import os; exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "nih_registered_analysis.py")).read().split("t0 = time.time()")[0])   # data, masks, Cell

def load(name):
    z = np.load(f"{OUT}/{name}.npz", allow_pickle=False)
    keys = [tuple(k.split("|")) for k in z["keys"]]
    return {k: i for i, k in enumerate(keys)}, z["point"], z["boot"]
KI, PT, BT = load("boot_main")
KIP, PTP, BTP = load("boot_pa")

def ci(arr, level=0.95):
    a = (1 - level) / 2
    arr = arr[~np.isnan(arr)]
    if arr.size == 0: return float('nan'), float('nan')
    return float(np.quantile(arr, a)), float(np.quantile(arr, 1 - a))

def macro(KI, PT, BT, s, g, flist):
    idx = [KI[(s, g, f)] for f in flist]
    p = float(np.nanmean(PT[idx]))
    sub = BT[:, idx]
    dropped = int(np.isnan(sub).sum())
    return p, np.nanmean(sub, axis=1), dropped

def fmt(p, lo, hi, d=3): return f"{p:.{d}f} ({lo:.{d}f} to {hi:.{d}f})"

R = {}            # machine-readable results
lines = []        # human-readable report
def out(s=""): lines.append(s); print(s)

SEX, AGE = elig["sex"]["eligible"], elig["age"]["eligible"]
SEXPA, AGEPA = elig["sex_PA"]["eligible"], elig["age_PA"]["eligible"]
AGES = ["0-18", "18-40", "40-60", "60+"]

# ---------------- descriptives ----------------
out("== Cohort ==")
desc = []
for g, m in G.items():
    if g.startswith("age5"): continue
    sub = T[m]
    row = {"group": g, "images": int(m.sum()), "patients": int(sub.patient.nunique()),
           "AP_share": round(float((sub.view == "AP").mean()), 3)}
    for f in FINDINGS: row[f"prev_{f}"] = round(float(sub[f].mean()), 4)
    desc.append(row)
desc = pd.DataFrame(desc); desc.to_csv(f"{OUT}/table_cohort.csv", index=False)
out(desc[["group", "images", "patients", "AP_share"]].to_string(index=False))

# ---------------- overall per-finding AUROC ----------------
out("\n== Overall AUROC on the full test split (95% CI) ==")
rows = []
for f in FINDINGS:
    r = {"finding": f}
    for s in SETS:
        i = KI[(s, "all", f)]; lo, hi = ci(BT[:, i]); r[s] = fmt(PT[i], lo, hi)
        r[s + "_pt"] = PT[i]
    rows.append(r)
r = {"finding": "MACRO (14)"}
for s in SETS:
    p, bm, _ = macro(KI, PT, BT, s, "all", FINDINGS); lo, hi = ci(bm); r[s] = fmt(p, lo, hi); r[s + "_pt"] = p
rows.append(r)
ov = pd.DataFrame(rows); ov.to_csv(f"{OUT}/table_overall_auroc.csv", index=False)
out(ov[["finding"] + list(SETS)].to_string(index=False))

# ---------------- primary: subgroup macro-AUROC and gaps ----------------
def subgroup_block(KI, PT, BT, s, sexlist, agelist, label):
    res = {"model": s, "analysis": label}
    mac = {}
    for g in ["sex_F", "sex_M"]:
        p, bm, d = macro(KI, PT, BT, s, g, sexlist); mac[g] = (p, bm); res[f"macro_{g}"] = (p, *ci(bm)); res[f"drop_{g}"] = d
    if agelist:
        for a in AGES:
            g = f"age_{a}"; p, bm, d = macro(KI, PT, BT, s, g, agelist); mac[g] = (p, bm); res[f"macro_{g}"] = (p, *ci(bm)); res[f"drop_{g}"] = d
    gaps = {}
    p = mac["sex_F"][0] - mac["sex_M"][0]; bm = mac["sex_F"][1] - mac["sex_M"][1]
    gaps["sex_F_minus_M"] = (p, bm)
    if agelist:
        for a in ["0-18", "18-40", "60+"]:
            p = mac[f"age_{a}"][0] - mac["age_40-60"][0]; bm = mac[f"age_{a}"][1] - mac["age_40-60"][1]
            gaps[f"age_{a}_minus_40-60"] = (p, bm)
    for k, (p, bm) in gaps.items():
        lvl = 0.95 if k.startswith("sex") else 1 - 0.05 / 3
        res[k] = {"point": p, "ci95": ci(bm, 0.95), "ci_primary": ci(bm, lvl), "level": lvl,
                  "excludes_zero": bool(ci(bm, lvl)[0] > 0 or ci(bm, lvl)[1] < 0)}
    return res, gaps

def report_block(res):
    out(f"\n-- {res['model']} | {res['analysis']} --")
    for k, v in res.items():
        if k.startswith("macro_"):
            p, lo, hi = v; out(f"  {k[6:]:<10} macro-AUROC {fmt(p, lo, hi)}")
    for k, v in res.items():
        if isinstance(v, dict):
            lo, hi = v["ci_primary"]; lo95, hi95 = v["ci95"]
            lv = f"{v['level']*100:.2f}%"
            out(f"  GAP {k:<22} {v['point']:+.3f}  {lv} CI {lo:+.3f} to {hi:+.3f}"
                + ("" if v["level"] == 0.95 else f"  (95% {lo95:+.3f} to {hi95:+.3f})")
                + ("  <- interval excludes 0" if v["excludes_zero"] else ""))
    drops = {k: v for k, v in res.items() if k.startswith("drop_") and v}
    if drops: out(f"  resample-level dropped findings: {drops}")

out("\n== PRIMARY: subgroup macro-AUROC and gaps (patient bootstrap, 1,000 resamples) ==")
out(f"Sex comparison findings ({len(SEX)}): {SEX}")
out(f"Age comparison findings ({len(AGE)}): {AGE}")
PRIMARY = {}
for s in SETS:
    res, gaps = subgroup_block(KI, PT, BT, s, SEX, AGE, "primary (all test images)")
    PRIMARY[s] = (res, gaps); report_block(res)
    R[f"primary_{s}"] = res

# ---------------- H3: CLEAR gap minus CheXzero gap ----------------
out("\n== H3: CLEAR (concept, primary) gap minus CheXzero gap ==")
H3 = {}
for k in PRIMARY["chexzero"][1]:
    pc, bc = PRIMARY["chexzero"][1][k]; pl, bl = PRIMARY["clear_concept"][1][k]
    lvl = 0.95 if k.startswith("sex") else 1 - 0.05 / 3
    d, bd = pl - pc, bl - bc
    lo, hi = ci(bd, lvl)
    H3[k] = {"point": d, "ci_primary": (lo, hi), "ci95": ci(bd), "level": lvl, "excludes_zero": bool(lo > 0 or hi < 0)}
    out(f"  {k:<22} {d:+.3f}  {lvl*100:.2f}% CI {lo:+.3f} to {hi:+.3f}" + ("  <- excludes 0" if (lo > 0 or hi < 0) else ""))
R["H3"] = H3

# ---------------- S1: PA only ----------------
out("\n== S1: PA images only ==")
out(f"Sex findings ({len(SEXPA)}); age findings ({len(AGEPA)}: {AGEPA}, descriptive only)")
for s in ["chexzero", "clear_concept"]:
    res, _ = subgroup_block(KIP, PTP, BTP, s, SEXPA, AGEPA, "S1 PA only")
    report_block(res); R[f"S1_{s}"] = res

# ---------------- exploratory: five age bands ----------------
out("\n== Exploratory: five original age bands (descriptive) ==")
B5 = ["0-18", "18-40", "40-60", "60-80", "80+"]
elig5 = []
for f in FINDINGS:
    ok = True
    for a in B5:
        sub = T[(T.age_band5 == a) & (T[f] == 1)]
        ok &= len(sub) >= 30 and sub.patient.nunique() >= 10
    if ok: elig5.append(f)
out(f"Eligible findings across all five bands: {elig5}")
if elig5:
    for s in ["chexzero", "clear_concept"]:
        parts = []
        for a in B5:
            p, bm, _ = macro(KI, PT, BT, s, f"age5_{a}", elig5); lo, hi = ci(bm); parts.append(f"{a}: {fmt(p, lo, hi)}")
        out(f"  {s}: " + " | ".join(parts))

# ---------------- per-finding AUROC by subgroup ----------------
rows = []
for s in ["chexzero", "clear_concept", "clear_direct", "clear_concept_instr"]:
    for f in FINDINGS:
        r = {"model": s, "finding": f}
        for g in ["sex_F", "sex_M"] + [f"age_{a}" for a in AGES]:
            i = KI[(s, g, f)]; lo, hi = ci(BT[:, i]); r[g] = fmt(PT[i], lo, hi) if not np.isnan(PT[i]) else "n/a"
        rows.append(r)
pd.DataFrame(rows).to_csv(f"{OUT}/table_perfinding_subgroup_auroc.csv", index=False)

# ---------------- secondary: false-negative rate at Youden threshold (held-out patients) ----------------
from sklearn.metrics import roc_curve
out("\n== Secondary: false-negative rate at the Youden threshold set on held-out training patients ==")
rng2 = np.random.default_rng(SEED)
Wb = [np.bincount(rng2.integers(0, NP, NP), minlength=NP).astype(np.float64) for _ in range(B)]
fnr_rows = []
FNR = {}
for s in ["chexzero", "clear_concept"]:
    thr = {}
    for j, f in enumerate(FINDINGS):
        fpr, tpr, th = roc_curve(Y_held[:, j], S_held[s][:, j])
        thr[f] = float(th[np.argmax(tpr - fpr)])
    FNR[s] = {"thresholds": thr}
    # per finding, per group: FNR = weighted share of positives scored below threshold
    groups = ["sex_F", "sex_M"] + [f"age_{a}" for a in AGES]
    point, boots = {}, {}
    for g in groups:
        m = G[g]
        for j, f in enumerate(FINDINGS):
            pos = m & (Y[:, j] == 1)
            miss = (S_test[s][:, j] < thr[f]).astype(float)
            pidx = pat_idx[pos]; mi = miss[pos]
            point[(g, f)] = float(mi.mean()) if pos.any() else np.nan
            boots[(g, f)] = np.array([(w[pidx] * mi).sum() / w[pidx].sum() if w[pidx].sum() > 0 else np.nan for w in Wb])
    for g in groups:
        flist = SEX if g.startswith("sex") else AGE
        p = np.nanmean([point[(g, f)] for f in flist]); bm = np.nanmean([boots[(g, f)] for f in flist], axis=0)
        lo, hi = ci(bm); fnr_rows.append({"model": s, "group": g, "mean_FNR": fmt(p, lo, hi)})
        FNR[s][g] = (p, bm)
        for f in flist:
            lo, hi = ci(boots[(g, f)]); fnr_rows.append({"model": s, "group": g, "finding": f, "FNR": fmt(point[(g, f)], lo, hi)})
    out(f"-- {s}: mean FNR over eligible findings")
    for g in groups:
        p, bm = FNR[s][g]; lo, hi = ci(bm); out(f"  {g:<10} {fmt(p, lo, hi)}")
    p = FNR[s]["sex_F"][0] - FNR[s]["sex_M"][0]; bm = FNR[s]["sex_F"][1] - FNR[s]["sex_M"][1]; lo, hi = ci(bm)
    out(f"  FNR gap F minus M {p:+.3f} (95% CI {lo:+.3f} to {hi:+.3f})")
    for a in ["0-18", "18-40", "60+"]:
        p = FNR[s][f"age_{a}"][0] - FNR[s]["age_40-60"][0]; bm = FNR[s][f"age_{a}"][1] - FNR[s]["age_40-60"][1]
        lo, hi = ci(bm, 1 - 0.05 / 3); out(f"  FNR gap {a} minus 40-60 {p:+.3f} (98.33% CI {lo:+.3f} to {hi:+.3f})")
pd.DataFrame(fnr_rows).to_csv(f"{OUT}/table_fnr.csv", index=False)

open(f"{OUT}/report.txt", "w").write("\n".join(lines))
json.dump(R, open(f"{OUT}/results.json", "w"), indent=1, default=lambda o: list(o) if isinstance(o, tuple) else str(o))
