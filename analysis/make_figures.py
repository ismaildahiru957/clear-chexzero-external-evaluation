"""Publication figures for the Study 1 manuscript. Every number is parsed from the run outputs in results/, not typed by hand.
Run from anywhere: python analysis/make_figures.py  (figures are written to figures/, or to $FIG_OUT if set)."""
import re, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))         # repository root
RES = os.path.join(ROOT, "results")
REPORT = f"{RES}/nih_registered/report.txt"                                 # registered NIH analysis report
AMEND = f"{RES}/amendment1/run_log.txt"                                     # Amendment 1 run log
VIN = f"{RES}/amendment2_vindr/run_log.txt"                                 # Amendment 2 run log
FOLLOW = f"{RES}/posthoc_followup/run_log.txt"                              # post hoc follow-up log
SUBSETS = f"{RES}/posthoc_vindr_subsets/rerun_output_2026-10-18.txt"        # post hoc VinDr subset summary
OUT = os.environ.get("FIG_OUT", os.path.join(ROOT, "figures")); os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False})
COL = {"CheXzero": "#2a78d6", "CLEAR concept": "#eb6834", "CLEAR direct": "#1baf7a"}
MK = {"CheXzero": "o", "CLEAR concept": "s", "CLEAR direct": "^"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dddcd7"
NUM = r"([+-]?\d*\.\d+)"

def save(fig, name):
    for ext in ("pdf", "png", "tiff"):
        kw = {"dpi": 600} if ext != "pdf" else {}
        if ext == "tiff": kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)

def strip_log(path):
    out = []
    for line in open(path, encoding="utf-8"):
        m = re.match(r"^\s*[\d.]+s\s+\d+\s(.*)$", line.rstrip("\n"))
        out.append(m.group(1) if m else line.rstrip("\n"))
    return out

# ---------- parse the registered NIH report ----------
rep = open(REPORT).read()
FIND = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax",
        "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
NIH = {}
for f in FIND + ["MACRO (14)"]:
    m = re.search(re.escape(f) + r"\s+" + r"\s+".join([NUM + r" \(" + NUM + r" to " + NUM + r"\)"] * 4), rep)
    v = [float(x) for x in m.groups()]
    NIH[f] = {"CheXzero": v[0:3], "CLEAR concept": v[3:6], "CLEAR direct": v[6:9], "CLEAR instr": v[9:12]}

# ---------- parse Amendment 1 summary ----------
am = strip_log(AMEND)
T = {}
for line in am:
    m = re.match(r"\s*(chexzero|clear_concept|clear_direct)\s+(T\d) macro " + NUM + r" \(" + NUM + "-" + NUM + r"\) \| (.*)", line)
    if m:
        model = {"chexzero": "CheXzero", "clear_concept": "CLEAR concept", "clear_direct": "CLEAR direct"}[m.group(1)]
        gaps = dict((g, (float(a), float(b), float(c))) for g, a, b, c in
                    re.findall(r"(sex_F-M|age_0-18-40-60|age_18-40-40-60|age_60\+-40-60): " + NUM + r" \(" + NUM + "," + NUM + r"\)", m.group(6)))
        T[(model, m.group(2))] = {"macro": tuple(float(x) for x in m.group(3, 4, 5)), "gaps": gaps}
assert len(T) == 12, len(T)
PROBE, REL = {}, {}
for line in am:
    m = re.match(r"\s*(\w+)\s+'.*': " + NUM + r" \(" + NUM + "-" + NUM + r"\) vs " + NUM + " / " + NUM + " / " + NUM, line)
    if m and m.group(1) in FIND: PROBE[m.group(1)] = [float(x) for x in m.groups()[1:]]
    m2 = re.match(r"\s*(\w+)\s+" + NUM + " / " + NUM + " / " + NUM + r"\s*$", line)
    if m2 and m2.group(1) in FIND: REL[m2.group(1)] = [float(x) for x in m2.groups()[1:]]
assert len(PROBE) == 14 and len(REL) == 14

# ---------- parse follow-up paired contrasts ----------
fo = strip_log(FOLLOW)
PAIR = {}
for line in fo:
    m = re.match(r"\s*(\S+)\s+" + NUM + r" \(" + NUM + " to " + NUM + r"\)\s*$", line)
    if m: PAIR[m.group(1)] = tuple(float(x) for x in m.groups()[1:])

# ---------- parse Amendment 2 (VinDr) ----------
vn = strip_log(VIN)
VMAJ = {}
sec = None
for line in vn:
    if "-- majority labels" in line: sec = "maj"
    if "-- any labels" in line: sec = "any"
    m = re.match(r"\s*(chexzero|clear_concept|clear_direct)\s+macro " + NUM + r" \(" + NUM + "-" + NUM + r"\)(?: \| minus CheXzero " + NUM + r" \(" + NUM + " to " + NUM + r"\))?", line)
    if m and sec == "maj": VMAJ[m.group(1)] = [float(x) if x else None for x in m.groups()[1:]]

# VinDr subsets: summary printed by notebooks/07_posthoc_vindr_subsets.ipynb
_sub = {}
for line in open(SUBSETS, encoding="utf-8"):
    m = re.match(r"-- (\w+) \(n=(\d+), (\d+) findings\): CheXzero [\d.]+ \| clear_concept [\d.]+, minus CheXzero " + NUM + r" \(" + NUM + " to " + NUM
                 + r"\), wins/losses \d+/\d+ \| clear_direct [\d.]+, minus CheXzero " + NUM + r" \(" + NUM + " to " + NUM + r"\)", line)
    if m: _sub[m.group(1)] = (int(m.group(2)), int(m.group(3)), tuple(float(x) for x in m.group(4, 5, 6)), tuple(float(x) for x in m.group(7, 8, 9)))
VSUB = [(lab,) + _sub[k] for lab, k in [("All images", "all"), ("Sex recorded", "sex_known"), ("Age recorded", "age_known"),
                                        ("Sex missing", "sex_missing"), ("Age missing", "age_missing")]]
assert abs(VMAJ["clear_concept"][3] - 0.057) < 1e-9 and abs(VMAJ["clear_direct"][3] - 0.062) < 1e-9

NICE = {f: f.replace("_", " ") for f in FIND}
NICE["Pleural_Thickening"] = "Pleural thickening"
MODELS = ["CheXzero", "CLEAR concept", "CLEAR direct"]

def style_ax(ax, xlabel=None):
    ax.tick_params(colors=MUTED, length=2.5); ax.xaxis.label.set_color(INK)
    for s in ("left", "bottom"): ax.spines[s].set_color(MUTED)
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    if xlabel: ax.set_xlabel(xlabel)

# ================= Figure 1: scoring routes (a) and study design (b) =================
# Every count below is read from a run file. Text that is not a result (model and encoder names, prompt wording,
# the three VinDr-CXR readers per image [Nguyen et al.], the OSF dates) is written as text.
import json
RUN = json.load(open(f"{RES}/nih_registered/run_info.json"))          # registered NIH run, started 2026-10-01 15:25:10 UTC
N_CKPT = len(RUN["chexzero_checkpoints"]); N_CONC = RUN["n_concepts"]; LLM_DIM = RUN["llm_dim"]
M_SHAPE = None
for line in am:                                                           # Amendment 1 run log: projection matrix shape
    m = re.match(r"\s*M: \((\d+), (\d+)\)", line)
    if m: M_SHAPE = (int(m.group(1)), int(m.group(2)))
assert M_SHAPE and M_SHAPE[1] == LLM_DIM, M_SHAPE
CLEAR_DIM = M_SHAPE[0]
CZ_PX = int(re.search(r"Resize\((\d+), interpolation", open(f"{ROOT}/notebooks/02_nih_registered_scoring.ipynb").read()).group(1))
_clp = re.search(r"Resize\(size=\((\d+), (\d+)\)", json.load(open(f"{RES}/posthoc_vindr_resizing/vin_squash_results.json"))["run_info"]["clear_preprocess"])
assert _clp.group(1) == _clp.group(2); CL_PX = int(_clp.group(1))         # CLEAR's own input size, as recorded at run time
N_BOOT = int(re.search(r"patient bootstrap, ([\d,]+) resamples", rep).group(1).replace(",", ""))
IND_LOG = f"{RES}/indiana/run_log.txt"                                  # Indiana run log
m = re.search(r"images (\d+) \| reports (\d+) \| frontal (\d+)", open(IND_LOG).read())
IND_IMG, IND_REP = int(m.group(1)), int(m.group(2))
VIN_N = int(re.search(r'"n_images": (\d+)', open(VIN).read()).group(1))
VIN_K = int(re.search(r"-- majority labels, eligible findings: (\d+)", open(VIN).read()).group(1))
N_FIND = len(FIND); ADJ = 100 * (1 - 0.05 / 3)

ELIG = json.load(open(f"{RES}/nih_registered/eligibility.json"))
N_ELIG_SEX, N_ELIG_AGE = len(ELIG["sex"]["eligible"]), len(ELIG["age"]["eligible"])
_co = pd.read_csv(f"{RES}/nih_registered/cohort.csv"); _te = _co[_co.set == "test"]
assert _te.sex.notna().all() and _te.age.notna().all()
N_F_PAT, N_M_PAT = _te[_te.sex == "F"].patient.nunique(), _te[_te.sex == "M"].patient.nunique()

def fig1():
    U = 0.0696                                                             # inches per data unit on both axes
    W = 6.6; XW = W / U; ya, gap = 43.5, 2.5
    LH = 1.76; G = 2.2                                                     # body line height and box gap in data units
    def bh(n): return 3.4 + 1.1 + n * LH + 0.9                            # box height for n body lines
    hR, hA, hI, hV = bh(4), bh(3), bh(3), bh(3)
    hM, hP, hL = bh(4), bh(8), bh(4)
    CONTENT = max(hR + G + hA + G + hI + G + hV, hM + G + hP + G + hL)
    yb = CONTENT + 7.2 + 6.2
    H = (ya + yb + gap) * U
    fig = plt.figure(figsize=(W, H))
    a = fig.add_axes([0, (yb + gap) * U / H, 1, ya * U / H]); a.set_xlim(0, XW); a.set_ylim(0, ya); a.axis("off")
    b = fig.add_axes([0, 0, 1, yb * U / H]); b.set_xlim(0, XW); b.set_ylim(0, yb); b.axis("off")
    EDGE, BAND, PILL = "#b4b2ab", "#eeece6", "#3d3c39"
    def box(ax, x, y, w, h, title, body, tcol=INK, pill=None):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc="white", ec=EDGE, lw=0.6, zorder=1))
        ax.add_patch(plt.Rectangle((x, y + h - 3.4), w, 3.4, fc=BAND, ec=EDGE, lw=0.6, zorder=2))
        ax.text(x + 0.9, y + h - 1.7, title, fontsize=6.9, fontweight="bold", color=tcol, va="center", zorder=3)
        if pill:
            ax.text(x + w - 0.8, y + h - 1.7, pill, fontsize=5.4, color="white", ha="right", va="center", zorder=3,
                    bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.5", fc=PILL, ec="none"))
        if body: ax.text(x + 0.9, y + h - 4.5, body, fontsize=6.2, color=INK, va="top", linespacing=1.42, zorder=3)
    def arrow(ax, x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.7, mutation_scale=7, shrinkA=0, shrinkB=0), zorder=4)
    def header(ax, x, w, y, txt):
        ax.text(x, y, txt, fontsize=5.8, color=MUTED, fontweight="bold", va="bottom")
        ax.plot([x, x + w], [y - 0.5, y - 0.5], color=EDGE, lw=0.6)

    # ---------- panel a: how each model scores one image for finding f ----------
    a.text(0, ya - 0.2, "a  How each model scores an image for finding f", fontsize=7.8, fontweight="bold", color=INK, va="top")
    a.text(0, ya - 3.2, "Released models, frozen; shown with the registered prompt template T1", fontsize=6.3, color=MUTED, va="top")
    X = [(0, 12.0), (15.5, 21.5), (40.5, 34.5), (78.5, XW - 78.6)]       # input, image encoder, text side, score
    for (x, w), t in zip(X, ["INPUT", "IMAGE ENCODER", "TEXT SIDE AND COMPARISON", "SCORE"]): header(a, x, w, ya - 7.6, t)
    def link(ys, c):
        for yy in ys: arrow(a, X[c][0] + X[c][1] + 0.4, yy, X[c + 1][0] - 0.4, yy)
    hc = bh(3); yc = ya - 9.4 - hc                                         # CheXzero row
    box(a, X[0][0], yc, X[0][1], hc, "Radiograph", f"{CZ_PX} x {CZ_PX} px")
    box(a, X[1][0], yc, X[1][1], hc, "CheXzero", "CLIP ViT-B/32\nimage embedding", tcol=COL["CheXzero"])
    box(a, X[2][0], yc, X[2][1], hc, "Prompt pair", "\"{f}\" and \"no {f}\", CheXzero text encoder;\ncosine similarity with each")
    box(a, X[3][0], yc, X[3][1], hc, "Score", f"Softmax over the two\nsimilarities; mean over\n{N_CKPT} checkpoints")
    mid = lambda y, h: y + (h - 3.4) / 2
    link([mid(yc, hc)], 0); link([mid(yc, hc)], 1); link([mid(yc, hc)], 2)
    hk = bh(3); yk = 0.0; hd = bh(2); yd = yk + hk + 1.8; top = yd + hd     # CLEAR rows: one encoder, two routes
    box(a, X[0][0], yk, X[0][1], top - yk, "Radiograph", f"{CL_PX} x {CL_PX} px")
    box(a, X[1][0], yk, X[1][1], top - yk, "CLEAR", f"DINOv2 ViT-B/14\n(with registers)\n{CLEAR_DIM}-dimensional\nimage embedding")
    box(a, X[2][0], yd, X[2][1], hd, "Direct route", "\"{f}\" and \"no {f}\", CLEAR text encoder;\ncosine similarity with each", tcol=COL["CLEAR direct"])
    box(a, X[2][0], yk, X[2][1], hk, "Concept route",
        f"Similarity to each of {N_CONC:,} concepts weights its\nSFR-Embedding-Mistral vector ({LLM_DIM:,}-d); the\nnormalised sum is compared with \"{{f}}\" and \"no {{f}}\"",
        tcol=COL["CLEAR concept"])
    box(a, X[3][0], yk, X[3][1], top - yk, "Score", "Softmax over the two\nsimilarities, for each\nroute")
    link([mid(yk, top - yk)], 0); link([mid(yd, hd), mid(yk, hk)], 1); link([mid(yd, hd), mid(yk, hk)], 2)
    assert yc > top + 1.5, (yc, top)

    # ---------- panel b: study design ----------
    b.text(0, yb - 0.2, "b  Study design", fontsize=7.8, fontweight="bold", color=INK, va="top")
    ty = yb - 6.2                                                          # registration timeline
    b.plot([1.0, XW - 1.0], [ty, ty], color=EDGE, lw=1.0, zorder=1)
    for xx, date, txt in [(2.0, "1 Oct 2026", "Protocol registered on OSF, before any performance metric was computed"),
                          (50.0, "3 Oct 2026", "Amendments 1 and 2 filed on OSF, before their new analyses were run")]:
        b.plot([xx], [ty], "o", ms=4.2, color=PILL, zorder=2)
        b.text(xx + 1.2, ty + 0.8, date, fontsize=6.4, fontweight="bold", color=INK, va="bottom")
        b.text(xx + 1.2, ty - 0.8, txt, fontsize=6.0, color=MUTED, va="top")
    C1, C2, C3 = (0, 27.0), (33.5, 28.0), (67.5, XW - 67.6)
    for (x, w), t in zip([C1, C2, C3], ["MODELS, PROMPTS AND RULES", "DATASETS", "ANALYSES"]): header(b, x, w, ty - 5.4, t)
    TOPY = ty - 7.2
    # analyses (right) and datasets (middle), aligned row by row
    yR = TOPY - hR; yA = yR - G - hA; hN = TOPY - yA
    yV = 0.0; yI = yV + hV + G
    hE = yI + hI - yV; yE = 0.0
    assert yA - G >= yI + hI - 0.01, (yA, yI + hI)
    box(b, C3[0], yR, C3[1], hR, "Primary analysis (NIH)",
        f"Macro-AUROC over {N_FIND} findings\nFemale minus male (95% CI)\nAge bands minus 40-60 ({ADJ:.2f}% CI)\nSecondary: posteroanterior only; FNR", pill="Registered")
    box(b, C3[0], yA, C3[1], hA, "Prompts and concepts",
        "NIH: prompt robustness, T1 to T4\nConcept relevance, single-concept\nprobe, representation spread", pill="Amendment 1")
    box(b, C3[0], yE, C3[1], hE, "External comparison",
        "CLEAR concept minus CheXzero,\nmacro-AUROC\nRule fixed before running:\nconsistent with the CLEAR report,\ninconsistent, or inconclusive\nVinDr-CXR: sex and age subgroups\n(exploratory)")
    box(b, C2[0], yA, C2[1], hN, "NIH ChestX-ray14",
        f"Official test split: {RUN['n_test_images']:,} images,\n{RUN['n_test_patients']:,} patients, {N_FIND} text-mined labels\n"
        f"Sex: {N_F_PAT:,} female, {N_M_PAT:,} male patients\nAge: 0-18, 18-40, 40-60, over 60 years\n"
        f"Held-out training patients: {RUN['n_heldout_patients']:,}\n(thresholds, concept selection)\nPatient bootstrap, {N_BOOT:,} resamples", pill="Registered")
    box(b, C2[0], yI, C2[1], hI, "Indiana University collection",
        f"{IND_IMG:,} images, {IND_REP:,} reports\nProblems-field labels; report bootstrap\nRun before Amendment 1, disclosed in it")
    box(b, C2[0], yV, C2[1], hV, "VinDr-CXR",
        f"{VIN_N:,} training images, {VIN_K} findings\nMajority of 3 radiologists\nImage bootstrap; DICOM sex and age", pill="Amendment 2")
    # left column: models, prompts, eligibility rule
    yM = TOPY - hM; yL = 0.0
    yP = yL + hL + G; hP = yM - G - yP
    box(b, C1[0], yM, C1[1], hM, "Models (frozen, zero-shot)",
        f"CheXzero, mean of {N_CKPT} checkpoints\nCLEAR direct route\nCLEAR concept route\nScoring as in panel a")
    for i, m in enumerate(["CheXzero", "CLEAR direct", "CLEAR concept"]):          # same colours and markers as Figs 2 to 7
        b.plot([C1[0] + C1[1] - 1.8], [yM + hM - 4.5 - 0.62 - i * LH], MK[m], color=COL[m], ms=3.6, mec="white", mew=0.3, zorder=4)
    box(b, C1[0], yP, C1[1], hP, "Prompt templates",
        "T1 (registered):\n    {f} / no {f}\nAdded in Amendment 1:\n    T2  chest x-ray showing {f} /\n          chest x-ray showing no {f}\n"
        "    T3  there is {f} / there is no {f}\n    T4  findings consistent with {f} /\n          no evidence of {f}")
    box(b, C1[0], yL, C1[1], hL, "Eligibility rule (registered)",
        f"A finding enters a subgroup contrast\nonly with at least 30 positive images\nfrom at least 10 patients per group\nNIH: {N_ELIG_SEX} findings for sex, {N_ELIG_AGE} for age")
    arrow(b, C1[0] + C1[1] / 2, yP + hP + 0.3, C1[0] + C1[1] / 2, yM - 0.3)
    # connectors
    bus = C1[0] + C1[1] + 3.2
    ys = [mid(yA, hN), mid(yI, hI), mid(yV, hV)]
    ym = mid(yM, hM)
    b.plot([C1[0] + C1[1] + 0.3, bus], [ym, ym], color=MUTED, lw=0.7)
    b.plot([bus, bus], [min(ys), max(ys + [ym])], color=MUTED, lw=0.7)
    for yy in ys: arrow(b, bus, yy, C2[0] - 0.4, yy)
    xr0, xr1 = C2[0] + C2[1] + 0.4, C3[0] - 0.4; xm = (xr0 + xr1) / 2
    b.plot([xr0, xm], [mid(yA, hN)] * 2, color=MUTED, lw=0.7)                    # NIH to both NIH analyses
    b.plot([xm, xm], [mid(yA, hA), mid(yR, hR)], color=MUTED, lw=0.7)
    arrow(b, xm, mid(yR, hR), xr1, mid(yR, hR)); arrow(b, xm, mid(yA, hA), xr1, mid(yA, hA))
    arrow(b, xr0, mid(yI, hI), xr1, mid(yI, hI)); arrow(b, xr0, mid(yV, hV), xr1, mid(yV, hV))
    save(fig, "Figure_1_study_design")
    print("Fig1 values:", dict(N_CKPT=N_CKPT, N_CONC=N_CONC, LLM_DIM=LLM_DIM, CLEAR_DIM=CLEAR_DIM, CZ_PX=CZ_PX, CL_PX=CL_PX,
          N_BOOT=N_BOOT, IND=(IND_IMG, IND_REP), VIN=(VIN_N, VIN_K), N_FIND=N_FIND, ADJ=round(ADJ, 2), ELIG=(N_ELIG_SEX, N_ELIG_AGE),
          PAT=(N_F_PAT, N_M_PAT), RUN=(RUN["n_test_images"], RUN["n_test_patients"], RUN["n_heldout_patients"])))

# ================= Figure 2: NIH per-finding AUROC =================
def fig2():
    order = sorted(FIND, key=lambda f: NIH[f]["CheXzero"][0])
    rows = order + ["MACRO (14)"]
    fig, ax = plt.subplots(figsize=(3.5, 4.6))
    y = np.arange(len(rows))[::-1].astype(float); y[-1] -= 0.6
    off = {"CheXzero": 0.22, "CLEAR concept": 0.0, "CLEAR direct": -0.22}
    for mdl in MODELS:
        for yi, f in zip(y, rows):
            e, lo, hi = NIH[f][mdl]
            ax.plot([lo, hi], [yi + off[mdl]] * 2, color=COL[mdl], lw=0.9, solid_capstyle="round")
            ax.plot(e, yi + off[mdl], MK[mdl], color=COL[mdl], ms=3.6 if f != "MACRO (14)" else 4.4,
                    mec="white", mew=0.4, label=mdl if f == rows[0] else None)
    ax.axvline(0.5, color=MUTED, lw=0.6, ls=(0, (2, 2)))
    ax.axhline(y[-1] + 0.8, color=GRID, lw=0.6)
    ax.set_yticks(y); ax.set_yticklabels([NICE.get(f, "Macro-AUROC (14)") for f in rows])
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_xlim(0.28, 0.9); style_ax(ax, "AUROC (95% CI)")
    ax.legend(loc="upper right", frameon=False, handletextpad=0.3, borderaxespad=0.2)
    save(fig, "Figure_2_nih_per_finding")

# ================= Figure 3: subgroup gaps across templates =================
def fig3():
    contr = [("sex_F-M", "Female minus male\n(95% CI)"), ("age_0-18-40-60", "0-18 minus 40-60 years\n(98.33% CI)"),
             ("age_18-40-40-60", "18-40 minus 40-60 years\n(98.33% CI)"), ("age_60+-40-60", ">60 minus 40-60 years\n(98.33% CI)")]
    fig, axs = plt.subplots(1, 4, figsize=(7.1, 2.5), sharey=True)
    temps = ["T1", "T2", "T3", "T4"]; off = {"CheXzero": 0.22, "CLEAR concept": 0.0, "CLEAR direct": -0.22}
    for ax, (g, title) in zip(axs, contr):
        for ti, t in enumerate(temps):
            yy = 3 - ti
            for mdl in MODELS:
                e, lo, hi = T[(mdl, t)]["gaps"][g]
                excl = lo > 0 or hi < 0
                ax.plot([lo, hi], [yy + off[mdl]] * 2, color=COL[mdl], lw=0.9)
                ax.plot(e, yy + off[mdl], MK[mdl], color=COL[mdl] if excl else "white", mec=COL[mdl], mew=0.8, ms=3.6,
                        label=mdl if (ti == 0 and g == "sex_F-M") else None)
        ax.axvline(0, color=MUTED, lw=0.6)
        ax.set_title(title, color=INK, pad=4); style_ax(ax, "Difference in macro-AUROC")
        ax.set_xlim(-0.09, 0.075)
    axs[0].set_yticks([3, 2, 1, 0]); axs[0].set_yticklabels(["T1 (registered)", "T2", "T3", "T4"])
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.text(0.5, -0.12, "Filled marker: interval excludes zero. Open marker: interval includes zero.", ha="center", fontsize=7, color=MUTED)
    fig.tight_layout(w_pad=0.6)
    save(fig, "Figure_3_subgroups_by_template")

# ================= Figure 4: prompt templates =================
def fig4():
    temps = ["T1", "T2", "T3", "T4"]; tl = ['T1\n"{f}"', 'T2\n"chest x-ray\nshowing {f}"', 'T3\n"there is {f}"', 'T4\n"findings\nconsistent with {f}"']
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.7), gridspec_kw={"width_ratios": [1.15, 1]})
    off = {"CheXzero": -0.2, "CLEAR concept": 0.0, "CLEAR direct": 0.2}
    for mdl in MODELS:
        for i, t in enumerate(temps):
            e, lo, hi = T[(mdl, t)]["macro"]
            a.plot([i + off[mdl]] * 2, [lo, hi], color=COL[mdl], lw=0.9)
            a.plot(i + off[mdl], e, MK[mdl], color=COL[mdl], mec="white", mew=0.4, ms=4.2, label=mdl if i == 0 else None)
    a.set_xticks(range(4)); a.set_xticklabels(tl, fontsize=6.8); a.set_ylabel("Macro-AUROC, NIH (95% CI)")
    a.set_ylim(0.60, 0.75); a.grid(axis="y", color=GRID, lw=0.5); a.set_axisbelow(True)
    a.tick_params(colors=MUTED, length=2.5); a.legend(frameon=False, loc="upper center", ncol=3, fontsize=6.8, columnspacing=1.0, handletextpad=0.3)
    a.set_title("a  Accuracy by prompt template", loc="left", color=INK)
    for j, (key, mdl) in enumerate([("clear_concept_minus_chexzero_", "CLEAR concept"), ("clear_direct_minus_chexzero_", "CLEAR direct")]):
        for i, t in enumerate(temps):
            e, lo, hi = PAIR[key + t]
            xx = i + (-0.12 if j == 0 else 0.12)
            b.plot([xx, xx], [lo, hi], color=COL[mdl], lw=0.9)
            b.plot(xx, e, MK[mdl], color=COL[mdl], mec="white", mew=0.4, ms=4.2)
    b.axhline(0, color=MUTED, lw=0.6)
    from matplotlib.lines import Line2D
    b.legend(handles=[Line2D([], [], marker=MK[m], color=COL[m], ls="", ms=4.2, label=m + " minus CheXzero") for m in ("CLEAR concept", "CLEAR direct")],
             frameon=False, loc="upper left", fontsize=6.8)
    b.set_xticks(range(4)); b.set_xticklabels(temps); b.set_ylabel("Difference from CheXzero (paired 95% CI)")
    b.grid(axis="y", color=GRID, lw=0.5); b.set_axisbelow(True); b.tick_params(colors=MUTED, length=2.5)
    b.set_title("b  CLEAR minus CheXzero, same template", loc="left", color=INK)
    fig.tight_layout(w_pad=2)
    save(fig, "Figure_4_prompt_templates")

# ================= Figure 5: concept audit =================
def fig5():
    order = sorted(FIND, key=lambda f: PROBE[f][0] - PROBE[f][3])
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 3.5), gridspec_kw={"width_ratios": [1.45, 1]}, sharey=True)
    y = np.arange(len(order))[::-1]
    for yi, f in zip(y, order):
        p, plo, phi, conc, dire, cz = PROBE[f]
        a.plot([conc, p], [yi, yi], color=GRID, lw=2.2, solid_capstyle="round", zorder=1)
        a.plot([plo, phi], [yi, yi], color=INK, lw=0.9, zorder=2)
        a.plot(p, yi, "D", color=INK, ms=3.6, zorder=3, label="Single relevant concept" if yi == y[0] else None)
        a.plot(conc, yi, MK["CLEAR concept"], color=COL["CLEAR concept"], ms=3.8, mec="white", mew=0.4, zorder=3,
               label="CLEAR concept route" if yi == y[0] else None)
        a.plot(dire, yi, MK["CLEAR direct"], color=COL["CLEAR direct"], ms=3.8, mec="white", mew=0.4, zorder=3,
               label="CLEAR direct route" if yi == y[0] else None)
    a.axvline(0.5, color=MUTED, lw=0.6, ls=(0, (2, 2)))
    a.set_yticks(y); a.set_yticklabels([NICE[f] for f in order]); a.set_xlim(0.33, 0.95)
    style_ax(a, "Test AUROC"); a.legend(frameon=False, loc="upper left", fontsize=6.8, handletextpad=0.3, bbox_to_anchor=(0.0, 1.0))
    a.set_title("a  Single concept chosen on held-out patients", loc="left", color=INK)
    tp = [REL[f][0] for f in order]
    b.barh(y, tp, height=0.62, color=COL["CLEAR concept"], edgecolor="white", linewidth=0.6)
    for yi, v in zip(y, tp): b.text(v + 0.015, yi, f"{v:.2f}", va="center", fontsize=6.6, color=MUTED)
    b.set_xlim(0, 1); style_ax(b, "Share of top-10 contributing concepts\nthat are relevant (true positives)")
    b.set_title("b  Explanation relevance", loc="left", color=INK)
    fig.tight_layout(w_pad=1.2)
    save(fig, "Figure_5_concept_audit")

# ================= Figure 7: cross-dataset differences =================
def fig6():
    ilog = open(IND_LOG).read()
    allsec = ilog[:ilog.index("-- frontal_only")]                      # primary estimate: all images
    def ind(model):
        m = re.search(model + r"\s+macro [\d.]+ \([\d.]+-[\d.]+\) \| minus CheXzero " + NUM + r" \(" + NUM + " to " + NUM + r"\)", allsec)
        return tuple(float(x) for x in m.groups())
    n_terms = int(re.search(r"\| findings (\d+)", ilog).group(1))
    rows = [("NIH ChestX-ray14, test split", RUN["n_test_images"], N_FIND, (PAIR["clear_concept_minus_chexzero_T1"]), (PAIR["clear_direct_minus_chexzero_T1"]))]
    rows += [("Indiana University collection", IND_IMG, n_terms, ind("clear_concept"), ind("clear_direct"))]
    print("Indiana row:", rows[-1])
    rows += [("VinDr-CXR, " + l.lower() if i else "VinDr-CXR, all images", n, k, c, d) for i, (l, n, k, c, d) in enumerate(VSUB)]
    fig, ax = plt.subplots(figsize=(5.0, 3.1))
    y = np.arange(len(rows))[::-1].astype(float); y[0] += 0.5; y[1] += 0.25          # gaps between datasets
    for yi, (lab, n, k, c, d) in zip(y, rows):
        for mdl, v, o in (("CLEAR concept", c, 0.13), ("CLEAR direct", d, -0.13)):
            ax.plot([v[1], v[2]], [yi + o] * 2, color=COL[mdl], lw=0.9)
            ax.plot(v[0], yi + o, MK[mdl], color=COL[mdl], mec="white", mew=0.4, ms=4.0, label=mdl + " minus CheXzero" if yi == y[0] else None)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_yticks(y); ax.set_yticklabels([f"{lab}\n(n = {n:,}; {k} {'Problems terms' if lab.startswith('Indiana') else 'findings'})" for lab, n, k, c, d in rows], fontsize=6.9)
    style_ax(ax, "Difference in macro-AUROC from CheXzero (95% CI)")
    ax.legend(frameon=False, loc="upper right", fontsize=6.8)
    save(fig, "Figure_7_cross_dataset")

# ================= Figure 6, example images: concept contributions for three random true positives (post hoc) =================
# Reads the Kaggle output of study1_example_images.ipynb (fig7_outputs.zip, unpacked) and the selection file.
EXD = f"{RES}/posthoc_example_images"
def fig_examples():
    import textwrap
    from PIL import Image as _Image
    R = json.load(open(f"{EXD}/fig7_results.json")); EXS = R["examples"]
    SELF = json.load(open(f"{EXD}/fig7_selection.json"))
    for f, e in EXS.items():                                               # consistency checks against the registered run
        assert abs(e["score"] - SELF[f]["score"]) < 2e-5 and e["image"] == SELF[f]["image"]
        assert abs(e["margin_check"][0] - e["margin_check"][1]) <= 0.01 * abs(e["margin_check"][1]) + 1e-6
    NC = R["run_info"]["n_concepts"]
    order = list(EXS)                                                      # Cardiomegaly, Emphysema, Hernia (draw order)
    fig = plt.figure(figsize=(6.55, 7.3))
    REL_C, OTH_C = COL["CLEAR concept"], "#a9a7a1"
    for r, f in enumerate(order):
        e = EXS[f]; top = e["top10"]
        y0 = 1 - (r + 1) / 3
        axi = fig.add_axes([0.0, y0 + 0.03, 0.21, 0.225])
        axi.imshow(np.asarray(_Image.open(f"{EXD}/{f}_{e['image']}")), cmap="gray", vmin=0, vmax=255); axi.axis("off")
        axb = fig.add_axes([0.655, y0 + 0.03, 0.32, 0.225])
        vals = [t["contribution"] for t in top]; yy = np.arange(len(top))[::-1]
        axb.barh(yy, vals, height=0.66, color=[REL_C if t["relevant"] else OTH_C for t in top], edgecolor="white", linewidth=0.4)
        axb.set_yticks(yy); axb.set_yticklabels([textwrap.fill(t["concept"], 56) for t in top], fontsize=6.0, linespacing=0.95)
        axb.set_xlim(0, max(vals) * 1.12); style_ax(axb)
        axb.tick_params(axis="x", labelsize=6.5)
        if r == 2: axb.set_xlabel("Contribution to the score margin", fontsize=7)
        labs = e["labels"].replace("_", " ").replace("|", ", ")
        fig.text(0.0, y0 + 0.305, f"{'abc'[r]}  {f}", fontsize=8, fontweight="bold", color=INK, va="bottom")
        fig.text(0.0, y0 + 0.288, f"Image {e['image']}; NIH labels: {labs}; concept-route score {e['score']:.4f} (held-out threshold {e['threshold']:.4f})",
                 fontsize=6.3, color=MUTED, va="bottom")
        fig.text(0.0, y0 + 0.271, f"Sum of the contributions of all {NC:,} concepts {e['contribution_sum']:+,.1f}; sum of the 10 shown {sum(vals):+.2f}",
                 fontsize=6.3, color=MUTED, va="bottom")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=REL_C, label="Matches the finding's relevance pattern (Amendment 1)"),
                        Patch(color=OTH_C, label="Does not match")], loc="upper center", ncol=2, frameon=False,
               fontsize=6.8, bbox_to_anchor=(0.5, -0.005))
    save(fig, "Figure_6_example_images")

for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig_examples): f()
print("NIH macro:", NIH["MACRO (14)"]); print("pairs:", {k: v for k, v in PAIR.items() if "minus_chexzero" in k})
print("VinDr maj:", VMAJ); print("files:", sorted(os.listdir(OUT)))
