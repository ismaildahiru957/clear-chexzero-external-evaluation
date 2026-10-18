# CLEAR and CheXzero: a preregistered external evaluation

This repository holds the code, run logs and per-image model scores behind the paper *Zero-shot performance of a concept-grounded chest radiograph foundation model depends on the evaluation dataset: a preregistered external evaluation of CLEAR and CheXzero*. Citation details will be added when the paper is published.

We compared two chest radiograph foundation models in zero-shot classification:

- **CLEAR** (Han et al., Nat Biomed Eng 2026), scored through its concept route and its direct route;
- **CheXzero** (Tiu et al., Nat Biomed Eng 2022), as the ensemble of its 10 released checkpoints.

The datasets were NIH ChestX-ray14, the Indiana University collection and VinDr-CXR. Neither model was trained or fine-tuned here.

## Registration

| What | Where | When |
|---|---|---|
| Preregistration (OSF Preregistration template) | https://osf.io/dsbem | 1 October 2026, before any performance metric was computed on test images |
| Amendment 1 (prompt robustness, concept audit) and Amendment 2 (VinDr-CXR) | https://osf.io/ty7d5 | 3 October 2026, before their runs |

Everything labelled *post hoc* below was decided after the primary results were known. The paper marks it the same way.

## Model versions

| Component | Version |
|---|---|
| CLEAR code | https://github.com/peterhan91/CLEAR, commit `57d2eae57ddabf8e56c8013ec675336d71b3d8e7` |
| CLEAR weights and concept bank | Hugging Face `peterhan91/CLEAR`, revision `142f169c8b006b0f28cf55c6a237c3003db5ed8f` |
| Concept text encoder | `Salesforce/SFR-Embedding-Mistral` (Hugging Face) |
| CheXzero | the 10 released checkpoints, from the CheXzero authors' Google Drive folder (link in `notebooks/00_fetch_chexzero_weights.ipynb`), loaded with OpenAI CLIP (https://github.com/openai/CLIP) |

The checkpoint file names and sizes are listed in each run's `run_info`. Every GPU run with saved outputs recorded the CLEAR commit above, and every one except the resizing check also recorded the Hugging Face revision above. The resizing check reproduced the saved VinDr-CXR scores exactly on 256 images before scoring anything new. The example-image run matched the registered scores of its three images within 3 x 10^-7.

## Data

No images are included, apart from three NIH ChestX-ray14 test images used in Fig. 6.

| Dataset | Source used | What this repository contains |
|---|---|---|
| NIH ChestX-ray14 | Kaggle dataset "NIH Chest X-rays" (original: https://nihcc.app.box.com/v/ChestXray-NIHCC) | Cohort file (labels, age, sex and view from `Data_Entry_2017.csv`, test or held-out set), per-image scores |
| Indiana University collection | Kaggle dataset "Chest X-rays (Indiana University)" by raddar (original: Open-i, https://openi.nlm.nih.gov/) | Per-image scores and an image index (report ID and file name). No report text and no labels |
| VinDr-CXR | Kaggle competition "VinBigData Chest X-ray Abnormalities Detection" (https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection), training split | Per-image scores and an image index (image ID). No labels and no DICOM header data |

NIH ChestX-ray14 is provided by the NIH Clinical Center. Please cite Wang et al., CVPR 2017, when you use it.

The Indiana and VinDr-CXR label files are left out on purpose:
- the Indiana file holds the reports' Problems text word for word;
- the VinDr-CXR file holds the competition labels and the age and sex read from the DICOM headers.

We did not have the right to redistribute either. Each file is rebuilt exactly by cell 2 of its notebook from the provider's own files. To check a rebuilt file against ours, compare its SHA-256:

| File | Written by | SHA-256 |
|---|---|---|
| Indiana `cohort.csv` (7,466 rows) | `notebooks/03_indiana.ipynb` | `6e4da8ae1c0ade9f5d2b6963b7e2c62d0d822a2b9241954c289c66c7ca54d313` |
| VinDr-CXR `cohort.csv` (15,000 rows) | `notebooks/05_amendment2_vindr.ipynb` | `799e2df84a97413d16aebb955b8b9b207a04327bcbedd035a539747415fcbd37` |

The rows of the Indiana and VinDr-CXR score arrays follow the row order of these files, which is also the order of `image_index.csv`.

## Repository layout

```
notebooks/   the Kaggle notebooks, exactly as run (numbered in run order)
analysis/    local analysis, figure script and recheck scripts
results/     outputs of each run: scores, results files, run logs
figures/     Figs. 1 to 7 of the paper, as produced by analysis/make_figures.py
```

| Notebook | Role | Run (UTC) | Saved outputs in `results/` |
|---|---|---|---|
| `00_fetch_chexzero_weights` | Copies the 10 CheXzero checkpoints into a private Kaggle dataset | before the registered run | none |
| `01_smoke_test` | 100-image technical check of both models | before registration | none |
| `02_nih_registered_scoring` | Registered run: scores the NIH test split and held-out patients | 1 Oct 2026, 15:25 | `nih_registered/` |
| `03_indiana` | Indiana University comparison, decision rule fixed in the notebook before the run | 2 Oct 2026, 16:37 | `indiana/` |
| `04_amendment1_prompts_and_audit` | Amendment 1: four prompt templates, concept audit | 3 Oct 2026, 09:56 | `amendment1/` |
| `05_amendment2_vindr` | Amendment 2: VinDr-CXR | 3 Oct 2026, 10:42 | `amendment2_vindr/` |
| `06_posthoc_followup` | Post hoc: paired contrasts between templates and models (CPU) | 4 Oct 2026 | `posthoc_followup/` |
| `07_posthoc_vindr_subsets` | Post hoc: VinDr-CXR with and without recorded demographics (CPU) | 4 Oct 2026 | `posthoc_vindr_subsets/` |
| `08_posthoc_vindr_resizing` | Post hoc: VinDr-CXR scored with CLEAR's own resizing | 11 Oct 2026, 11:40 | `posthoc_vindr_resizing/` |
| `09_posthoc_example_images` | Post hoc: concept contributions for the three Fig. 6 images | 14 Oct 2026, 04:25 | `posthoc_example_images/` |

The GPU notebooks ran on Kaggle with two NVIDIA T4 GPUs and internet access. Their Python, PyTorch and Transformers versions are in each `run_info`:
- the NIH and Indiana runs used PyTorch 2.10.0;
- the later runs used PyTorch 2.11.0.

Each notebook finds its inputs under `/kaggle/input` and writes to `/kaggle/working`. Notebooks 04 to 09 take earlier notebooks' outputs as inputs, as their first cells explain.

## Reproducing the analyses

The local scripts need Python 3.11 or later and the packages in `requirements.txt`. Run them from the repository root.

```bash
pip install -r requirements.txt

# registered NIH analysis: writes report.txt, results.json, the tables and the bootstrap files
NIH_OUT=/tmp/nih python analysis/nih_registered_analysis.py
NIH_OUT=/tmp/nih python analysis/nih_registered_summary.py

# all seven figures, from the files in results/
FIG_OUT=/tmp/figs python analysis/make_figures.py

# recompute saved results from the per-image scores
python analysis/recheck/recheck_amendment1.py
python analysis/recheck/recheck_indiana.py        path/to/indiana/cohort.csv
python analysis/recheck/recheck_vindr.py          path/to/vindr/cohort.csv
python analysis/recheck/recheck_vindr_resizing.py path/to/vindr/cohort.csv
```

Leave out `NIH_OUT` and `FIG_OUT` to overwrite the copies in `results/nih_registered/` and `figures/`.

`make_figures.py` reads every number from `results/`. No value is typed in by hand.

## Checks run before release (18 October 2026)

We reran every analysis here from the files in this repository and compared the output with what is saved:

| Check | Result |
|---|---|
| Registered NIH analysis | `report.txt`, `results.json`, the four tables and both bootstrap files identical |
| Amendment 1: macro-AUROC, intervals, subgroup gaps and per-finding AUROC for 3 models x 4 templates, and the held-out thresholds | 194 of 194 values identical |
| Amendment 1: template 1 scores against the registered run's scores | within 7 x 10^-7 on all 34,033 images |
| Indiana: 5 scores, all and frontal images, plus 510 per-term AUROCs | identical |
| VinDr-CXR: 4 scores, majority and any-radiologist labels | identical |
| VinDr-CXR resizing check: 5 subsets | 90 of 90 items identical |
| Post hoc follow-up (notebook 06 rerun on CPU) | `followup_results.json` identical |
| Figures 1 to 7 | pixel-identical to the figures in the paper |
| Notebooks 02 to 05, 08 and 09 | byte-identical to the output of their generator scripts, each last changed before its run started |

The concept-audit parts of Amendment 1 cannot be recomputed from saved scores, because they need CLEAR's image features. These are the explanation relevance, the single-concept probe and the representation spread. We checked them against the Amendment 1 run log instead: all 144 values printed there agree with `amend1_results.json`.

## Notes

- **VinDr-CXR subsets (notebook 07).** The original run on 4 October printed its summary and saved no file. `results/posthoc_vindr_subsets/rerun_output_2026-10-18.txt` is a rerun of the same notebook on the saved VinDr-CXR outputs, made on 18 October 2026. It reproduces the numbers reported in the paper, and the figure script reads it.
- **Registered sensitivity analysis S2.** This analysis, against the radiologist labels released by Google for part of the NIH test split, was not carried out because the labels could not be obtained. The paper reports it as registered but not done.
- **Path changes.** Two local scripts were edited after the analysis only to point at this repository's folders: `analysis/nih_registered_analysis.py` (lines 9 to 12) and the first line of `analysis/nih_registered_summary.py`. `analysis/make_figures.py` was also re-pointed to read `results/`. In the same edit, its VinDr-CXR subset values, which had been transcribed from the printed summary, are now read from the rerun file above. The figures did not change.
- **Kaggle run logs** are included as downloaded. Early attempts that stopped with errors before producing outputs are not included.

## Licence

The code is released under the MIT licence (`LICENSE`). The result files we produced (scores, results files, tables and logs) are released under CC BY 4.0.

The datasets and model weights remain under their providers' own terms.

## Citation

See `CITATION.cff`. When using results from this repository, please also cite the CLEAR, CheXzero and dataset papers listed in the paper.
