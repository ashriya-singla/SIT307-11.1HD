# SIT307 Task 11.1HD — Reproducing and Improving Stacking-Based Heart-Disease Prediction

Reproduction of **Bhagat, Sharma & Agarwal (2025)**, *"An efficient stacking-based ensemble technique for early heart attack prediction"*, Multimedia Tools and Applications 84:36351–36375, plus a proposed improvement, **LF-DPS (Leakage-Free, Diversity-Pruned Stacking)**.

## Repository layout

```
data/heart.csv                       Kaggle "Heart Disease Dataset" (1,025 rows, 14 columns) used by the paper
src/common.py                        paths, preprocessing, model factories, metrics, paper's reported numbers
src/part1_reproduce.py               Part 1: E1 headline, E2 30 seeds, E3 meta-learners, E4 leakage, E5 de-duplicated
src/part2_proposed.py                Part 2: LF-DPS + ablation A0..A3 under 5x5 repeated nested CV, stats tests
src/part2_estimation_validity.py     Part 2 / E6: estimated vs true accuracy on locked external patients
src/make_report.py                   builds report/SIT307_11.1HD_Technical_Report.pdf from results/
src/make_video_script.py             builds report/SIT307_11.1HD_Video_Script.pdf (numbers from results/)
notebooks/HD_walkthrough.ipynb       guided walk-through that runs everything and shows outputs
run_all.py                           one-command end-to-end run
results/                             all CSV/JSON outputs and figures (regenerated on every run)
```

## Install

Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run / reproduce

```bash
python run_all.py              # Part 1 -> Part 2 -> E6 -> PDF report
python run_all.py --report     # only rebuild the PDF from existing results/
```

Or open `notebooks/HD_walkthrough.ipynb` and choose **Restart & Run All**.

Approximate runtime on a 4-core laptop: Part 1 about 3 min, Part 2 about 10–15 min, E6 about 10 min. On a single core, the whole run takes about 1 hour.

All randomness is seeded (split seeds 0–29 and 42, outer CV seed 42, E6 seeds 100–104), so results are identical across runs on the same library versions.

**XGBoost note:** if `xgboost` is not installed, the code automatically substitutes scikit-learn's `HistGradientBoostingClassifier` and records this in `results/data_audit.json`. The report states which backend was used. Install `xgboost` (it is in `requirements.txt`) for the faithful reproduction. Because the report is generated from `results/`, every table and figure updates automatically after a rerun.

## Key outputs

| File | Content |
|---|---|
| `results/part1_headline.csv` | E1: all 7 models on the seed-42 80/20 split |
| `results/part1_seeds_summary.csv` | E2: mean/SD over 30 splits (duplicates kept) |
| `results/part1_meta_summary.csv` | E3: meta-learner sensitivity |
| `results/part1_leakage.csv` | E4: accuracy on seen vs unseen test rows |
| `results/part1_dedup_summary.csv` | E5: paper protocol on 302 unique patients |
| `results/part2_summary.csv`, `part2_stats.csv` | Ablation A0–A3 + corrected t-tests / Wilcoxon |
| `results/part2_validity_summary.csv` | E6: estimated vs true external accuracy |
| `results/figures/*.png` | All figures used in the report |

## Before submitting

1. Run `python run_all.py` on your own machine with `xgboost` installed.
2. Edit the `STUDENT` block at the top of `src/make_report.py` (name, ID, video link, code link) and the GenAI acknowledgement text, then run `python run_all.py --report`.
3. Run the notebook top-to-bottom and save it with outputs.
