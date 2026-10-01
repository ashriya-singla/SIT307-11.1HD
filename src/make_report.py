
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import matplotlib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from common import FIGS, METRICS, MODEL_ORDER, PAPER, RESULTS, ROOT

# ======================================================================
STUDENT = {
    "name": "Ashriya Singla",
    "id": "225682089",
    "video_url": "[INSERT UNLISTED YOUTUBE LINK]",
    "code_url": "[INSERT GITHUB / ONEDRIVE LINK]",
}
# ======================================================================

OUT = ROOT / "report" / "SIT307_11.1HD_Technical_Report.pdf"

# ---------------------------------------------------------------- fonts (bundled with matplotlib)
_ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for name, file in [("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"),
                   ("DV-I", "DejaVuSans-Oblique.ttf"), ("DV-BI", "DejaVuSans-BoldOblique.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(_ttf / file)))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontName="DV", fontSize=9.3, leading=13.2,
                      alignment=TA_JUSTIFY, spaceAfter=5)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=7.6, leading=9.6, alignment=0, spaceAfter=0)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=8, leading=10.5, alignment=TA_CENTER,
                     textColor=colors.HexColor("#333333"), spaceAfter=9)
H1 = ParagraphStyle("h1", parent=BODY, fontName="DV-B", fontSize=13, leading=16, spaceBefore=10,
                    spaceAfter=6, alignment=0, textColor=colors.HexColor("#1F3A5F"))
H2 = ParagraphStyle("h2", parent=H1, fontSize=10.8, leading=13.5, spaceBefore=8, spaceAfter=4)
H3 = ParagraphStyle("h3", parent=H1, fontSize=9.6, leading=12, spaceBefore=5, spaceAfter=3,
                    textColor=colors.black)
TITLE = ParagraphStyle("title", parent=H1, fontSize=17, leading=22, alignment=TA_CENTER, spaceAfter=8)
REF = ParagraphStyle("ref", parent=BODY, fontSize=8.2, leading=10.8, leftIndent=16, firstLineIndent=-16,
                     alignment=0, spaceAfter=2)


# ---------------------------------------------------------------- load results
def csv(name, **kw):
    return pd.read_csv(RESULTS / name, **kw)


def js(name):
    with open(RESULTS / name) as f:
        return json.load(f)


audit = js("data_audit.json")
head = csv("part1_headline.csv", index_col=0)
s_leak = csv("part1_seeds_summary.csv", index_col=0)
s_dedup = csv("part1_dedup_summary.csv", index_col=0)
leak = csv("part1_leakage.csv", index_col=0)
leak_s = js("part1_leakage_summary.json")
meta = csv("part1_meta_summary.csv", index_col=0)
imp = csv("part1_rf_importance.csv", index_col=0).iloc[:, 0].sort_values(ascending=False)
p2 = csv("part2_summary.csv", index_col=0)
p2f = csv("part2_folds.csv")
p2s = csv("part2_stats.csv")
p2m = js("part2_meta.json")
freq = csv("part2_selection_freq.csv", index_col=0).iloc[:, 0]
corr = csv("part2_oof_corr.csv", index_col=0)
params = csv("part2_tuned_params.csv")
val = csv("part2_validity_summary.csv", index_col=0)
valm = js("part2_validity_meta.json")

M = list(p2.index)                       # ablation method names (order as run)
A0, A1, A2, A2b, A3, BS = M
XGBB = audit["xgb_backend"]
USING_XGB = XGBB.startswith("xgboost")


def f3(x): return f"{x:.3f}"
def f4(x): return f"{x:.4f}"
def pc(x, d=2): return f"{100 * x:.{d}f}%"
def ms(m, s): return f"{m:.3f} ± {s:.3f}"


def stat(comp, met):
    r = p2s[(p2s.comparison == f"A3 vs {comp}") & (p2s.metric == met)].iloc[0]
    return r.mean_diff, r.p_corrected, r.p_wilcoxon


def sig(p): return "statistically significant" if p < 0.05 else "not statistically significant"


# ---------------------------------------------------------------- flowable helpers
story = []
P = lambda t, st=BODY: story.append(Paragraph(t, st))
fig_no = [0]
tab_no = [0]


def figure(path, caption, width=16.5):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(path)).getSize()
    w = width * cm
    fig_no[0] += 1
    story.append(KeepTogether([Image(str(path), width=w, height=w * ih / iw),
                               Paragraph(f"<b>Fig. {fig_no[0]}.</b> {caption}", CAP)]))


def table(data, caption, col_widths=None, highlight_rows=(), font=7.6, first_col_left=True):
    tab_no[0] += 1
    cells = [[Paragraph(str(c), SMALL) if isinstance(c, str) and len(c) > 28 else c for c in row]
             for row in data]
    t = Table(cells, colWidths=col_widths, repeatRows=1, hAlign="CENTER")
    style = [("FONT", (0, 0), (-1, -1), "DV", font), ("FONT", (0, 0), (-1, 0), "DV-B", font),
             ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3A5F")),
             ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
             ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#AAAAAA")),
             ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5F9")]),
             ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2)]
    if not first_col_left:
        style.append(("ALIGN", (0, 0), (0, -1), "CENTER"))
    for r in highlight_rows:
        style += [("BACKGROUND", (0, r), (-1, r), colors.HexColor("#E3F1E6")),
                  ("FONT", (0, r), (-1, r), "DV-B", font)]
    t.setStyle(TableStyle(style))
    story.append(KeepTogether([Paragraph(f"<b>Table {tab_no[0]}.</b> {caption}", CAP), t, Spacer(1, 8)]))


# ======================================================================
# Title block
# ======================================================================
P("Leakage, Duplicates and the Limits of Stacking: Reproducing and Improving a "
  "Stacking Ensemble for Heart-Disease Prediction", TITLE)
P(f"SIT307 Machine Learning — Task 11.1HD (Machine Learning Mini-Research)<br/>"
  f"{STUDENT['name']} ({STUDENT['id']}) — {date.today():%d %B %Y}",
  ParagraphStyle("sub", parent=BODY, alignment=TA_CENTER, fontSize=9.5))
table([["Deliverable", "Link"],
       ["Video presentation (≤ 5 min, unlisted)", STUDENT["video_url"]],
       ["Code, data, notebook and README", STUDENT["code_url"]]],
      "Submission links.", col_widths=[6.5 * cm, 9.5 * cm])

g_d = s_dedup.loc["Stacking", "Accuracy_mean"]
v_paper = val.loc["Paper protocol (paper stacking)"]
v_lf = val.loc["LF protocol (LF-DPS)"]
P("<b>Abstract.</b> Bhagat <i>et al.</i> [1] report that a 5-fold stacking ensemble of six classifiers "
  f"predicts heart disease with 98.53% accuracy on the 1,025-record Kaggle heart-disease dataset. "
  f"I reproduce the full study (six base learners, stacking, all reported metrics) and obtain "
  f"{pc(head.loc['Stacking', 'Accuracy'])} on an 80/20 split with seed 42 and "
  f"{pc(s_leak.loc['Stacking', 'Accuracy_mean'])} ± {pc(s_leak.loc['Stacking', 'Accuracy_std'])} over 30 random splits, "
  f"confirming that the headline number is reproducible. A data audit, however, shows that "
  f"{audit['n_duplicates']} of the {audit['n_rows']} rows are exact duplicates (only {audit['n_unique']} unique patients), "
  f"so on average {pc(leak_s['mean_frac_test_seen'], 1)} of test rows also appear in the training set. "
  f"When the same pipeline is evaluated on unique patients its accuracy falls to {pc(g_d)}. "
  f"I then propose <b>LF-DPS</b> (Leakage-Free, Diversity-Pruned Stacking), which combines patient-level "
  f"de-duplication, repair of sentinel-coded missing values, feature-type-aware encoding, nested "
  f"hyper-parameter optimisation and greedy selection of complementary base learners, evaluated with "
  f"5×5 repeated nested cross-validation and corrected resampled t-tests. On locked external patients, "
  f"the paper's protocol over-estimates accuracy by {100 * v_paper.optimism_Accuracy_mean:.1f} points, whereas "
  f"LF-DPS's internal estimate is within {abs(100 * v_lf.optimism_Accuracy_mean):.1f} points of its true external accuracy, "
  f"using a mean of {p2m['mean_stack_size']:.1f} base learners instead of six.")

# ======================================================================
P("1. Introduction", H1)
P("Cardiovascular disease causes about 17.9 million deaths per year, and early risk stratification "
  "from routinely collected clinical variables is a long-standing machine-learning application "
  "[1], [10], [11]. Numerous studies on the UCI/Kaggle heart-disease data report accuracies above 95% "
  "[1], [17], [18]. This is far higher than the roughly 77–85% typically reported on the original Cleveland "
  "cohort [11]. Such a gap should prompt scrutiny of how models were validated, because data leakage is "
  "one of the most common causes of irreproducible ML-based science [7], [8]. This report reviews and "
  "reproduces one such paper [1] (Part 1) and designs, implements and evaluates an improved learning "
  "pipeline that targets its key limitation (Part 2). Section 7 summarises the video presentation (Part 3).")

# ======================================================================
P("2. Part 1 — Critical Review of the Selected Paper", H1)
P("2.1 Research problem, motivation and research gap", H2)
P("The paper addresses binary prediction of heart disease (target = 1) from 13 clinical attributes. Its "
  "motivation is that conventional risk scores (e.g., Framingham) predict individual risk poorly, while "
  "ML can exploit patterns in electronic health data [1]. The related-work table in [1] summarises "
  "state-of-the-art results ranging from 85.48% (majority-vote ensemble [12]) and 88.47% (HRFLM hybrid RF + "
  "linear model [2]) to 92.37% (FCMIM-SVM with LOSO validation [10]) and 97.89% (Relief + RF on merged "
  "datasets [9]). The gap the authors claim is that a heterogeneous stacking ensemble of six standard classifiers, "
  "with a meta-classifier trained on their predictions, can outperform both the individual classifiers and "
  "these prior methods. The claimed contributions are (i) a merged dataset from four sources, (ii) six "
  "classifiers, (iii) 5-fold stacking, and (iv) a comparative evaluation.")

P("2.2 Dataset and feature set", H2)
P(f"The study uses the Kaggle \"Heart Disease Dataset\" [19], a 1,025-row file derived from the UCI "
  f"Cleveland, Hungary, Switzerland and VA Long Beach databases [10]. It contains 13 predictors plus the "
  f"target (Table 2). My audit (Table 3) reveals two properties that the paper does not report. First, only "
  f"{audit['n_unique']} rows are unique: every patient occurs "
  f"{min(audit['copies_per_patient'])}–{max(audit['copies_per_patient'])} times, and no duplicate group has "
  f"conflicting labels, so the file is effectively the {audit['n_unique']}-patient Cleveland data replicated. "
  f"Second, although the paper's own attribute table gives ca ∈ [0, 3] and thal ∈ [1, 3], the data contain "
  f"ca = 4 ({audit['ca_eq_4_unique']} patients) and thal = 0 ({audit['thal_eq_0_unique']} patients). These counts "
  f"match exactly the 4 and 2 missing ('?') values of the original Cleveland file [10], so they are "
  f"disguised missing values rather than real categories. The paper states that it imputes missing "
  f"values with linear regression, but there are no explicit missing values to impute.")
table([["Attribute", "Description", "Type (used in Part 2)"],
       ["age, trestbps, chol, thalach, oldpeak", "Age; resting BP; cholesterol; max heart rate; ST depression", "Continuous"],
       ["sex, fbs, exang", "Sex; fasting blood sugar > 120 mg/dl; exercise-induced angina", "Binary"],
       ["cp, restecg, slope, thal", "Chest-pain type; resting ECG; ST slope; thalassaemia", "Nominal (one-hot)"],
       ["ca", "Number of major vessels coloured by fluoroscopy (0–3)", "Ordinal count"],
       ["target", "1 = heart disease, 0 = no disease", "Label"]],
      "Feature set (13 predictors) used by the paper [1] and in this study.",
      col_widths=[4.6 * cm, 8 * cm, 3.4 * cm])
table([["Property", "Value"],
       ["Rows / predictors", f"{audit['n_rows']} / {audit['n_features']}"],
       ["Unique rows (patients)", f"{audit['n_unique']}"],
       ["Exact duplicate rows", f"{audit['n_duplicates']} ({pc(audit['n_duplicates'] / audit['n_rows'], 1)})"],
       ["Copies per patient (count of patients)", ", ".join(f"{k}× : {v}" for k, v in audit["copies_per_patient"].items())],
       ["Class balance, all rows (0 / 1)", f"{audit['class_counts_full']['0']} / {audit['class_counts_full']['1']}"],
       ["Class balance, unique rows (0 / 1)", f"{audit['class_counts_unique']['0']} / {audit['class_counts_unique']['1']}"],
       ["Explicit NaN values", f"{audit['explicit_missing']}"],
       ["Sentinel ca = 4 (rows / patients)", f"{audit['ca_eq_4_rows']} / {audit['ca_eq_4_unique']}"],
       ["Sentinel thal = 0 (rows / patients)", f"{audit['thal_eq_0_rows']} / {audit['thal_eq_0_unique']}"]],
      "Data audit of heart.csv (results/data_audit.json).", col_widths=[7 * cm, 6 * cm])

P("2.3 Machine-learning methods and experimental protocol", H2)
P("The pipeline in [1] (Fig. 1 and Table 8 of the paper) is as follows. The data are split into training "
  "and testing sets and then pre-processed (linear-regression imputation, normalisation, categorical "
  "encoding). Six base classifiers are trained: Logistic Regression (LR), Decision Tree (DT), Random "
  "Forest (RF), Extreme Gradient Boosting (XGB), Gaussian Naive Bayes (NB) and K-Nearest Neighbours (KNN). "
  "Their predictions are then stacked [2], [3] with a meta-classifier using 5-fold stacking. The paper does "
  "not report the split ratio, the random seed, whether the split is stratified, any hyper-parameters, or "
  "the meta-classifier. Only a single train/test evaluation is reported, with no repeated resampling, "
  "confidence intervals or significance tests.")

P("2.4 Evaluation metrics and reported results", H2)
P("The paper reports Accuracy, Precision, Recall, F1, Specificity, MCC and AUC (Table 4). The reported "
  "stacking accuracy of 0.9853 equals 202/205, which implies a 20% test split of 1,025 rows. Precision = 1 "
  "and Recall = 0.9727 then imply TP = 107, FN = 3, FP = 0 and TN = 95. Three internal inconsistencies weaken "
  "the reported evidence. (i) In Table 11 of [1], the Sensitivity column is identical to Accuracy, "
  "Specificity to F1, and MCC to Precision for every model; for example, MCC = 1.000 is reported alongside "
  "3 false negatives. Recomputing MCC from the implied confusion matrix gives 0.971, so these columns "
  "appear to be copy errors. (ii) The abstract states that RF and DT reach 92.68% and XGB 90.73%, while the "
  "introduction states that XGB is best at 93.17% and stacking reaches 96.58%. (iii) Contribution 1 "
  "claims a newly merged dataset, but the data are the public Kaggle file [19]. In the comparisons below I "
  "therefore use only the internally consistent columns (Accuracy, Precision, Recall, F1, AUC).")
table([["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"]] +
      [[m] + [f4(PAPER[m][k]) for k in ["Accuracy", "Precision", "Recall", "F1", "AUC"]] for m in MODEL_ORDER],
      "Results reported by Bhagat et al. [1] (Table 11; consistent columns only).",
      col_widths=[2.6 * cm] + [2.2 * cm] * 5, highlight_rows=(7,))

# ======================================================================
P("3. Part 1 — Reproduction Study", H1)
P("3.1 Implementation and justified assumptions", H2)
xgb_note = ("XGBoost " + XGBB.split()[1]) if USING_XGB else \
    "scikit-learn's HistGradientBoostingClassifier (histogram gradient boosting) as a stand-in because " \
    "xgboost was unavailable in the execution environment; installing xgboost switches it back automatically"
P(f"All code is written in Python with scikit-learn [12]. XGB is implemented with {xgb_note} [13]. Where [1] is "
  "silent, I made the assumptions in Table 5. Each one is chosen to be the most likely reading of the "
  "paper and is supported by evidence from the paper itself or by standard practice.")
table([["Aspect", "Assumption", "Justification"],
       ["Split", "80/20, random, non-stratified", "0.9853 = 202/205 implies a 205-row test set; the implied 110 positives (vs. ~105 under stratification) suggests a plain random split."],
       ["Seed", "42 for headline; 30 seeds (0–29) for robustness", "Seed unreported; 42 is the conventional default; multiple seeds quantify split variance."],
       ["Pre-processing", "Linear-regression IterativeImputer + StandardScaler fitted on training data", "Sec. 3.1 of [1]; the imputer is a no-op because there are no NaNs."],
       ["Encoding", "Keep 13 label-coded features", "Fig. 2 of [1] shows importances for the 13 original features, so no one-hot expansion was used."],
       ["Hyper-parameters", "Library defaults (e.g., RF 100 trees, KNN k = 5)", "None reported; defaults are what an unspecified model would use."],
       ["Stacking", "StackingClassifier, all six learners, 5-fold, predict_proba, LR meta", "The paper says 5-fold stacking; LR is the most common meta-learner [3], [15]. E3 tests alternatives."],
       ["Threshold", "0.5 on predicted probability", "Standard; no other threshold reported."]],
      "Implementation assumptions for unreported details.", col_widths=[2.9 * cm, 5.0 * cm, 8.3 * cm])

P("Six experiments were run (src/part1_reproduce.py). E1 is the headline reproduction (seed 42). E2 "
  "repeats E1 over 30 random splits. E3 varies the meta-learner. E4 audits duplicate leakage. E5 repeats "
  "the protocol on the 302 unique patients. E6 (Section 4.4) compares estimated with true external "
  "accuracy.", BODY)

P("3.2 Headline reproduction (E1)", H2)
d_stack = head.loc["Stacking", "Accuracy"] - PAPER["Stacking"]["Accuracy"]
match_txt = ("reproduces the paper's headline accuracy exactly (to within one test sample)"
             if abs(d_stack) < 0.005 else f"differs from the paper's headline by {100 * d_stack:+.2f} points")
P(f"Table 6 compares E1 with the paper. The reproduced stacking ensemble reaches "
  f"{pc(head.loc['Stacking', 'Accuracy'])} accuracy with precision {f3(head.loc['Stacking', 'Precision'])} "
  f"(TP={int(head.loc['Stacking', 'TP'])}, FN={int(head.loc['Stacking', 'FN'])}, FP={int(head.loc['Stacking', 'FP'])}, "
  f"TN={int(head.loc['Stacking', 'TN'])}), which {match_txt}. The tree-based learners (DT, RF, XGB) reach similar "
  f"accuracy on their own, however ({pc(head.loc['DT', 'Accuracy'])}, {pc(head.loc['RF', 'Accuracy'])}, "
  f"{pc(head.loc['XGB', 'Accuracy'])}). So in my reproduction stacking does not add accuracy over its best "
  f"base learner, unlike in the paper. The linear and probabilistic models (LR, NB, KNN) are within a few "
  f"points of the paper on this split, and within 1–3 points on average over 30 splits (Table 7).")
rows = [["Model", "Paper Acc", "Repro Acc", "Paper F1", "Repro F1", "Paper AUC", "Repro AUC", "Repro Prec", "Repro Rec"]]
for m in MODEL_ORDER:
    rows.append([m, f4(PAPER[m]["Accuracy"]), f4(head.loc[m, "Accuracy"]), f4(PAPER[m]["F1"]),
                 f4(head.loc[m, "F1"]), f4(PAPER[m]["AUC"]), f4(head.loc[m, "AUC"]),
                 f4(head.loc[m, "Precision"]), f4(head.loc[m, "Recall"])])
table(rows, "E1: paper vs reproduced results on the 80/20 split (seed 42, 205 test rows).",
      col_widths=[1.9 * cm] + [1.8 * cm] * 8, highlight_rows=(7,))
figure(FIGS / "p1_confusion.png", "E1 confusion matrices on the 205-row test split (seed 42).", width=16)
figure(FIGS / "p1_roc.png", "E1 ROC curves for all reproduced models (seed 42).", width=9.5)

P("3.3 Robustness to the random split (E2) and meta-learner choice (E3)", H2)
rows = [["Model", "Accuracy", "Precision", "Recall", "F1", "AUC", "Train acc"]]
for m in MODEL_ORDER:
    rows.append([m] + [ms(s_leak.loc[m, f"{k}_mean"], s_leak.loc[m, f"{k}_std"])
                       for k in ["Accuracy", "Precision", "Recall", "F1", "AUC"]] +
                [f3(s_leak.loc[m, "train_acc_mean"])])
table(rows, "E2: mean ± SD over 30 random 80/20 splits of the 1,025-row file (paper protocol).",
      col_widths=[1.9 * cm] + [2.5 * cm] * 5 + [1.7 * cm], highlight_rows=(7,))
tree_mean = s_leak.loc[["DT", "RF", "XGB"], "Accuracy_mean"]
P(f"Across 30 splits (Table 7), the trees achieve {pc(tree_mean.min(), 1)}–{pc(tree_mean.max(), 1)} mean accuracy "
  f"with a training accuracy of {f3(s_leak.loc['DT', 'train_acc_mean'])} for DT. Stacking averages "
  f"{pc(s_leak.loc['Stacking', 'Accuracy_mean'])}, so the paper's 98.53% is a typical outcome of this "
  f"protocol rather than a lucky split. The paper's DT and RF accuracies (92.68%) are "
  f"{100 * (tree_mean.min() - .9268):.1f}–{100 * (tree_mean.max() - .9268):.1f} points below my defaults. This is "
  f"consistent with the authors having used depth-limited trees, a detail that is not reported. For LR, NB and "
  f"KNN, the 30-split means ({pc(s_leak.loc['LR', 'Accuracy_mean'])}, {pc(s_leak.loc['NB', 'Accuracy_mean'])}, "
  f"{pc(s_leak.loc['KNN', 'Accuracy_mean'])}) are close to the paper's values (84.39%, 84.39%, 85.85%).")
rows = [["Meta-learner", "Accuracy", "F1", "AUC"]] + \
       [[k, ms(meta.loc[k, "Accuracy_mean"], meta.loc[k, "Accuracy_std"]),
         ms(meta.loc[k, "F1_mean"], meta.loc[k, "F1_std"]), ms(meta.loc[k, "AUC_mean"], meta.loc[k, "AUC_std"])]
        for k in meta.index]
table(rows, "E3: stacking with different meta-learners (10 random splits, paper protocol).",
      col_widths=[4 * cm, 3.4 * cm, 3.4 * cm, 3.4 * cm])
P(f"E3 (Table 8) shows that the unreported meta-learner barely matters: all four choices give "
  f"{pc(meta['Accuracy_mean'].min())}–{pc(meta['Accuracy_mean'].max())} accuracy, so this assumption cannot "
  f"explain any gap to the paper.")

P("3.4 Critical analysis: why the reproduced (and published) numbers are inflated (E4, E5)", H2)
P(f"<b>Duplicate leakage (E4).</b> Because each patient appears 3–4 times, a random row-level split places "
  f"copies of almost every test patient in the training set. On average, {pc(leak_s['mean_frac_test_seen'], 1)} of "
  f"the 205 test rows (range {pc(leak_s['min_frac_test_seen'], 1)}–{pc(leak_s['max_frac_test_seen'], 1)}) have an "
  f"identical twin in training, leaving only about {leak_s['mean_unseen_per_split']:.0f} genuinely new patients per split. "
  f"The test set therefore largely measures memorisation, a textbook case of leakage [7], [8]. Fig. 4 and "
  f"Table 9 pool the 30 splits: on test rows seen during training, the high-capacity learners (DT, RF, XGB, "
  f"Stacking) are {pc(leak.loc[['DT', 'RF', 'XGB', 'Stacking'], 'acc_seen'].min(), 1)}–"
  f"{pc(leak.loc[['DT', 'RF', 'XGB', 'Stacking'], 'acc_seen'].max(), 1)} accurate, but on the "
  f"{int(leak.loc['LR', 'n_unseen'])} unseen rows they fall to "
  f"{pc(leak.loc[['DT', 'RF', 'XGB', 'Stacking'], 'acc_unseen'].min(), 1)}–"
  f"{pc(leak.loc[['DT', 'RF', 'XGB', 'Stacking'], 'acc_unseen'].max(), 1)}. The low-capacity LR and NB gain "
  f"nothing from memorisation: their accuracy on unseen rows is, if anything, higher, a difference within the sampling "
  f"noise of only {int(leak.loc['LR', 'n_unseen'])} pooled rows. This explains why the paper's trees and stacking appear to "
  f"dominate LR, NB and KNN.")
rows = [["Model", "Seen rows", "Acc (seen)", "Unseen rows", "Acc (unseen)", "Gap (pts)"]]
for m in MODEL_ORDER:
    rows.append([m, int(leak.loc[m, "n_seen"]), f3(leak.loc[m, "acc_seen"]), int(leak.loc[m, "n_unseen"]),
                 f3(leak.loc[m, "acc_unseen"]), f"{100 * (leak.loc[m, 'acc_seen'] - leak.loc[m, 'acc_unseen']):.1f}"])
table(rows, "E4: accuracy on test rows with vs without an exact duplicate in training (pooled over 30 splits).",
      col_widths=[2.2 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm])
figure(FIGS / "p1_leakage.png", "E4 duplicate-leakage audit. Memorising learners are perfect on seen rows only.", width=12)

best_u = s_dedup["Accuracy_mean"].idxmax()
P(f"<b>De-duplicated replication (E5).</b> Repeating the paper's protocol on the {audit['n_unique']} unique "
  f"patients (Table 10, Fig. 5) removes the leak. Stacking drops to {pc(s_dedup.loc['Stacking', 'Accuracy_mean'])} "
  f"± {pc(s_dedup.loc['Stacking', 'Accuracy_std'])} and DT to {pc(s_dedup.loc['DT', 'Accuracy_mean'])}, and the "
  f"ranking inverts: the best model is now {best_u} ({pc(s_dedup.loc[best_u, 'Accuracy_mean'])}). This is in line "
  f"with the 77–85% typically reported on the original Cleveland cohort [11]. The paper's central "
  f"claim, that stacking considerably enhances its base classifiers to reach 98.53%, is therefore not "
  f"supported once duplicates are handled. The single-split design and the absence of variance estimates "
  f"(SD of about {pc(s_dedup['Accuracy_std'].mean(), 1)} per model on unique data) also mean that differences of a "
  f"few points between models cannot be interpreted.")
rows = [["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"]]
for m in MODEL_ORDER:
    rows.append([m] + [ms(s_dedup.loc[m, f"{k}_mean"], s_dedup.loc[m, f"{k}_std"])
                       for k in ["Accuracy", "Precision", "Recall", "F1", "AUC"]])
table(rows, "E5: paper protocol on the 302 unique patients (30 random 80/20 splits, mean ± SD).",
      col_widths=[2 * cm] + [2.8 * cm] * 5, highlight_rows=(7,))
figure(FIGS / "p1_paper_vs_repro.png", "Accuracy reported in [1] vs reproduced with duplicates (E2) and "
       "without duplicates (E5). Error bars show ±1 SD over 30 splits.", width=15)
top3 = ", ".join(imp.index[:3])
P(f"<b>Feature importance.</b> RF impurity importance (Fig. 6) ranks {top3} highest. The paper's Fig. 2 ranks "
  "exang, then thal and cp. Both agree that angina, thalassaemia and chest-pain type carry most of the signal, "
  "but the paper does not state which model or importance measure it used, so exact agreement cannot be expected.")
figure(FIGS / "p1_importance.png", "Random-Forest feature importance (E1 training data).", width=9)

# ======================================================================
P("4. Part 2 — Proposed Solution: Leakage-Free, Diversity-Pruned Stacking (LF-DPS)", H1)
P("4.1 Motivation and limitation addressed", H2)
P("The key limitation of [1] is that its <b>experimental protocol cannot measure generalisation</b>. "
  "Row-level splitting of a file that is 70.5% duplicates, a single split, no variance estimate, "
  "and pre-processing that ignores sentinel missing codes together produce an estimate that is about "
  "15 points too optimistic (Section 3.4). A second, related weakness is that the ensemble is not designed. "
  "All six untuned learners are stacked regardless of whether they add information, so strongly correlated "
  "tree models are combined with weak, miscalibrated ones. This inflates variance on small data [15]. "
  "LF-DPS restructures the learning pipeline so that (a) every choice is made without access to test "
  "patients, and (b) the ensemble contains only learners that add complementary signal. It is expected "
  "to improve on [1] in two ways. First, its performance estimates should be unbiased, as nested CV "
  "removes selection bias [5], [6] and de-duplication removes leakage [7]. Second, pruning redundant "
  "members of a heterogeneous library typically matches or beats stacking everything [9], [15] with fewer "
  "models.")

P("4.2 Methodology and differences from the original approach", H2)
P("LF-DPS consists of five components (src/part2_proposed.py), summarised in Table 11. "
  "<b>C1 — patient-level de-duplication and repeated nested CV:</b> duplicates are collapsed to one row per "
  "patient, which is the correct unit of analysis, and all evaluation uses repeated stratified "
  "cross-validation over patients. "
  "<b>C2 — data-integrity repair:</b> ca = 4 and thal = 0 are mapped to missing and imputed inside each "
  "training fold (median for ca, mode for thal), so the model never learns a spurious 'unknown' category. "
  "<b>C3 — feature-type-aware encoding:</b> nominal variables (cp, restecg, slope, thal) are one-hot encoded "
  "rather than treated as ordered integers, which matters for the distance- and linear-based learners; "
  "continuous and ordinal variables are standardised; and every transformer is fitted in-fold. "
  "<b>C4 — nested hyper-parameter optimisation:</b> each candidate learner (the paper's six plus an RBF-SVM, "
  "which adds a smooth non-linear decision boundary) is tuned by randomized search [14] "
  f"({p2m['n_iter']} configurations, 5-fold inner CV, AUC objective) using only the outer-training data. "
  "<b>C5 — diversity-pruned stacking:</b> inner out-of-fold probabilities of the tuned learners are "
  "computed, and a greedy forward search (in the spirit of ensemble selection [9]) adds, one at a time, the "
  "learner that most increases the inner-CV AUC of a logistic meta-learner. The search stops when the gain "
  f"falls below {p2m['min_gain']} AUC, with at least two members. Because a learner that is highly "
  "correlated with those already chosen cannot raise the meta-learner's AUC, the rule selects for "
  "complementary rather than individually strong models. The selected learners are stacked on "
  "predicted probabilities with an L2-regularised logistic meta-learner [15].")
table([["Aspect", "Bhagat et al. [1]", "LF-DPS (proposed)"],
       ["Unit of analysis", "1,025 rows (duplicates kept)", "302 unique patients"],
       ["Validation", "Single random 80/20 split", "5×5 repeated stratified nested CV (25 outer folds)"],
       ["Missing values", "Regression imputation (no-op); sentinels kept", "ca=4, thal=0 → missing; in-fold imputation"],
       ["Encoding", "Label codes, all features scaled", "One-hot nominal; scaled continuous/ordinal"],
       ["Hyper-parameters", "Unreported (defaults)", "Nested randomized search, AUC objective"],
       ["Ensemble members", "All 6, fixed", "Greedy diversity-pruned subset of 7 tuned learners"],
       ["Meta-learner input", "Unreported", "Out-of-fold probabilities, LR meta-learner"],
       ["Statistics", "None", "Mean ± SD, corrected t-test [4], Wilcoxon, Brier, calibration, external check"]],
      "Methodological differences between [1] and the proposed LF-DPS.",
      col_widths=[3.2 * cm, 5.6 * cm, 7.4 * cm])

P("4.3 Experimental protocol", H2)
P(f"All methods are evaluated on the same 25 outer folds (5 folds × 5 repeats, stratified, seed 42) of the 302 "
  f"unique patients. Each outer fold has about 241 training and 61 test patients. To isolate the effect of "
  f"each component, an ablation ladder is evaluated: <b>A0</b> is the paper's stacking pipeline exactly as reproduced in "
  f"Part 1; <b>A1</b> adds C2 + C3; <b>A2</b> adds C4 (six tuned learners, all stacked); <b>A2b</b> adds the SVM to the "
  f"pool (all seven stacked, no pruning); and <b>A3 = LF-DPS</b> adds C5. As an additional baseline, "
  f"<b>Best single</b> is the single tuned learner with the highest inner-CV AUC, selected per fold. Metrics are "
  f"Accuracy, Precision, Recall, F1, AUC, Specificity, MCC [20] and the Brier score (probability calibration) [18]. "
  f"A3 is compared with every other method using the Nadeau–Bengio corrected resampled t-test [4], which "
  f"accounts for overlapping training sets in repeated CV, and the Wilcoxon signed-rank test. "
  f"Finally, E6 (src/part2_estimation_validity.py) locks away 20% of unique patients "
  f"(n = {valm['n_external_patients']}) as an external test set in each of {valm['n_rep']} repetitions. It then "
  f"compares each protocol's internal estimate with the true accuracy on these never-seen patients. XGB backend: {XGBB}.")

P("4.4 Results", H2)
rows = [["Method", "Accuracy", "F1", "AUC", "MCC", "Brier ↓"]]
for m in M:
    rows.append([m] + [ms(p2.loc[m, f"{k}_mean"], p2.loc[m, f"{k}_std"])
                       for k in ["Accuracy", "F1", "AUC", "MCC", "Brier"]])
table(rows, "Ablation study on 302 unique patients (25 outer folds, mean ± SD). A3 = proposed LF-DPS. "
      "Precision, Recall and Specificity are in Appendix B.",
      col_widths=[3.8 * cm] + [2.5 * cm] * 5, highlight_rows=(5,), font=7.2)

d_acc, p_acc, w_acc = stat(A0, "Accuracy")
d_auc, p_auc, w_auc = stat(A0, "AUC")
d_f1, p_f1, _ = stat(A0, "F1")
d_br, p_br, _ = stat(A0, "Brier")
d_bs, p_bs, _ = stat(BS, "Accuracy")
d_bsa, p_bsa, _ = stat(BS, "AUC")
d_2b, p_2b, _ = stat(A2b, "AUC")
P(f"Table 12 and Fig. 7 report the ablation. Compared with the reproduced baseline A0 under the same leakage-free "
  f"folds, LF-DPS changes accuracy by {100 * d_acc:+.2f} points (corrected p = {p_acc:.3f}; Wilcoxon p = {w_acc:.3f}), "
  f"F1 by {100 * d_f1:+.2f} points (p = {p_f1:.3f}), AUC by {d_auc:+.4f} (p = {p_auc:.3f}) and the Brier score by "
  f"{d_br:+.4f} (p = {p_br:.3f}). The accuracy difference is {sig(p_acc)} and the AUC difference is {sig(p_auc)}. "
  f"Against the best tuned single learner, the difference is {100 * d_bs:+.2f} accuracy points (p = {p_bs:.3f}) and "
  f"{d_bsa:+.4f} AUC (p = {p_bsa:.3f}). Against stacking all seven tuned learners (A2b), the AUC difference is "
  f"{d_2b:+.4f} (p = {p_2b:.3f}), while LF-DPS uses on average {p2m['mean_stack_size']:.1f} members "
  f"(range {p2m['min_stack_size']}–{p2m['max_stack_size']}) instead of seven. The full set of paired tests is in Table 13.")
figure(FIGS / "p2_boxplots.png", "Distribution of Accuracy, F1 and AUC over the 25 outer folds (green = LF-DPS).", width=16.5)

rows = [["Comparison", "Metric", "Mean diff", "t (corr.)", "p (corr.)", "p (Wilcoxon)"]]
for _, r in p2s[p2s.metric.isin(["Accuracy", "AUC", "Brier"])].iterrows():
    rows.append([r.comparison.replace("A3 vs ", "A3 vs\u00a0"), r.metric, f"{r.mean_diff:+.4f}",
                 f"{r.t_corrected:+.2f}", f"{r.p_corrected:.3f}", f"{r.p_wilcoxon:.3f}"])
table(rows, "Paired significance tests of LF-DPS against each alternative (25 outer folds). "
      "Negative Brier differences favour LF-DPS.",
      col_widths=[5.2 * cm, 1.9 * cm, 2 * cm, 1.9 * cm, 1.9 * cm, 2.1 * cm], font=7)

sel_top = freq.sort_values(ascending=False)
P(f"<b>What the pruning selects.</b> Fig. 9 shows how often each learner was chosen and the mean correlation "
  f"of inner out-of-fold probabilities. The most frequently selected learners are {sel_top.index[0]} "
  f"({pc(sel_top.iloc[0], 0)} of folds), {sel_top.index[1]} ({pc(sel_top.iloc[1], 0)}) and {sel_top.index[2]} "
  f"({pc(sel_top.iloc[2], 0)}). The tree ensembles are strongly inter-correlated "
  f"(RF–XGB r = {corr.loc['RF', 'XGB']:.2f}), and rarely enter together. The selection pattern therefore "
  f"confirms that C5 combines complementary families rather than redundant copies. Fig. 8 shows pooled ROC and "
  f"reliability curves: all three models are reasonably well calibrated (curves track the diagonal), and LF-DPS has the "
  f"highest pooled AUC, with a lower mean Brier score than A0 (Table 12).")
figure(FIGS / "p2_roc_calibration.png", "Pooled out-of-sample ROC curves and reliability diagram (25 folds).", width=15.5)
figure(FIGS / "p2_selection.png", "Left: fraction of outer folds in which each tuned learner is selected by C5. "
       "Right: mean correlation between inner out-of-fold probabilities.", width=15.5)

P("<b>Validity of the performance estimate (E6).</b> Table 14 and Fig. 10 answer the question the paper cannot: "
  "how close is each protocol's reported number to the accuracy obtained on patients never seen during "
  "development?")
rows = [["Protocol / model", "Estimated acc", "True external acc", "Optimism (pts)", "Est. AUC", "True AUC"]]
for k in val.index:
    rows.append([k, ms(val.loc[k, "est_Accuracy_mean"], val.loc[k, "est_Accuracy_std"]),
                 ms(val.loc[k, "true_Accuracy_mean"], val.loc[k, "true_Accuracy_std"]),
                 f"{100 * val.loc[k, 'optimism_Accuracy_mean']:+.1f} ± {100 * val.loc[k, 'optimism_Accuracy_std']:.1f}",
                 f3(val.loc[k, "est_AUC_mean"]), f3(val.loc[k, "true_AUC_mean"])])
table(rows, f"E6: internal estimate vs true accuracy on locked external patients ({valm['n_rep']} repetitions).",
      col_widths=[4.6 * cm, 2.6 * cm, 2.8 * cm, 2.3 * cm, 1.8 * cm, 1.8 * cm], highlight_rows=(3,))
v_a0 = val.loc["LF protocol (paper stacking)"]
P(f"The paper's protocol estimates {pc(v_paper.est_Accuracy_mean, 1)} accuracy, but the same model achieves "
  f"only {pc(v_paper.true_Accuracy_mean, 1)} on new patients, an optimism of "
  f"{100 * v_paper.optimism_Accuracy_mean:.1f} points. Under the leakage-free protocol, the estimates for the "
  f"paper's stacking ({100 * v_a0.optimism_Accuracy_mean:+.1f} pts) and for LF-DPS "
  f"({100 * v_lf.optimism_Accuracy_mean:+.1f} pts) are close to reality. LF-DPS's true external accuracy is "
  f"{pc(v_lf.true_Accuracy_mean, 1)}, compared with {pc(v_paper.true_Accuracy_mean, 1)} for the paper's model trained "
  f"with duplicates. This is the central empirical evidence for the proposed solution: it replaces an "
  f"unreliable number with one that can be trusted for decision-making.")
figure(FIGS / "p2_validity.png", "E6: estimated vs true external accuracy for each protocol.", width=12)

P("4.5 Comparative analysis against the paper and the reproduced baseline", H2)
rows = [["Source", "Evaluation", "Accuracy", "F1", "AUC"],
        ["Bhagat et al. [1]", "Single 80/20 split, duplicates", f4(PAPER["Stacking"]["Accuracy"]),
         f4(PAPER["Stacking"]["F1"]), f4(PAPER["Stacking"]["AUC"])],
        ["Reproduction (E1)", "Same, seed 42", f4(head.loc["Stacking", "Accuracy"]), f4(head.loc["Stacking", "F1"]),
         f4(head.loc["Stacking", "AUC"])],
        ["Reproduction (E2)", "30 splits, duplicates", f4(s_leak.loc["Stacking", "Accuracy_mean"]),
         f4(s_leak.loc["Stacking", "F1_mean"]), f4(s_leak.loc["Stacking", "AUC_mean"])],
        ["Reproduction (A0)", "25-fold repeated CV, unique patients", f4(p2.loc[A0, "Accuracy_mean"]),
         f4(p2.loc[A0, "F1_mean"]), f4(p2.loc[A0, "AUC_mean"])],
        ["LF-DPS (A3)", "25-fold repeated nested CV, unique patients", f4(p2.loc[A3, "Accuracy_mean"]),
         f4(p2.loc[A3, "F1_mean"]), f4(p2.loc[A3, "AUC_mean"])],
        ["LF-DPS (E6)", "Locked external patients", f4(v_lf.true_Accuracy_mean), f4(v_lf.true_F1_mean),
         f4(v_lf.true_AUC_mean)]]
table(rows, "Stacking results across evaluation settings.", col_widths=[3.4 * cm, 6 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm],
      highlight_rows=(5, 6))
P(f"Table 15 places all results side by side. The paper's 98.53% and my reproduction agree, and both are "
  f"artefacts of evaluating on duplicated patients. They should not be compared with the leakage-free rows. "
  f"Under a valid protocol, the paper's method achieves {pc(p2.loc[A0, 'Accuracy_mean'])} and LF-DPS "
  f"{pc(p2.loc[A3, 'Accuracy_mean'])}. The proposed contribution is therefore best characterised as follows. "
  f"(1) It gives a large, demonstrated improvement in the <i>validity</i> of the reported performance (E6). "
  f"(2) It adds principled handling of data-integrity errors and feature types. (3) It yields a smaller ensemble "
  f"whose predictive accuracy is on par with, or modestly different from, both the untuned full stack and the "
  f"best tuned single model (Table 13). On this small cohort, extra modelling complexity buys little "
  f"discrimination. The honest conclusion is that the signal ceiling of these 13 variables for about 300 "
  f"patients is roughly {pc(p2['Accuracy_mean'].max(), 0)}, not 98.5%.")

P("4.6 Limitations and future work", H2)
P("The evaluation relies on a single small cohort (302 patients, about 61 per test fold), so fold-to-fold "
  f"SD is about {pc(p2['Accuracy_std'].mean(), 1)} and small differences are hard to detect. Statistical power "
  "is limited, and the corrected t-test is deliberately conservative [4]. External validation used held-out "
  "patients from the same source rather than an independent hospital. Future work should include the "
  "Hungarian, Swiss and VA cohorts [10] as a genuine external test, cost-sensitive threshold selection "
  "(false negatives are costlier clinically), and reporting aligned with the TRIPOD+AI guideline [16]. "
  + ("" if USING_XGB else "Results in this build use a histogram gradient-boosting stand-in for XGBoost; "
     "re-running with xgboost installed regenerates every table and figure automatically."))

# ======================================================================
P("5. Conclusion", H1)
P(f"The published 98.53% accuracy of the stacking model in [1] was reproduced "
  f"({pc(head.loc['Stacking', 'Accuracy'])} with seed 42; {pc(s_leak.loc['Stacking', 'Accuracy_mean'])} over 30 splits). "
  f"It is, however, a consequence of {pc(audit['n_duplicates'] / audit['n_rows'], 1)} duplicate rows leaking across the "
  f"train/test split. On unique patients the same pipeline achieves about {pc(g_d, 0)}. The proposed LF-DPS "
  f"pipeline combines de-duplication, sentinel repair, type-aware encoding, nested tuning and diversity-pruned "
  f"stacking. It produces performance estimates that match accuracy on locked external patients (optimism "
  f"{100 * v_lf.optimism_Accuracy_mean:+.1f} vs {100 * v_paper.optimism_Accuracy_mean:+.1f} points) with a smaller "
  f"ensemble. The broader lesson is that validation design, not model choice, dominated the published result.")

P("6. Reproducibility", H1)
P("All results are generated by <font face='DV-B'>python run_all.py</font>, which runs Part 1, Part 2 and E6 "
  "and then rebuilds this PDF from the results files, so every number here is traceable to code output. "
  "Seeds are fixed throughout (split seeds 0–29 and 42; outer CV seed 42; E6 seeds 100–104). The README "
  "gives installation and run instructions, and the notebook provides a guided walk-through.")

P("7. Part 3 — Video Presentation", H1)
P(f"A video of at most five minutes (face, screen share and narration) is available at: "
  f"<b>{STUDENT['video_url']}</b>. It covers (i) the proposed LF-DPS solution and how it differs from [1], "
  "(ii) the reproduction and the duplicate-leakage finding, and (iii) a live run of the code showing model "
  "outputs and the comparative tables and figures in this report.")

P("Acknowledgement of Generative AI Use", H1)
P("Generative AI (Anthropic Claude) was used to assist with drafting code "
  "for the experimental pipeline, structuring the report and editing text. All experiments were executed, "
  "checked and interpreted by the author, who verified every reported number against the code output and "
  "takes full responsibility for the content.")

# ======================================================================
P("References", H1)
refs = [
    'M. Bhagat, A. Sharma, and P. Agarwal, "An efficient stacking-based ensemble technique for early heart attack prediction," <i>Multimedia Tools Appl.</i>, vol. 84, pp. 36351–36375, 2025, doi: 10.1007/s11042-024-19293-7.',
    'D. H. Wolpert, "Stacked generalization," <i>Neural Netw.</i>, vol. 5, no. 2, pp. 241–259, 1992.',
    'L. Breiman, "Stacked regressions," <i>Mach. Learn.</i>, vol. 24, no. 1, pp. 49–64, 1996.',
    'C. Nadeau and Y. Bengio, "Inference for the generalization error," <i>Mach. Learn.</i>, vol. 52, no. 3, pp. 239–281, 2003.',
    'G. C. Cawley and N. L. C. Talbot, "On over-fitting in model selection and subsequent selection bias in performance evaluation," <i>J. Mach. Learn. Res.</i>, vol. 11, pp. 2079–2107, 2010.',
    'S. Varma and R. Simon, "Bias in error estimation when using cross-validation for model selection," <i>BMC Bioinformatics</i>, vol. 7, no. 91, 2006.',
    'S. Kaufman, S. Rosset, C. Perlich, and O. Stitelman, "Leakage in data mining: Formulation, detection, and avoidance," <i>ACM Trans. Knowl. Discov. Data</i>, vol. 6, no. 4, pp. 1–21, 2012.',
    'S. Kapoor and A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science," <i>Patterns</i>, vol. 4, no. 9, Art. no. 100804, 2023.',
    'R. Caruana, A. Niculescu-Mizil, G. Crew, and A. Ksikes, "Ensemble selection from libraries of models," in <i>Proc. 21st Int. Conf. Mach. Learn. (ICML)</i>, 2004.',
    'A. Janosi, W. Steinbrunn, M. Pfisterer, and R. Detrano, "Heart Disease," UCI Machine Learning Repository, 1988, doi: 10.24432/C52P4X.',
    'R. Detrano <i>et al.</i>, "International application of a new probability algorithm for the diagnosis of coronary artery disease," <i>Am. J. Cardiol.</i>, vol. 64, no. 5, pp. 304–310, 1989.',
    'F. Pedregosa <i>et al.</i>, "Scikit-learn: Machine learning in Python," <i>J. Mach. Learn. Res.</i>, vol. 12, pp. 2825–2830, 2011.',
    'T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in <i>Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min.</i>, 2016, pp. 785–794.',
    'J. Bergstra and Y. Bengio, "Random search for hyper-parameter optimization," <i>J. Mach. Learn. Res.</i>, vol. 13, pp. 281–305, 2012.',
    'K. M. Ting and I. H. Witten, "Issues in stacked generalization," <i>J. Artif. Intell. Res.</i>, vol. 10, pp. 271–289, 1999.',
    'G. S. Collins <i>et al.</i>, "TRIPOD+AI statement: Updated guidance for reporting clinical prediction models that use regression or machine learning methods," <i>BMJ</i>, vol. 385, Art. no. e078378, 2024.',
    'R. Bharti <i>et al.</i>, "Prediction of heart disease using a combination of machine learning and deep learning," <i>Comput. Intell. Neurosci.</i>, vol. 2021, Art. no. 8387680, 2021.',
    'A. Niculescu-Mizil and R. Caruana, "Predicting good probabilities with supervised learning," in <i>Proc. 22nd Int. Conf. Mach. Learn. (ICML)</i>, 2005, pp. 625–632.',
    '"Heart Disease Dataset," Kaggle, 2019. [Online]. Available: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset',
    'D. Chicco and G. Jurman, "The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation," <i>BMC Genomics</i>, vol. 21, Art. no. 6, 2020.',
]
for i, r in enumerate(refs, 1):
    P(f"[{i}] {r}", REF)

# ======================================================================
story.append(PageBreak())
P("Appendix A — Most frequently selected hyper-parameters (C4)", H1)
rows = [["Learner", "Mean inner AUC", "Most frequent tuned configuration (mode over 25 folds)"]]
for mdl, g in params.groupby("model", sort=False):
    cfg = []
    for c in g.columns.drop(["fold", "model", "inner_auc"]):
        vals = g[c].dropna()
        if len(vals):
            v = vals.mode().iloc[0]
            try:
                v = f"{float(v):.3g}"
            except ValueError:
                pass
            cfg.append(f"{c}={v}")
    rows.append([mdl, f3(g.inner_auc.mean()), ", ".join(cfg)])
table(rows, "Tuned hyper-parameters (full per-fold values in results/part2_tuned_params.csv).",
      col_widths=[1.8 * cm, 2.4 * cm, 12 * cm], font=7)
P("Appendix B — Full metric summary of the ablation", H1)
rows = [["Method", "Precision", "Recall", "Specificity"]] + [[m] + [ms(p2.loc[m, f"{k}_mean"], p2.loc[m, f"{k}_std"])
        for k in ["Precision", "Recall", "Specificity"]] for m in M]
table(rows, "Additional metrics for the ablation (Table 12).", col_widths=[4 * cm, 3.4 * cm, 3.4 * cm, 3.4 * cm],
      highlight_rows=(5,))


def on_page(c, d):
    c.saveState()
    c.setFont("DV", 7.5)
    c.setFillColor(colors.HexColor("#666666"))
    c.drawString(1.8 * cm, 1.1 * cm, "SIT307 Task 11.1HD — Reproduction and improvement of stacking-based heart-disease prediction")
    c.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"Page {d.page}")
    c.restoreState()


OUT.parent.mkdir(exist_ok=True)
doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                        topMargin=1.6 * cm, bottomMargin=1.7 * cm,
                        title="SIT307 11.1HD Technical Report", author=STUDENT["name"])
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print("Report written to", OUT)
