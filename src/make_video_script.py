"""Build the 5-minute video script PDF (numbers read from results/, like the report)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from common import RESULTS, ROOT

OUT = ROOT / "report" / "SIT307_11.1HD_Video_Script.pdf"
_ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for n, f in [("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"),
             ("DV-BI", "DejaVuSans-BoldOblique.ttf")]:
    pdfmetrics.registerFont(TTFont(n, str(_ttf / f)))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

B = ParagraphStyle("b", fontName="DV", fontSize=9.6, leading=13.6, spaceAfter=4)
SAY = ParagraphStyle("say", parent=B, fontSize=10, leading=14.5)
SCR = ParagraphStyle("scr", parent=B, fontSize=8.8, leading=12, textColor=colors.HexColor("#1F3A5F"))
H1 = ParagraphStyle("h1", parent=B, fontName="DV-B", fontSize=15, leading=19, spaceAfter=6,
                    textColor=colors.HexColor("#1F3A5F"))
H2 = ParagraphStyle("h2", parent=B, fontName="DV-B", fontSize=11, leading=14, spaceBefore=8, spaceAfter=4,
                    textColor=colors.HexColor("#1F3A5F"))
Q = ParagraphStyle("q", parent=B, fontName="DV-B", spaceBefore=5, spaceAfter=1)


def js(n): return json.load(open(RESULTS / n))
def csv(n, **k): return pd.read_csv(RESULTS / n, **k)
def pc(x, d=1): return f"{100 * x:.{d}f}%"


au = js("data_audit.json"); ls = js("part1_leakage_summary.json"); pm = js("part2_meta.json")
hd = csv("part1_headline.csv", index_col=0); sl = csv("part1_seeds_summary.csv", index_col=0)
sd = csv("part1_dedup_summary.csv", index_col=0); lk = csv("part1_leakage.csv", index_col=0)
p2 = csv("part2_summary.csv", index_col=0); st = csv("part2_stats.csv"); va = csv("part2_validity_summary.csv", index_col=0)
fr = csv("part2_selection_freq.csv", index_col=0).iloc[:, 0]; co = csv("part2_oof_corr.csv", index_col=0)
M = list(p2.index); A0, A3 = M[0], M[4]
vp, vl = va.loc["Paper protocol (paper stacking)"], va.loc["LF protocol (LF-DPS)"]
p_acc = st[(st.comparison == f"A3 vs {A0}") & (st.metric == "Accuracy")].p_corrected.iloc[0]
trees = ["DT", "RF", "XGB", "Stacking"]
best_u = sd.Accuracy_mean.idxmax()
xgb_real = au["xgb_backend"].startswith("xgboost")

story = []
P = lambda t, s=B: story.append(Paragraph(t, s))


def segment(time, title, screen, say):
    t = Table([[Paragraph(f"<b>{time}</b>  {title}", ParagraphStyle("x", parent=B, fontName="DV-B",
                                                                     textColor=colors.white))],
               [Paragraph("<b>ON SCREEN:</b> " + screen, SCR)],
               [Paragraph("<b>SAY:</b> " + say, SAY)]], colWidths=[17.2 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3A5F")),
                           ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#E8EEF6")),
                           ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#1F3A5F")),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                           ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7)]))
    story.append(KeepTogether([t, Spacer(1, 9)]))


P("SIT307 Task 11.1HD — Video Presentation Script (≤ 5 minutes)", H1)
P("<b>How to use this:</b> do <i>not</i> read it word for word. Read it aloud twice, then record from the bullet "
  "of each segment in your own words; tutors can tell when someone is reading. Every number below was filled in "
  "automatically from <font face='DV-B'>results/</font>, so rerun <font face='DV-B'>python run_all.py</font> "
  "(with xgboost) before recording and the script updates to match your run. About 130–150 spoken words per "
  "minute; the SAY text totals roughly 700 words.")
P("<b>Before recording:</b> camera on with your face visible throughout (small webcam overlay is fine); "
  "open in tabs: the report PDF, VS Code with <font face='DV-B'>src/</font>, a terminal in the project folder, "
  "and the executed notebook. Increase editor/terminal font size so text is readable in the video. "
  "Brief order matters: present the <b>proposed solution first</b>, then the reproduction, then the code demo/results.")

segment("0:00 – 0:20", "Introduction", "Face camera, then the report title page.",
        "Hi, I'm [name], and this is my SIT307 HD mini-research. I picked the paper by Bhagat and colleagues from "
        "2025, which claims a stacking ensemble predicts heart disease with 98.5 percent accuracy. I reproduced the "
        "whole study, found out why that number is so high, and designed a method that fixes the problem. I'll start "
        "with my proposed solution, then show the reproduction, and finish with a live demo and results.")

segment("0:20 – 1:35", "Part 2 — Proposed solution (LF-DPS)",
        "Report Section 4.2 and Table 11 (paper vs proposed). Then switch to <font face='DV-B'>src/part2_proposed.py</font>: "
        "scroll the docstring (C1–C5), then point at <font face='DV-B'>typed_preprocessor()</font> in common.py, "
        "<font face='DV-B'>tune_pool()</font> and <font face='DV-B'>greedy_select()</font>.",
        f"The main limitation of the paper is its evaluation. The dataset has {au['n_rows']} rows but only "
        f"{au['n_unique']} unique patients — each patient is copied three or four times — and the paper splits rows "
        f"at random, so copies of the same patient end up in both training and test. My method, which I call LF-DPS, "
        f"Leakage-Free Diversity-Pruned Stacking, has five parts. One: I collapse duplicates so the unit is the "
        f"patient, and evaluate with five-by-five repeated cross-validation. Two: I found that ca equal to 4 and thal "
        f"equal to 0 aren't valid codes — they match exactly the missing values in the original UCI file — so I treat "
        f"them as missing and impute inside each fold. Three: categorical features like chest-pain type are one-hot "
        f"encoded instead of being treated as numbers, which matters for KNN and logistic regression. Four: every "
        f"model is tuned with randomized search in an <i>inner</i> loop, so tuning never sees the test patients. "
        f"And five, the actual ensemble change: instead of stacking all six models, a greedy search adds one model at "
        f"a time only if it raises the meta-learner's cross-validated AUC. A model that's highly correlated with ones "
        f"already chosen can't add AUC, so this naturally picks complementary models.")

segment("1:35 – 2:45", "Part 1 — Reproduction and critical analysis",
        "Terminal: show <font face='DV-B'>python src/part1_reproduce.py</font> (start it, or show the saved "
        "<font face='DV-B'>results/part1_log.txt</font>). Open report Table 6 (paper vs reproduced), "
        "then Table 3 (data audit), then Fig. 3 (leakage) and Fig. 4.",
        f"For the reproduction I used the same 13 features and six classifiers with 5-fold stacking. The paper doesn't "
        f"report its split, seed, hyper-parameters or meta-learner, so I inferred them: 98.53 percent is exactly 202 out "
        f"of 205, which means an 80/20 split. With seed 42, my stacking gets {pc(hd.loc['Stacking', 'Accuracy'], 2)} "
        f"with precision {hd.loc['Stacking', 'Precision']:.2f} — essentially the paper's result — and over 30 different "
        f"splits it averages {pc(sl.loc['Stacking', 'Accuracy_mean'])}. So it's reproducible. But when I audited the data, "
        f"on average {pc(ls['mean_frac_test_seen'])} of test rows had an identical copy in training. On those rows the "
        f"tree models and stacking are about {pc(lk.loc[trees, 'acc_seen'].min(), 0)} correct; on the genuinely unseen "
        f"ones they drop to between {pc(lk.loc[trees, 'acc_unseen'].min(), 0)} and {pc(lk.loc[trees, 'acc_unseen'].max(), 0)}. "
        f"They're memorising, not generalising. When I rerun the paper's exact pipeline on unique patients, stacking falls "
        f"to {pc(sd.loc['Stacking', 'Accuracy_mean'])}, and the best model becomes {best_u}. I also found errors in the "
        f"paper's results table — for example MCC is reported as 1.0 alongside three false negatives; recomputed, it's "
        f"about 0.97.")

segment("2:45 – 4:15", "Demo and results of the proposed method",
        "Notebook (already executed): show the Part 2 summary table cell, then the boxplot figure (Fig. 6), the stats "
        "table (Table 13), the selection-frequency/correlation figure (Fig. 8), and finally Fig. 9 (E6). "
        "Optionally run <font face='DV-B'>python run_all.py --report</font> to show the PDF regenerating from results.",
        f"Here are the results on unique patients, with all methods on the same 25 folds. The ablation goes from the "
        f"paper's stacking, A0, to my full method, A3. LF-DPS reaches {pc(p2.loc[A3, 'Accuracy_mean'])} accuracy versus "
        f"{pc(p2.loc[A0, 'Accuracy_mean'])} for the paper's method, and higher F1 and MCC, using on average "
        f"{pm['mean_stack_size']:.1f} models instead of six. But I want to be honest: with the Nadeau–Bengio corrected "
        f"t-test, which accounts for overlapping training sets in repeated CV, that accuracy gain has p equal to "
        f"{p_acc:.2f} — not significant. With only {au['n_unique']} patients, about 61 per test fold, the fold-to-fold "
        f"variation is bigger than the gain. The selection behaves as intended: the decision tree was "
        f"{'never' if fr.get('DT', 0) == 0 else 'rarely'} chosen, and random forest and boosting, which are correlated at "
        f"{co.loc['RF', 'XGB']:.2f}, rarely appear together. The strongest evidence is this last experiment. I locked away "
        f"20 percent of patients that no protocol ever sees. The paper's protocol estimated {pc(vp.est_Accuracy_mean)} but "
        f"got {pc(vp.true_Accuracy_mean)} on those new patients — over-estimating by "
        f"{100 * vp.optimism_Accuracy_mean:.0f} points. LF-DPS estimated {pc(vl.est_Accuracy_mean)} and achieved "
        f"{pc(vl.true_Accuracy_mean)}. So my method's numbers can actually be trusted, and it's also more accurate on "
        f"genuinely new patients.")

segment("4:15 – 4:50", "Conclusion",
        "Face camera (or report Section 5 / Table 15).",
        "To sum up: the paper's 98.5 percent is reproducible, but it's an artefact of duplicate patients leaking between "
        "training and test. On real unseen patients, these 13 variables support roughly 82 to 84 percent. My method "
        "gives an honest estimate, handles the data errors properly and uses a smaller ensemble. The main limitation is "
        "that it's one small cohort; next I'd validate on the Hungarian and Swiss cohorts as a true external test, and "
        "add cost-sensitive thresholds, since missing a sick patient is worse than a false alarm. The lesson for me was "
        "that validation design mattered far more than which model you pick.")

segment("4:50 – 5:00", "Close", "Report title page showing the code link.",
        "All code, data and the notebook are at the link in my report, and running run_all.py regenerates every number "
        "you've seen. Thanks for watching.")

if not xgb_real:
    P("<b>Note:</b> this script was generated with the XGBoost stand-in. Rerun with xgboost installed before recording.",
      ParagraphStyle("w", parent=B, textColor=colors.HexColor("#B00020")))

story.append(PageBreak())
P("Viva / Q&A crib sheet — understand these, don't memorise them", H1)
qa = [
    ("Why does a random row split leak here?",
     "Each patient appears 3–4 times. A random split puts copies of the same patient on both sides, so the test "
     "set mostly contains patients the model was trained on. It measures recall of memorised rows, not generalisation."),
    ("Why do trees benefit more than logistic regression?",
     "An unpruned tree or random forest can fit training rows perfectly (training accuracy ≈ 1.0), so an identical "
     "test row gets the right answer for free. Logistic regression has a single linear boundary and can't memorise "
     "individual points, so duplicates barely help it."),
    ("How did you infer the paper's split?",
     "0.9853 × 205 = 202 correct; 205 is 20% of 1,025. Precision 1.0 and recall 0.9727 imply TP=107, FN=3, FP=0, TN=95."),
    ("What is nested cross-validation and why use it?",
     "The outer loop estimates performance; the inner loop (only on outer-training data) chooses hyper-parameters. "
     "If you tune and report on the same folds, you pick the configuration that got lucky on those folds, which "
     "biases the estimate upward (Cawley & Talbot 2010)."),
    ("How does stacking work?",
     "Base models produce out-of-fold predicted probabilities on the training set (via internal 5-fold CV, so a "
     "model never predicts a row it was trained on). A meta-learner — logistic regression — is trained on those "
     "probabilities to combine them."),
    ("How does your greedy selection reward diversity?",
     "It starts with the model whose OOF probabilities give the best meta-learner AUC, then adds the model that "
     "increases AUC most, stopping when the gain is below 0.002. A model nearly identical to one already in the stack "
     "adds no new information, so it can't raise AUC and isn't chosen."),
    ("Why the corrected (Nadeau–Bengio) t-test?",
     "In repeated CV the training sets overlap heavily, so fold scores aren't independent; a normal paired t-test "
     "underestimates variance and gives too-small p-values. The correction inflates the variance term by "
     "(1/J + n_test/n_train)."),
    ("Your method isn't significantly better — so what did you improve?",
     f"The validity of the evaluation. E6 shows the paper's protocol over-estimates by about "
     f"{100 * vp.optimism_Accuracy_mean:.0f} points while mine is within "
     f"{'less than one point' if abs(100 * vl.optimism_Accuracy_mean) < 1 else f'about {abs(100 * vl.optimism_Accuracy_mean):.0f} points'} of true external accuracy. Plus correct data handling, a "
     "smaller ensemble, and higher accuracy on genuinely new patients. Small data limits how much any model can gain."),
    ("Why is ca = 4 a missing value?",
     "The paper's own table says ca is 0–3 and thal 1–3. In unique rows there are exactly 4 patients with ca=4 and 2 "
     "with thal=0 — the same as the 4 and 2 '?' values in the original UCI Cleveland file."),
    ("Why one-hot encode chest-pain type?",
     "cp is nominal: 0–3 are categories, not amounts. Treated as a number, KNN thinks type 3 is 'three times' type 1, "
     "and logistic regression forces a single monotonic effect."),
    ("What is the Brier score?",
     "Mean squared error between predicted probability and the 0/1 outcome; lower is better. It measures calibration "
     "and sharpness, which matters clinically when probabilities guide decisions."),
    ("What would you do next?",
     "External validation on other hospitals' cohorts, cost-sensitive thresholds to favour recall, and reporting "
     "under the TRIPOD+AI guideline."),
    ("Did you use AI tools?",
     "Answer honestly and consistently with your report's GenAI acknowledgement: say what you used it for and what "
     "you checked and decided yourself."),
]
for q, a in qa:
    story.append(KeepTogether([Paragraph(q, Q), Paragraph(a, B)]))

P("Recording checklist", H2)
for item in ["Face visible the whole time; audio clear (test 10 seconds first).",
             "Under 5:00 — anything after 5 minutes may not be marked.",
             "Proposed solution presented before the reproduction.",
             "Show real code running or real outputs, not only slides.",
             "Upload to YouTube as <b>Unlisted</b>; open the link in a private window to check it plays.",
             "Paste the link into the STUDENT block of make_report.py and rebuild the report."]:
    P("☐ " + item)

OUT.parent.mkdir(exist_ok=True)
SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.9 * cm, rightMargin=1.9 * cm, topMargin=1.6 * cm,
                  bottomMargin=1.6 * cm, title="SIT307 11.1HD Video Script").build(story)
print("Video script written to", OUT)
