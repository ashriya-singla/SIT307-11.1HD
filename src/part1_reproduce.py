"""Part 1 - Reproduction of Bhagat, Sharma & Agarwal (2025),
"An efficient stacking-based ensemble technique for early heart attack prediction".

Experiments
  E1  Headline reproduction: 80/20 random split (seed 42), 6 base learners + 5-fold stacking.
  E2  Split-seed robustness: E1 repeated over 30 random splits.
  E3  Meta-learner sensitivity (meta-classifier is not specified in the paper).
  E4  Duplicate-leakage audit: accuracy on test rows with / without an identical row in training.
  E5  Same protocol on the de-duplicated data (302 unique patients).
Outputs go to results/ (CSV/JSON) and results/figures/ (PNG).
"""
from __future__ import annotations

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from common import (FEATURES, FIGS, MODEL_ORDER, PAPER, RESULTS, SEED, TARGET,
                    XGB_BACKEND, data_audit, evaluate, load_data, make_xgb,
                    paper_models, paper_preprocessor, paper_stacking, save_json)

N_SEEDS = 30          # E2, E4, E5
N_SEEDS_META = 10     # E3
TEST_SIZE = 0.20      # 205 of 1025 -> paper's 98.53% = 202/205


def run_split(df, seed, models=None):
    """One 80/20 random (non-stratified) split; returns metrics + test-level predictions."""
    X, y = df[FEATURES], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed)
    models = models or paper_models(seed)
    rows, preds = [], {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        score = m.predict_proba(Xte)[:, 1]
        pred = (score >= 0.5).astype(int)
        r = evaluate(yte, pred, score)
        r.update(model=name, seed=seed,
                 train_acc=float((m.predict(Xtr) == ytr).mean()))
        rows.append(r)
        preds[name] = (pred, score)
    return rows, preds, (Xtr, Xte, ytr, yte), models


def seen_mask(Xtr, ytr, Xte, yte):
    """True where the test row (features + label) also occurs in the training set."""
    tr_keys = set(map(tuple, np.column_stack([Xtr.values, ytr.values])))
    return np.array([tuple(r) in tr_keys for r in np.column_stack([Xte.values, yte.values])])


def main():
    t0 = time.time()
    df = load_data()
    audit = data_audit(df)
    audit["xgb_backend"] = XGB_BACKEND
    save_json(audit, "data_audit.json")
    print("Data audit:", audit)

    # ------------------------------------------------ E1 headline split
    rows, preds, (Xtr, Xte, ytr, yte), models = run_split(df, SEED)
    head = pd.DataFrame(rows).set_index("model").loc[MODEL_ORDER]
    head.to_csv(RESULTS / "part1_headline.csv")
    print("\nE1 headline (seed 42):\n", head[["Accuracy", "Precision", "Recall", "F1", "AUC"]].round(4))

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2))
    for ax, name in zip(axes.ravel(), MODEL_ORDER):
        cm = confusion_matrix(yte, preds[name][0], labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=["No disease", "Disease"]).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name} (acc {head.loc[name, 'Accuracy']:.3f})")
    axes.ravel()[-1].axis("off")
    fig.suptitle("E1: Confusion matrices on the 20% test split (1,025-row data, seed 42)")
    fig.tight_layout(); fig.savefig(FIGS / "p1_confusion.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name in MODEL_ORDER:
        RocCurveDisplay.from_predictions(yte, preds[name][1], ax=ax,
                                         name=f"{name} (AUC={head.loc[name, 'AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title("E1: ROC curves (seed 42 split)")
    fig.tight_layout(); fig.savefig(FIGS / "p1_roc.png", dpi=150); plt.close(fig)

    rf = models["RF"].named_steps["clf"]
    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
    imp.to_csv(RESULTS / "part1_rf_importance.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    imp.plot.barh(ax=ax, color="#4C72B0")
    ax.set_xlabel("Mean decrease in impurity"); ax.set_title("E1: Random Forest feature importance")
    fig.tight_layout(); fig.savefig(FIGS / "p1_importance.png", dpi=150); plt.close(fig)

    # ------------------------------------------------ E2 + E4 + E5 over 30 seeds
    all_rows, dedup_rows, leak_rows = [], [], []
    df_u = load_data(dedup=True)
    for s in range(N_SEEDS):
        r, p, (Xtr, Xte, ytr, yte), _ = run_split(df, s)
        all_rows += r
        mask = seen_mask(Xtr, ytr, Xte, yte)
        for name in MODEL_ORDER:
            correct = p[name][0] == yte.values
            leak_rows.append({"seed": s, "model": name, "n_test": len(yte),
                              "n_seen": int(mask.sum()), "n_unseen": int((~mask).sum()),
                              "correct_seen": int(correct[mask].sum()),
                              "correct_unseen": int(correct[~mask].sum())})
        r_u, _, _, _ = run_split(df_u, s)
        dedup_rows += r_u
        print(f"  seed {s:2d} done ({time.time() - t0:.0f}s)")

    seeds = pd.DataFrame(all_rows); seeds.to_csv(RESULTS / "part1_seeds.csv", index=False)
    dedup = pd.DataFrame(dedup_rows); dedup.to_csv(RESULTS / "part1_dedup_seeds.csv", index=False)
    leak = pd.DataFrame(leak_rows); leak.to_csv(RESULTS / "part1_leakage_raw.csv", index=False)

    lk = leak.groupby("model")[["n_seen", "n_unseen", "correct_seen", "correct_unseen"]].sum()
    lk["acc_seen"] = lk.correct_seen / lk.n_seen
    lk["acc_unseen"] = lk.correct_unseen / lk.n_unseen
    lk = lk.loc[MODEL_ORDER]; lk.to_csv(RESULTS / "part1_leakage.csv")
    frac_seen = leak[leak.model == "LR"].eval("n_seen / n_test")
    save_json({"mean_frac_test_seen": frac_seen.mean(), "min_frac_test_seen": frac_seen.min(),
               "max_frac_test_seen": frac_seen.max(),
               "mean_unseen_per_split": leak[leak.model == "LR"].n_unseen.mean()},
              "part1_leakage_summary.json")
    print("\nE4 leakage:\n", lk.round(3))

    def summarise(d, tag):
        g = d.groupby("model")[["Accuracy", "Precision", "Recall", "F1", "AUC",
                                "Specificity", "MCC", "train_acc"]]
        out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std")).loc[MODEL_ORDER]
        out.to_csv(RESULTS / f"part1_{tag}_summary.csv")
        return out
    s_leaky, s_dedup = summarise(seeds, "seeds"), summarise(dedup, "dedup")
    print("\nE2 leaky 30-seed mean acc:\n", s_leaky["Accuracy_mean"].round(4))
    print("\nE5 dedup 30-seed mean acc:\n", s_dedup["Accuracy_mean"].round(4))

    # ------------------------------------------------ E3 meta-learner sensitivity
    metas = {"LogReg (default)": lambda s: LogisticRegression(max_iter=1000),
             "Random Forest": lambda s: RandomForestClassifier(random_state=s, n_jobs=1),
             "XGBoost": lambda s: make_xgb(s),
             "KNN": lambda s: KNeighborsClassifier()}
    meta_rows = []
    X, y = df[FEATURES], df[TARGET]
    for s in range(N_SEEDS_META):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=s)
        for mname, mk in metas.items():
            m = Pipeline([("prep", paper_preprocessor()),
                          ("clf", paper_stacking(s, meta=mk(s)))]).fit(Xtr, ytr)
            sc = m.predict_proba(Xte)[:, 1]
            r = evaluate(yte, (sc >= .5).astype(int), sc); r.update(meta=mname, seed=s)
            meta_rows.append(r)
    meta = pd.DataFrame(meta_rows)
    meta.to_csv(RESULTS / "part1_meta_raw.csv", index=False)
    mg = meta.groupby("meta")[["Accuracy", "F1", "AUC"]]
    mg.mean().add_suffix("_mean").join(mg.std().add_suffix("_std")).loc[list(metas)].to_csv(
        RESULTS / "part1_meta_summary.csv")
    print("\nE3 meta:\n", mg.mean().round(4))

    # ------------------------------------------------ figures: paper vs reproduced
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(MODEL_ORDER)); w = 0.27
    ax.bar(x - w, [PAPER[m]["Accuracy"] for m in MODEL_ORDER], w, label="Paper (reported)", color="#999999")
    ax.bar(x, s_leaky["Accuracy_mean"], w, yerr=s_leaky["Accuracy_std"], capsize=3,
           label="Reproduced, 1,025 rows (30 splits)", color="#4C72B0")
    ax.bar(x + w, s_dedup["Accuracy_mean"], w, yerr=s_dedup["Accuracy_std"], capsize=3,
           label="Reproduced, 302 unique rows (30 splits)", color="#DD8452")
    ax.set_xticks(x, MODEL_ORDER); ax.set_ylim(0.6, 1.02); ax.set_ylabel("Test accuracy")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8, frameon=False)
    ax.set_title("Paper vs reproduced accuracy (mean ± SD)")
    fig.tight_layout(); fig.savefig(FIGS / "p1_paper_vs_repro.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(x - 0.2, lk["acc_seen"], 0.4, label="Test rows WITH a duplicate in training", color="#C44E52")
    ax.bar(x + 0.2, lk["acc_unseen"], 0.4, label="Test rows with NO duplicate in training", color="#55A868")
    ax.set_xticks(x, MODEL_ORDER); ax.set_ylim(0.5, 1.02); ax.set_ylabel("Accuracy (pooled, 30 splits)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    ax.set_title("E4: Duplicate-leakage audit")
    fig.tight_layout(); fig.savefig(FIGS / "p1_leakage.png", dpi=150); plt.close(fig)

    print(f"\nPart 1 finished in {time.time() - t0:.0f}s. XGB backend: {XGB_BACKEND}")


if __name__ == "__main__":
    main()
