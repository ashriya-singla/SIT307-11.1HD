"""Part 2 - Proposed method: LF-DPS (Leakage-Free, Diversity-Pruned Stacking).

Limitation addressed: the paper's 98.53% comes from a random split of a file in
which 723/1025 rows are exact duplicates (only 302 unique patients), evaluated
once, with untuned defaults and sentinel-coded missing values treated as real
categories. LF-DPS changes the *learning pipeline and protocol*:

  C1  Patient-level de-duplication + repeated stratified nested CV (5x5 outer, 5 inner)
  C2  Data-integrity repair: ca=4 / thal=0 sentinels -> missing, imputed in-fold
  C3  Feature-type-aware encoding (one-hot nominal, scale continuous), fitted in-fold
  C4  Nested randomized hyper-parameter optimisation of every candidate base learner
  C5  Diversity-pruned stacking: greedy forward selection of base learners on inner
      out-of-fold probabilities, keeping only learners that add complementary signal,
      combined by a probability-level logistic meta-learner.

Ablation ladder evaluated on identical outer folds:
  A0 Paper stacking (reproduced baseline)      -> label-coded, defaults, LR meta
  A1 + C2/C3 typed preprocessing                -> defaults
  A2 + C4 tuned six paper learners, stack all
  A2b tuned six + SVM, stack all (no pruning)
  A3 LF-DPS = A2b pool + C5 pruning  (PROPOSED)
  BestSingle: single tuned learner with best inner-CV AUC
"""
from __future__ import annotations

import json
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import (RandomizedSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_predict, cross_val_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common import (FEATURES, FIGS, HAS_XGB, METRICS, RESULTS, SEED, TARGET,
                    XGB_BACKEND, evaluate, load_data, make_xgb, paper_base_learners,
                    paper_models, save_json, typed_preprocessor)

N_SPLITS, N_REPEATS = 5, 5
N_INNER = 5
N_ITER = 12            # randomized-search iterations per learner per outer fold
MIN_GAIN = 0.002       # min inner-CV AUC gain to add a learner (C5)
METHODS = ["A0 Paper stacking", "A1 +Typed prep", "A2 +Tuned (6)",
           "A2b +Tuned+SVM (7)", "A3 LF-DPS (proposed)", "Best single (tuned)"]


def search_space(seed):
    if HAS_XGB:
        xgb_space = {"n_estimators": randint(50, 300), "max_depth": randint(2, 6),
                     "learning_rate": loguniform(0.01, 0.3), "subsample": uniform(0.6, 0.4),
                     "colsample_bytree": uniform(0.6, 0.4), "min_child_weight": randint(1, 6),
                     "reg_lambda": loguniform(0.1, 10)}
    else:
        xgb_space = {"max_iter": randint(50, 300), "max_depth": [2, 3, 4, 5, None],
                     "learning_rate": loguniform(0.01, 0.3), "min_samples_leaf": randint(5, 40),
                     "l2_regularization": loguniform(1e-3, 10)}
    return {
        "LR": (LogisticRegression(max_iter=5000), {"C": loguniform(1e-3, 1e2)}),
        "NB": (GaussianNB(), {"var_smoothing": loguniform(1e-11, 1e-1)}),
        "KNN": (KNeighborsClassifier(), {"n_neighbors": randint(3, 40),
                                         "weights": ["uniform", "distance"], "p": [1, 2]}),
        "DT": (DecisionTreeClassifier(random_state=seed),
               {"max_depth": [2, 3, 4, 5, 6, 8, None], "min_samples_leaf": randint(1, 20),
                "criterion": ["gini", "entropy"]}),
        "RF": (RandomForestClassifier(random_state=seed, n_jobs=1),
               {"n_estimators": [100, 200, 300], "max_depth": [None, 3, 5, 8],
                "min_samples_leaf": randint(1, 10), "max_features": ["sqrt", 0.5]}),
        "XGB": (make_xgb(seed), xgb_space),
        "SVM": (SVC(probability=True, random_state=seed),
                {"C": loguniform(1e-2, 1e2), "gamma": loguniform(1e-4, 1)}),
    }


def typed(est):
    return Pipeline([("prep", typed_preprocessor()), ("clf", est)])


def stack(named_estimators, cv):
    return StackingClassifier(estimators=[(n, clone(e)) for n, e in named_estimators],
                              final_estimator=LogisticRegression(max_iter=1000),
                              cv=cv, stack_method="predict_proba", n_jobs=1)


def greedy_select(oof: pd.DataFrame, y, cv, min_gain=MIN_GAIN, min_size=2):
    """C5: forward selection of base learners maximising inner-CV AUC of the LR meta-learner."""
    def score(cols):
        return cross_val_score(LogisticRegression(max_iter=1000), oof[cols], y,
                               cv=cv, scoring="roc_auc").mean()
    remaining, selected, best, trace = list(oof.columns), [], -np.inf, []
    while remaining:
        cand = {c: score(selected + [c]) for c in remaining}
        c_best = max(cand, key=cand.get)
        gain = cand[c_best] - best
        if len(selected) >= min_size and gain < min_gain:
            break
        selected.append(c_best); remaining.remove(c_best)
        best = cand[c_best]; trace.append((c_best, best))
    return selected, trace


def tune_pool(Xtr, ytr, seed, inner, n_iter=None):
    """C2-C4: nested randomized search for every candidate learner (typed pipeline)."""
    tuned, inner_auc, params = {}, {}, {}
    for name, (est, space) in search_space(seed).items():
        rs = RandomizedSearchCV(typed(est), {f"clf__{k}": v for k, v in space.items()},
                                n_iter=n_iter or N_ITER, cv=inner, scoring="roc_auc",
                                random_state=seed, n_jobs=-1, refit=True)
        rs.fit(Xtr, ytr)
        tuned[name], inner_auc[name] = rs.best_estimator_, rs.best_score_
        params[name] = {k.replace("clf__", ""): str(v) for k, v in rs.best_params_.items()}
    return tuned, inner_auc, params


def build_lfdps(Xtr, ytr, seed, tuned=None):
    """Full LF-DPS on (de-duplicated) training data. Returns fitted model + diagnostics."""
    inner = StratifiedKFold(N_INNER, shuffle=True, random_state=seed)
    if tuned is None:
        tuned, _, _ = tune_pool(Xtr, ytr, seed, inner)
    oof = pd.DataFrame({n: cross_val_predict(m, Xtr, ytr, cv=inner, method="predict_proba")[:, 1]
                        for n, m in tuned.items()})
    sel, trace = greedy_select(oof, ytr, inner)
    model = stack([(n, tuned[n]) for n in sel], cv=inner).fit(Xtr, ytr)
    return model, sel, trace, oof


def corrected_ttest(a, b, test_train_ratio):
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated CV."""
    d = np.asarray(a) - np.asarray(b)
    J = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / J + test_train_ratio) * var)
    p = 2 * stats.t.sf(abs(t), df=J - 1)
    return float(t), float(p)


def main():
    t0 = time.time()
    df = load_data(dedup=True)
    X, y = df[FEATURES], df[TARGET]
    outer = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)

    rows, preds, selections, tuned_params, corr_mats = [], [], [], [], []
    cache = RESULTS / "cache"; cache.mkdir(exist_ok=True)
    for fold, (tr, te) in enumerate(outer.split(X, y)):
        cf = cache / f"part2_fold{fold:02d}.pkl"
        if cf.exists():                      # resume support: fold already computed
            with open(cf, "rb") as f:
                c = pickle.load(f)
            rows += c["rows"]; preds += c["preds"]; selections.append(c["sel"])
            tuned_params += c["params"]; corr_mats.append(c["corr"])
            continue
        n_rows, n_preds, n_params = len(rows), len(preds), len(tuned_params)
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
        fseed = SEED + fold
        inner = StratifiedKFold(N_INNER, shuffle=True, random_state=fseed)
        models = {}

        models[METHODS[0]] = paper_models(fseed)["Stacking"]
        models[METHODS[1]] = stack([(n, typed(e)) for n, e in paper_base_learners(fseed).items()], cv=5)

        # C4: nested hyper-parameter optimisation (inner CV only sees the outer-training fold)
        tuned, inner_auc, params = tune_pool(Xtr, ytr, fseed, inner)
        for name in tuned:
            tuned_params.append({"fold": fold, "model": name, "inner_auc": inner_auc[name], **params[name]})

        six = [n for n in tuned if n != "SVM"]
        models[METHODS[2]] = stack([(n, tuned[n]) for n in six], cv=inner)
        models[METHODS[3]] = stack(list(tuned.items()), cv=inner)

        # C5: diversity-pruned selection on inner out-of-fold probabilities (fitted inside build_lfdps)
        models[METHODS[4]], sel, trace, oof = build_lfdps(Xtr, ytr, fseed, tuned=tuned)
        corr_mats.append(oof.corr().values)
        selections.append({"fold": fold, "selected": sel, "trace": trace,
                           "inner_auc": inner_auc})

        best_name = max(inner_auc, key=inner_auc.get)
        models[METHODS[5]] = tuned[best_name]

        for mname, m in models.items():
            if mname not in (METHODS[4], METHODS[5]):   # A3 and best-single are already fitted
                m.fit(Xtr, ytr)
            sc = m.predict_proba(Xte)[:, 1]
            r = evaluate(yte, (sc >= 0.5).astype(int), sc)
            r.update(fold=fold, repeat=fold // N_SPLITS, method=mname,
                     best_single=best_name if mname == METHODS[5] else "")
            rows.append(r)
            preds += [{"fold": fold, "method": mname, "y": int(a), "score": float(b)}
                      for a, b in zip(yte, sc)]
        with open(cf, "wb") as f:
            pickle.dump({"rows": rows[n_rows:], "preds": preds[n_preds:], "sel": selections[-1],
                         "params": tuned_params[n_params:], "corr": corr_mats[-1]}, f)
        print(f"fold {fold + 1:2d}/{N_SPLITS * N_REPEATS}  selected={sel}  best_single={best_name}  "
              f"A0={rows[-6]['Accuracy']:.3f} A3={rows[-2]['Accuracy']:.3f}  ({time.time() - t0:.0f}s)",
              flush=True)

    res = pd.DataFrame(rows); res.to_csv(RESULTS / "part2_folds.csv", index=False)
    pd.DataFrame(preds).to_csv(RESULTS / "part2_predictions.csv", index=False)
    pd.DataFrame(tuned_params).to_csv(RESULTS / "part2_tuned_params.csv", index=False)
    with open(RESULTS / "part2_selections.json", "w") as f:
        json.dump(selections, f, indent=1, default=float)

    # ---------------------------------------------------------- summary + statistics
    g = res.groupby("method")[METRICS + ["Brier"]]
    summ = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std")).loc[METHODS]
    summ.to_csv(RESULTS / "part2_summary.csv")
    print("\n", g.mean().loc[METHODS].round(4))

    ratio = 1 / (N_SPLITS - 1)
    tests = []
    prop = res[res.method == METHODS[4]].sort_values("fold")
    for other in [m for m in METHODS if m != METHODS[4]]:
        o = res[res.method == other].sort_values("fold")
        for met in ["Accuracy", "F1", "AUC", "MCC", "Brier"]:
            t, p = corrected_ttest(prop[met].values, o[met].values, ratio)
            try:
                w_p = stats.wilcoxon(prop[met].values, o[met].values).pvalue
            except ValueError:
                w_p = 1.0
            tests.append({"comparison": f"A3 vs {other}", "metric": met,
                          "mean_diff": float((prop[met].values - o[met].values).mean()),
                          "t_corrected": t, "p_corrected": p, "p_wilcoxon": float(w_p)})
    pd.DataFrame(tests).to_csv(RESULTS / "part2_stats.csv", index=False)

    freq = pd.Series([n for s in selections for n in s["selected"]]).value_counts()
    freq = (freq / len(selections)).reindex(list(search_space(0)), fill_value=0)
    freq.to_csv(RESULTS / "part2_selection_freq.csv")
    sizes = [len(s["selected"]) for s in selections]
    bs = res[res.method == METHODS[5]].best_single.value_counts()
    save_json({"n_folds": len(selections), "mean_stack_size": float(np.mean(sizes)),
               "min_stack_size": int(min(sizes)), "max_stack_size": int(max(sizes)),
               "best_single_counts": bs.to_dict(), "xgb_backend": XGB_BACKEND,
               "n_iter": N_ITER, "min_gain": MIN_GAIN, "runtime_s": time.time() - t0},
              "part2_meta.json")
    mean_corr = pd.DataFrame(np.mean(corr_mats, axis=0), index=list(search_space(0)),
                             columns=list(search_space(0)))
    mean_corr.to_csv(RESULTS / "part2_oof_corr.csv")

    # ---------------------------------------------------------- figures
    short = ["A0", "A1", "A2", "A2b", "A3\n(LF-DPS)", "Best\nsingle"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, met in zip(axes, ["Accuracy", "F1", "AUC"]):
        data = [res[res.method == m][met].values for m in METHODS]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor("#55A868" if i == 4 else "#BBBBBB")
        ax.set_xticks(range(1, 7), short, fontsize=8); ax.set_title(f"{met} over 25 outer folds")
    fig.tight_layout(); fig.savefig(FIGS / "p2_boxplots.png", dpi=150); plt.close(fig)

    pr = pd.DataFrame(preds)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for m, col in [(METHODS[0], "#C44E52"), (METHODS[4], "#55A868"), (METHODS[5], "#4C72B0")]:
        d = pr[pr.method == m]
        fpr, tpr, _ = roc_curve(d.y, d.score)
        axes[0].plot(fpr, tpr, color=col, label=f"{m} (pooled AUC={roc_auc_score(d.y, d.score):.3f})")
        pt, pp = calibration_curve(d.y, d.score, n_bins=10, strategy="quantile")
        axes[1].plot(pp, pt, "o-", color=col, label=m)
    axes[0].plot([0, 1], [0, 1], "k--", lw=.8); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title("Pooled out-of-sample ROC (25 folds)"); axes[0].legend(fontsize=7, loc="lower right")
    axes[1].plot([0, 1], [0, 1], "k--", lw=.8); axes[1].set_xlabel("Mean predicted probability")
    axes[1].set_ylabel("Observed fraction positive"); axes[1].set_title("Reliability diagram")
    axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "p2_roc_calibration.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    freq.plot.bar(ax=axes[0], color="#55A868"); axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Fraction of outer folds selected"); axes[0].set_title("C5: base-learner selection frequency")
    im = axes[1].imshow(mean_corr.values, cmap="viridis", vmin=0.5, vmax=1)
    axes[1].set_xticks(range(7), mean_corr.columns); axes[1].set_yticks(range(7), mean_corr.index)
    for i in range(7):
        for j in range(7):
            axes[1].text(j, i, f"{mean_corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                         color="white" if mean_corr.values[i, j] < .8 else "black")
    axes[1].set_title("Mean correlation of inner OOF probabilities"); fig.colorbar(im, ax=axes[1])
    fig.tight_layout(); fig.savefig(FIGS / "p2_selection.png", dpi=150); plt.close(fig)
    print(f"\nPart 2 finished in {time.time() - t0:.0f}s. XGB backend: {XGB_BACKEND}")


if __name__ == "__main__":
    main()
