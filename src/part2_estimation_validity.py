"""Part 2 / E6 - Does each protocol's performance ESTIMATE match reality?

For each repetition, 20% of the UNIQUE patients are locked away as an external
test set (never seen by any protocol). On the remaining development patients:

  Paper protocol : random 80/20 split of development ROWS (duplicates included)
                   -> estimated accuracy; final model trained on all development
                   rows -> true accuracy on external patients.
  A0 under LF protocol : paper stacking, de-duplicated 5-fold CV estimate; final fit on
                   unique development patients -> external.
  LF-DPS (proposed)    : nested LF-DPS 5-fold CV estimate on unique development
                   patients; final LF-DPS fit -> external.

Optimism = estimated - true. A valid protocol has optimism close to zero.
"""
from __future__ import annotations

import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from common import FEATURES, FIGS, RESULTS, TARGET, evaluate, load_data, paper_models, save_json
from part2_proposed import build_lfdps

N_REP = 5
PROTOCOLS = ["Paper protocol (paper stacking)", "LF protocol (paper stacking)", "LF protocol (LF-DPS)"]


def metrics_of(model, X, y):
    sc = model.predict_proba(X)[:, 1]
    return evaluate(y, (sc >= 0.5).astype(int), sc)


def main():
    t0 = time.time()
    full = load_data()
    uniq = load_data(dedup=True).reset_index().rename(columns={"index": "pid"})
    full = full.merge(uniq, on=FEATURES + [TARGET], how="left")      # patient id per row
    rows = []
    cache = RESULTS / "cache"; cache.mkdir(exist_ok=True)
    for r in range(N_REP):
        seed = 100 + r
        cf = cache / f"e6_rep{r}.pkl"
        if cf.exists():                      # resume support
            with open(cf, "rb") as f:
                c = pickle.load(f)
            rows += c["rows"]; ext = c["ext"]
            continue
        n_rows = len(rows)
        dev_ids, ext_ids = train_test_split(uniq.pid, test_size=0.2, stratify=uniq[TARGET],
                                            random_state=seed)
        dev_full = full[full.pid.isin(dev_ids)]
        dev_u = uniq[uniq.pid.isin(dev_ids)]
        ext = uniq[uniq.pid.isin(ext_ids)]
        Xe, ye = ext[FEATURES], ext[TARGET]

        # --- paper protocol
        Xtr, Xte, ytr, yte = train_test_split(dev_full[FEATURES], dev_full[TARGET],
                                              test_size=0.2, random_state=seed)
        est = metrics_of(paper_models(seed)["Stacking"].fit(Xtr, ytr), Xte, yte)
        true = metrics_of(paper_models(seed)["Stacking"].fit(dev_full[FEATURES], dev_full[TARGET]), Xe, ye)
        rows.append({"rep": r, "protocol": PROTOCOLS[0], **{f"est_{k}": est[k] for k in ["Accuracy", "F1", "AUC"]},
                     **{f"true_{k}": true[k] for k in ["Accuracy", "F1", "AUC"]}})

        # --- leakage-free protocol: 5-fold CV on unique development patients
        Xd, yd = dev_u[FEATURES], dev_u[TARGET]
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        fold_a0, fold_lf = [], []
        for k, (tr, te) in enumerate(cv.split(Xd, yd)):
            a0 = paper_models(seed)["Stacking"].fit(Xd.iloc[tr], yd.iloc[tr])
            fold_a0.append(metrics_of(a0, Xd.iloc[te], yd.iloc[te]))
            lf, _, _, _ = build_lfdps(Xd.iloc[tr], yd.iloc[tr], seed + k)
            fold_lf.append(metrics_of(lf, Xd.iloc[te], yd.iloc[te]))
        a0_true = metrics_of(paper_models(seed)["Stacking"].fit(Xd, yd), Xe, ye)
        lf_final, sel, _, _ = build_lfdps(Xd, yd, seed)
        lf_true = metrics_of(lf_final, Xe, ye)
        for name, folds, true in [(PROTOCOLS[1], fold_a0, a0_true), (PROTOCOLS[2], fold_lf, lf_true)]:
            rows.append({"rep": r, "protocol": name,
                         **{f"est_{k}": float(np.mean([f[k] for f in folds])) for k in ["Accuracy", "F1", "AUC"]},
                         **{f"true_{k}": true[k] for k in ["Accuracy", "F1", "AUC"]}})
        with open(cf, "wb") as f:
            pickle.dump({"rows": rows[n_rows:], "ext": ext}, f)
        print(f"rep {r + 1}/{N_REP}: ext n={len(ext)}  LF-DPS final={sel}  ({time.time() - t0:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    for k in ["Accuracy", "F1", "AUC"]:
        res[f"optimism_{k}"] = res[f"est_{k}"] - res[f"true_{k}"]
    res.to_csv(RESULTS / "part2_validity_raw.csv", index=False)
    summ = res.groupby("protocol").agg(["mean", "std"]).drop(columns="rep").loc[PROTOCOLS]
    summ.columns = ["_".join(c) for c in summ.columns]
    summ.to_csv(RESULTS / "part2_validity_summary.csv")
    save_json({"n_rep": N_REP, "n_external_patients": int(len(ext)),
               "runtime_s": time.time() - t0}, "part2_validity_meta.json")
    print(summ[["est_Accuracy_mean", "true_Accuracy_mean", "optimism_Accuracy_mean"]].round(4))

    fig, ax = plt.subplots(figsize=(8.5, 4))
    x = np.arange(3)
    ax.bar(x - 0.2, summ.est_Accuracy_mean, 0.4, yerr=summ.est_Accuracy_std, capsize=3,
           label="Estimated (internal validation)", color="#8172B2")
    ax.bar(x + 0.2, summ.true_Accuracy_mean, 0.4, yerr=summ.true_Accuracy_std, capsize=3,
           label="True (locked external patients)", color="#55A868")
    for i, o in enumerate(summ.optimism_Accuracy_mean):
        ax.text(i, 1.005, f"optimism {o * 100:+.1f} pts", ha="center", fontsize=8)
    ax.set_xticks(x, ["Paper protocol\n(paper stacking)", "LF protocol\n(paper stacking)",
                      "LF protocol\n(LF-DPS, proposed)"], fontsize=8)
    ax.set_ylim(0.6, 1.05); ax.set_ylabel("Accuracy"); ax.legend(fontsize=8, loc="lower right")
    ax.set_title(f"E6: Estimated vs true accuracy ({N_REP} repetitions)")
    fig.tight_layout(); fig.savefig(FIGS / "p2_validity.png", dpi=150); plt.close(fig)
    print(f"E6 finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
