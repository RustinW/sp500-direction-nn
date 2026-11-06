"""Classification metrics and significance tests on out-of-sample predictions."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

from .walkforward import MODELS


def classification_metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> Dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    # AUC needs both classes in y and a score that varies; a constant score
    # (always-up) ranks nothing and is 0.5 by definition.
    if len(np.unique(y)) < 2:
        auc = float("nan")
    elif np.ptp(prob) == 0:
        auc = 0.5
    else:
        auc = float(roc_auc_score(y, prob))
    return {
        "n": int(len(y)),
        "up_rate": float(np.mean(y)),
        "accuracy": float(np.mean(y == pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "auc": auc,
        "pred_up_rate": float(np.mean(pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def per_fold_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, g in preds.groupby("fold"):
        for m in MODELS:
            row = {"fold": int(fold), "model": m}
            row.update(classification_metrics(g["y"].to_numpy(), g[f"pred_{m}"].to_numpy(), g[f"p_{m}"].to_numpy()))
            rows.append(row)
    return pd.DataFrame(rows)


def pooled_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in MODELS:
        row = {"model": m}
        row.update(classification_metrics(preds["y"].to_numpy(), preds[f"pred_{m}"].to_numpy(), preds[f"p_{m}"].to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def significance_tests(preds: pd.DataFrame, model: str) -> Dict:
    """Does ``model`` beat a coin flip, and does it beat always-up?

    All tests are one-sided ("model is better") and pooled over every
    out-of-sample day.

    * vs 50%: exact binomial test of the number of correct calls against p=0.5.
    * vs always-up, unpaired: exact binomial test against p0 = the realised
      share of up days in the test period (the accuracy always-up achieved).
      Treats p0 as a known constant, so it is slightly anti-conservative.
    * vs always-up, paired (exact McNemar): on days where the two disagree
      about being right, is the model right more than half the time? This is
      a binomial test on the discordant days and is the more appropriate one.
    """
    y = preds["y"].to_numpy()
    correct = preds[f"pred_{model}"].to_numpy() == y
    base_correct = preds["pred_always_up"].to_numpy() == y
    n, k = int(len(y)), int(correct.sum())
    base_acc = float(base_correct.mean())

    vs_half = binomtest(k, n, 0.5, alternative="greater")
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    vs_base = binomtest(k, n, base_acc, alternative="greater")

    model_only = int((correct & ~base_correct).sum())  # model right, always-up wrong
    base_only = int((~correct & base_correct).sum())  # always-up right, model wrong
    n_disc = model_only + base_only
    mcnemar_p = (
        float(binomtest(model_only, n_disc, 0.5, alternative="greater").pvalue) if n_disc else float("nan")
    )

    return {
        "model": model,
        "n": n,
        "n_correct": k,
        "accuracy": k / n,
        "accuracy_ci95": [float(ci.low), float(ci.high)],
        "p_value_vs_50pct": float(vs_half.pvalue),
        "always_up_accuracy": base_acc,
        "p_value_vs_always_up_binomial": float(vs_base.pvalue),
        "discordant_model_right": model_only,
        "discordant_always_up_right": base_only,
        "p_value_vs_always_up_mcnemar": mcnemar_p,
    }
