"""Expanding-window walk-forward evaluation.

Fold k tests on calendar year Y_k and trains on every row whose *label* was
already known before Y_k started. The training window only ever grows; rows
are never shuffled across the train/test boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .features import NEXT_RET, TARGET
from .models import fit_predict_logreg, fit_predict_mlp

MODELS = ["mlp", "logreg", "prev_dir", "always_up"]
MODEL_LABELS = {
    "mlp": "Neural net (MLP)",
    "logreg": "Logistic regression",
    "prev_dir": "Previous-day direction",
    "always_up": "Always up",
}


@dataclass(frozen=True)
class Fold:
    year: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def make_folds(ds: pd.DataFrame, cfg: Config) -> Iterator[Fold]:
    years = ds.index.year
    for year in range(cfg.first_test_year, cfg.last_test_year + 1):
        test_mask = years == year
        if not test_mask.any():
            continue
        test_start = ds.index[test_mask][0]
        # Purge: a training row is usable only if its label (the next day's
        # return) was realised strictly before the first test date. This drops
        # the last trading day of the prior year, whose label is the first
        # test day's return.
        train_mask = (ds["label_date"] < test_start).to_numpy()
        yield Fold(year, np.flatnonzero(train_mask), np.flatnonzero(test_mask))


def run_walkforward(
    ds: pd.DataFrame, feature_cols: List[str], cfg: Config, verbose: bool = True
) -> Tuple[pd.DataFrame, List[Dict]]:
    """Out-of-sample P(up) for every model on every test day."""
    X = ds[feature_cols].to_numpy(dtype=np.float64)
    y = ds[TARGET].to_numpy(dtype=np.int64)

    parts, fold_info = [], []
    for fold in make_folds(ds, cfg):
        X_tr, y_tr = X[fold.train_idx], y[fold.train_idx]
        X_te = X[fold.test_idx]
        test_dates = ds.index[fold.test_idx]

        p_mlp, info = fit_predict_mlp(X_tr, y_tr, X_te, cfg)
        p_lr = fit_predict_logreg(X_tr, y_tr, X_te, cfg)

        part = pd.DataFrame(
            {
                "fold": fold.year,
                "label_date": ds["label_date"].to_numpy()[fold.test_idx],
                "y": y[fold.test_idx],
                NEXT_RET: ds[NEXT_RET].to_numpy()[fold.test_idx],
                "p_mlp": p_mlp,
                "p_logreg": p_lr,
                # Previous-direction baseline: today's realised direction as the forecast for tomorrow.
                "p_prev_dir": (ds["ret_lag0"].to_numpy()[fold.test_idx] > 0).astype(float),
                "p_always_up": 1.0,
            },
            index=test_dates,
        )
        parts.append(part)

        info.update(
            year=fold.year,
            train_start=str(ds.index[fold.train_idx[0]].date()),
            train_end=str(ds.index[fold.train_idx[-1]].date()),
            test_start=str(test_dates[0].date()),
            test_end=str(test_dates[-1].date()),
            n_train=len(fold.train_idx),
            n_test=len(fold.test_idx),
        )
        fold_info.append(info)
        if verbose:
            print(
                f"fold {fold.year}: train {info['train_start']}..{info['train_end']} "
                f"(n={info['n_train']})  test n={info['n_test']}  "
                f"mlp best epochs={info['best_epochs']}",
                flush=True,
            )

    preds = pd.concat(parts)
    for m in MODELS:
        preds[f"pred_{m}"] = (preds[f"p_{m}"] > cfg.threshold).astype(int)
    return preds, fold_info
