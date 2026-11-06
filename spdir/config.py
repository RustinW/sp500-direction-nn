"""Every choice that affects the results lives here.

These values were fixed before the first out-of-sample run and have not been
changed in response to test results. If you change them after looking at the
test metrics, the reported numbers stop being out-of-sample.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"


@dataclass(frozen=True)
class Config:
    # --- data ---------------------------------------------------------------
    ticker: str = "SPY"
    vix_ticker: str = "^VIX"
    start: str = "2000-01-01"
    end: str = "2026-10-31"  # exclusive in yfinance; last bar is 2026-10-30

    # --- walk-forward -------------------------------------------------------
    first_test_year: int = 2010  # train on everything before, expanding by one year per fold
    last_test_year: int = 2026
    val_fraction: float = 0.15  # most recent slice of the *training* fold, for early stopping only

    # --- neural network (fixed a priori, not tuned) --------------------------
    hidden: Tuple[int, ...] = (32, 16)
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 20
    ensemble_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)  # probabilities are averaged

    # --- logistic regression baseline ---------------------------------------
    logreg_C: float = 1.0

    # --- decision rule and back-test ----------------------------------------
    threshold: float = 0.5  # predict "up" / be long when P(up) > threshold
    cost_bps: float = 5.0  # charged per unit of position change (one way)
    trading_days: int = 252

    seed: int = 42


CFG = Config()
