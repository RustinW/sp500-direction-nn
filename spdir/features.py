"""Features and target.

Timing convention
-----------------
Row ``t`` is "after the close of trading day t".

* Features in row ``t`` use bars up to and including day ``t`` only. Every
  transform below is a backward-looking ``rolling``/``ewm``/``shift(k>=0)``;
  there is no centred window, no negative shift, and no statistic computed
  over the whole sample. ``tests/test_leakage.py`` checks this by recomputing
  features on truncated history.
* The target in row ``t`` is the direction of the close-to-close return from
  day ``t`` to day ``t+1``. It is the only place a negative shift appears.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

TARGET = "y_up_next"
NEXT_RET = "ret_next"


def _rsi(price: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI, scaled to [0, 1]. Recursive, so strictly backward-looking."""
    delta = price.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss
    # No down moves in the window -> RSI is 1; keep warm-up rows NaN.
    return (1.0 - 1.0 / (1.0 + rs)).where((loss > 0) | loss.isna(), 1.0)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix indexed by date. Rows in the warm-up period contain NaN."""
    px = raw["adj_close"]  # dividend-adjusted, so returns are total returns
    ret = px.pct_change()
    f = pd.DataFrame(index=raw.index)

    # Lagged daily returns: ret_lag0 is the return realised at today's close.
    for k in range(5):
        f[f"ret_lag{k}"] = ret.shift(k)

    # Momentum over longer windows.
    for n in (5, 10, 21, 63):
        f[f"mom_{n}"] = px.pct_change(n)

    # Realised volatility.
    for n in (5, 21, 63):
        f[f"vol_{n}"] = ret.rolling(n).std()

    f["rsi_14"] = _rsi(px, 14)

    # Moving-average spreads.
    sma = {n: px.rolling(n).mean() for n in (10, 50, 200)}
    for n in (10, 50, 200):
        f[f"ma_spread_{n}"] = px / sma[n] - 1.0
    f["ma_50_200"] = sma[50] / sma[200] - 1.0

    # VIX.
    log_vix = np.log(raw["vix"])
    f["vix_log"] = log_vix
    f["vix_chg_1"] = log_vix.diff(1)
    f["vix_chg_5"] = log_vix.diff(5)
    f["vix_vs_ma21"] = raw["vix"] / raw["vix"].rolling(21).mean() - 1.0

    # Volume relative to its own recent average, and the day's range.
    f["volume_ratio_21"] = np.log(raw["volume"] / raw["volume"].rolling(21).mean())
    f["range_hl"] = (raw["high"] - raw["low"]) / raw["close"]

    return f


def build_target(raw: pd.DataFrame) -> pd.DataFrame:
    """Next-day return and its sign. The last row has no target yet (NaN)."""
    ret_next = raw["adj_close"].pct_change().shift(-1)
    out = pd.DataFrame({NEXT_RET: ret_next})
    # Label is undefined (NaN) where the next return is unknown; a flat day counts as "not up".
    out[TARGET] = (ret_next > 0).astype(float).where(ret_next.notna())
    return out


def make_dataset(raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Features + target with warm-up rows and the final unlabeled row removed."""
    feats = build_features(raw)
    cols = list(feats.columns)
    ds = feats.join(build_target(raw)).dropna()
    ds[TARGET] = ds[TARGET].astype(int)
    # Date on which each row's label becomes known (the next trading day).
    next_date = pd.Series(raw.index, index=raw.index).shift(-1)
    ds["label_date"] = next_date.reindex(ds.index)
    return ds, cols
