"""Long/flat back-test on out-of-sample predictions.

Position for day t+1 is decided after the close of day t: long SPY when the
model says "up", otherwise flat (cash earning 0%). The position earns the
close-to-close return from t to t+1. Every change of position costs
``cost_bps`` of the traded notional, deducted from that day's return. The
strategy starts flat, so its first entry is charged; buy-and-hold pays one
entry cost on day one.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from .config import Config
from .features import NEXT_RET


def strategy_returns(position: np.ndarray, next_ret: np.ndarray, cost_bps: float) -> np.ndarray:
    prev = np.concatenate([[0.0], position[:-1]])
    turnover = np.abs(position - prev)
    return position * next_ret - turnover * cost_bps / 1e4


def performance(returns: np.ndarray, trading_days: int = 252) -> Dict:
    equity = np.cumprod(1.0 + returns)
    years = len(returns) / trading_days
    sd = returns.std(ddof=1)
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    return {
        "total_return": float(equity[-1] - 1.0),
        "cagr": float(equity[-1] ** (1.0 / years) - 1.0),
        "ann_vol": float(sd * np.sqrt(trading_days)),
        "sharpe": float(returns.mean() / sd * np.sqrt(trading_days)) if sd > 0 else float("nan"),
        "max_drawdown": float((equity / peak - 1.0).min()),
    }


def versus_buy_and_hold(position: np.ndarray, next_ret: np.ndarray, cfg: Config) -> Dict:
    """How solid is the strategy's edge over buy-and-hold?

    A long/flat strategy differs from buy-and-hold only on its flat days, so a
    handful of days can decide the comparison. This reports a paired one-sided
    t-test on the daily net return difference (approximate: daily returns are
    fat-tailed and volatility clusters) and what happens to CAGR if the single
    most helpful flat day is taken away (i.e. the strategy had been long).
    """
    strat = strategy_returns(position, next_ret, cfg.cost_bps)
    bh = strategy_returns(np.ones(len(position)), next_ret, cfg.cost_bps)
    diff = strat - bh
    t = ttest_1samp(diff, 0.0, alternative="greater")
    flat = position == 0
    out = {
        "flat_days": int(flat.sum()),
        "mean_next_return_on_flat_days": float(next_ret[flat].mean()) if flat.any() else float("nan"),
        "up_rate_on_flat_days": float((next_ret[flat] > 0).mean()) if flat.any() else float("nan"),
        "excess_t_stat": float(t.statistic),
        "excess_p_value": float(t.pvalue),
        "cagr_without_best_flat_day": float("nan"),
        "best_flat_day_return_avoided": float("nan"),
    }
    if flat.any():
        best = int(np.flatnonzero(flat)[np.argmin(next_ret[flat])])
        pos2 = position.copy()
        pos2[best] = 1.0
        out["best_flat_day_return_avoided"] = float(next_ret[best])
        out["cagr_without_best_flat_day"] = performance(
            strategy_returns(pos2, next_ret, cfg.cost_bps), cfg.trading_days
        )["cagr"]
    return out


def run_backtests(preds: pd.DataFrame, cfg: Config) -> Dict:
    """Returns {"table": metrics per strategy, "vs_bh": robustness table, "equity": equity curves}."""
    next_ret = preds[NEXT_RET].to_numpy()
    strategies = {
        "buy_and_hold": np.ones(len(preds)),
        "mlp": preds["pred_mlp"].to_numpy(dtype=float),
        "logreg": preds["pred_logreg"].to_numpy(dtype=float),
        "prev_dir": preds["pred_prev_dir"].to_numpy(dtype=float),
    }
    rows, equity, vs_rows = [], {}, []
    for name, pos in strategies.items():
        if name != "buy_and_hold":
            vs_rows.append({"strategy": name, **versus_buy_and_hold(pos, next_ret, cfg)})
        for label, bps in (("net", cfg.cost_bps), ("gross", 0.0)):
            r = strategy_returns(pos, next_ret, bps)
            row = {"strategy": name, "costs": label, "cost_bps": bps}
            row.update(performance(r, cfg.trading_days))
            prev = np.concatenate([[0.0], pos[:-1]])
            row["time_in_market"] = float(pos.mean())
            row["n_trades"] = int(np.abs(pos - prev).sum())
            rows.append(row)
            if label == "net":
                equity[name] = np.cumprod(1.0 + r)
    # Index equity by the date the return is realised (the day after the signal).
    eq = pd.DataFrame(equity, index=pd.DatetimeIndex(preds["label_date"], name="Date"))
    return {"table": pd.DataFrame(rows), "vs_bh": pd.DataFrame(vs_rows), "equity": eq}
