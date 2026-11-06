"""Download (once) and load daily SPY and VIX bars.

The raw download is cached as CSV in ``data/``. Later runs read the cache, so
results are reproducible even though Yahoo revises adjusted prices over time.
Delete the CSVs (or pass ``refresh=True``) to re-download.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import CFG, DATA_DIR, Config


def _cache_path(ticker: str, cfg: Config) -> Path:
    safe = ticker.replace("^", "")
    return DATA_DIR / f"{safe}_{cfg.start}_{cfg.end}.csv"


def _download(ticker: str, cfg: Config) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(
        ticker, start=cfg.start, end=cfg.end, auto_adjust=False, progress=False
    )
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns.name = None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]


def load_ticker(ticker: str, cfg: Config = CFG, refresh: bool = False) -> pd.DataFrame:
    path = _cache_path(ticker, cfg)
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col="Date", parse_dates=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = _download(ticker, cfg)
    df.to_csv(path)
    return df


def load_raw(cfg: Config = CFG, refresh: bool = False) -> pd.DataFrame:
    """One row per SPY trading day, with the same day's VIX close alongside."""
    spy = load_ticker(cfg.ticker, cfg, refresh)
    vix = load_ticker(cfg.vix_ticker, cfg, refresh)

    raw = pd.DataFrame(
        {
            "adj_close": spy["Adj Close"],
            "close": spy["Close"],
            "high": spy["High"],
            "low": spy["Low"],
            "volume": spy["Volume"].astype(float),
        }
    )
    # Align VIX to SPY dates. A missing VIX print is filled with the last
    # *earlier* value only (forward fill never looks ahead).
    raw["vix"] = vix["Close"].reindex(raw.index).ffill()
    raw = raw.sort_index()
    if not raw.index.is_unique:
        raise ValueError("Duplicate dates in raw data")
    return raw
