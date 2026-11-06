"""Automated leakage checks. Run with: python -m pytest -q"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spdir.config import CFG
from spdir.data import load_raw
from spdir.features import NEXT_RET, TARGET, build_features, make_dataset
from spdir.walkforward import make_folds


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return load_raw(CFG)


@pytest.fixture(scope="module")
def dataset(raw):
    return make_dataset(raw)


def test_features_do_not_change_when_future_is_removed(raw):
    """Features at date t must be identical whether or not bars after t exist."""
    full = build_features(raw)
    rng = np.random.default_rng(0)
    for cut in sorted(rng.choice(np.arange(300, len(raw) - 1), size=25, replace=False)):
        truncated = build_features(raw.iloc[: cut + 1])
        pd.testing.assert_series_equal(full.iloc[cut], truncated.iloc[-1], check_names=False, rtol=1e-12, atol=1e-12)


def test_features_do_not_change_when_future_is_corrupted(raw):
    """Scrambling every bar after t must leave features at or before t untouched."""
    cut = len(raw) // 2
    bad = raw.copy()
    bad.iloc[cut + 1 :] = bad.iloc[cut + 1 :].to_numpy() * 3.7 + 11.0
    a, b = build_features(raw).iloc[: cut + 1], build_features(bad).iloc[: cut + 1]
    pd.testing.assert_frame_equal(a, b)


def test_target_is_next_day_direction(raw, dataset):
    ds, _ = dataset
    ret = raw["adj_close"].pct_change()
    pos = raw.index.get_indexer(ds.index)
    next_ret = ret.to_numpy()[pos + 1]
    np.testing.assert_allclose(ds[NEXT_RET].to_numpy(), next_ret)
    np.testing.assert_array_equal(ds[TARGET].to_numpy(), (next_ret > 0).astype(int))
    np.testing.assert_array_equal(ds["label_date"].to_numpy(), raw.index.to_numpy()[pos + 1])


def test_no_feature_equals_the_target_return(dataset):
    ds, cols = dataset
    for c in cols:
        assert abs(np.corrcoef(ds[c], ds[NEXT_RET])[0, 1]) < 0.2, f"{c} is suspiciously correlated with the next return"


def test_folds_are_expanding_ordered_and_purged(dataset):
    ds, _ = dataset
    folds = list(make_folds(ds, CFG))
    assert len(folds) == CFG.last_test_year - CFG.first_test_year + 1
    prev_train, seen_test = 0, set()
    for f in folds:
        test_start = ds.index[f.test_idx[0]]
        # Every training feature date and every training label date precede the test window.
        assert ds.index[f.train_idx].max() < test_start
        assert ds["label_date"].iloc[f.train_idx].max() < test_start
        # Contiguous in time, no overlap between folds, training set only grows.
        assert np.array_equal(f.train_idx, np.arange(len(f.train_idx)))
        assert not seen_test & set(f.test_idx)
        seen_test |= set(f.test_idx)
        assert len(f.train_idx) > prev_train
        prev_train = len(f.train_idx)
