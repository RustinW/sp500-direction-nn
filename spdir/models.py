"""Models. Each ``fit_predict_*`` function sees the training fold and the test
features only, and returns P(up) for the test rows. Scalers are fitted inside
these functions, on training rows only.
"""
from __future__ import annotations

import copy
from typing import Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn

from .config import Config


def set_determinism() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden: Tuple[int, ...], dropout: float):
        super().__init__()
        layers = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _train_one(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Config,
    seed: int,
) -> Tuple[MLP, int]:
    """Train one MLP with early stopping on the validation slice."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    model = MLP(X_fit.shape[1], cfg.hidden, cfg.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    Xf = torch.tensor(X_fit, dtype=torch.float32)
    yf = torch.tensor(y_fit, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.float32)

    best_loss, best_state, best_epoch, bad = float("inf"), None, 0, 0
    n = len(Xf)
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)  # shuffles *within the training fold* only
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            opt.zero_grad()
            loss_fn(model(Xf[idx]), yf[idx]).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, bad = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def fit_predict_mlp(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, cfg: Config
) -> Tuple[np.ndarray, dict]:
    """Seed-averaged MLP. The validation slice is the most recent part of the
    training fold; the test fold is never used for stopping or scaling."""
    n_val = max(1, int(round(len(X_train) * cfg.val_fraction)))
    X_fit, y_fit = X_train[:-n_val], y_train[:-n_val]
    X_val, y_val = X_train[-n_val:], y_train[-n_val:]

    scaler = StandardScaler().fit(X_fit)
    X_fit_s, X_val_s, X_test_s = (scaler.transform(a) for a in (X_fit, X_val, X_test))
    Xt = torch.tensor(X_test_s, dtype=torch.float32)

    probs, epochs = [], []
    for seed in cfg.ensemble_seeds:
        model, best_epoch = _train_one(X_fit_s, y_fit, X_val_s, y_val, cfg, cfg.seed + seed)
        with torch.no_grad():
            probs.append(torch.sigmoid(model(Xt)).numpy())
        epochs.append(best_epoch)
    return np.mean(probs, axis=0), {"best_epochs": epochs, "n_fit": len(X_fit), "n_val": n_val}


def fit_predict_logreg(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, cfg: Config
) -> np.ndarray:
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(C=cfg.logreg_C, max_iter=2000)
    # numpy 2.0 on Apple Silicon (Accelerate BLAS) emits spurious
    # divide/overflow/invalid warnings from matmul on perfectly finite inputs.
    # Silence them here and check the outputs explicitly instead.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        clf.fit(scaler.transform(X_train), y_train)
        proba = clf.predict_proba(scaler.transform(X_test))[:, 1]
    if not (np.isfinite(clf.coef_).all() and np.isfinite(proba).all()):
        raise FloatingPointError("Logistic regression produced non-finite values")
    return proba
