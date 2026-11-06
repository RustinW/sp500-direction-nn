"""The two figures: per-fold accuracy and equity curves."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator, NullLocator, PercentFormatter

from .config import Config
from .walkforward import MODEL_LABELS

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e7e6e2"
# Colour follows the model in both figures. The reference series (always-up /
# buy-and-hold: "do nothing clever") is neutral grey rather than a hue.
COLORS = {"mlp": "#2a78d6", "logreg": "#eb6834", "prev_dir": "#1baf7a", "always_up": "#8a8984", "buy_and_hold": "#8a8984"}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, length=0, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_fold_accuracy(fold_metrics: pd.DataFrame, path: Path) -> None:
    acc = fold_metrics.pivot(index="fold", columns="model", values="accuracy")
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=SURFACE)
    _style(ax)
    ax.axhline(0.5, color=TEXT_2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate("coin flip", xy=(acc.index[0] - 0.4, 0.5), xytext=(0, 4), textcoords="offset points", fontsize=8.5, color=TEXT_2)
    for m, marker in (("always_up", "s"), ("prev_dir", "^"), ("logreg", "D"), ("mlp", "o")):
        ax.plot(
            acc.index, acc[m], color=COLORS[m], linewidth=2, marker=marker, markersize=6.5,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label=MODEL_LABELS[m], zorder=3 if m == "mlp" else 2,
        )
    ax.set_xticks(list(acc.index))
    ax.set_xlim(acc.index[0] - 0.5, acc.index[-1] + 0.5)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Out-of-sample accuracy", color=TEXT_2, fontsize=9.5)
    ax.set_title("Next-day direction accuracy by walk-forward fold (test year)", loc="left", color=TEXT, fontsize=12.5, pad=30)
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles[::-1], labels[::-1], loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, frameon=False, fontsize=9.5, handlelength=2.2, columnspacing=1.6, borderaxespad=0.2)
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_equity(equity: pd.DataFrame, cfg: Config, path: Path) -> None:
    labels = {
        "buy_and_hold": "Buy and hold",
        "mlp": "MLP long/flat",
        "logreg": "Logistic long/flat",
        "prev_dir": "Previous-day long/flat",
    }
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
    _style(ax)
    for name in ("prev_dir", "logreg", "buy_and_hold", "mlp"):
        ax.plot(equity.index, equity[name], color=COLORS[name], linewidth=1.5, label=labels[name], zorder=3 if name == "mlp" else 2)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.set_yticks([0.5, 1, 2, 4, 8])
    ax.yaxis.set_minor_locator(NullLocator())
    lo, hi = float(equity.min().min()), float(equity.max().max())
    ax.set_ylim(min(0.9, lo * 0.9), hi * 1.15)
    ax.set_ylabel("Growth of 1 (log scale)", color=TEXT_2, fontsize=9.5)
    ax.set_title(
        f"Out-of-sample equity, net of {cfg.cost_bps:g} bps per trade",
        loc="left", color=TEXT, fontsize=12.5, pad=30,
    )
    handles, lbls = ax.get_legend_handles_labels()
    order = [lbls.index(labels[k]) for k in ("mlp", "logreg", "prev_dir", "buy_and_hold")]
    leg = ax.legend([handles[i] for i in order], [lbls[i] for i in order], loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, frameon=False, fontsize=9.5, handlelength=2.2, columnspacing=1.6, borderaxespad=0.2)
    for t in leg.get_texts():
        t.set_color(TEXT)
    # Direct labels with the final multiple at the right edge.
    # Labels are pushed apart (in log space) where two lines end close together.
    ends = equity.iloc[-1].sort_values()
    min_gap, y_prev = 1.13, 0.0
    for name, val in ends.items():
        y_lab = max(val, y_prev * min_gap)
        ax.annotate(f"{val:.2f}×", xy=(equity.index[-1], y_lab), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=TEXT)
        y_prev = y_lab
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.subplots_adjust(right=0.94)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
