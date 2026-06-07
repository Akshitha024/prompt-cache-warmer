"""Six chart families."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from pcw.types import BenchResult


def _save(fig: Figure, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def hit_rate_bars(rows: list[BenchResult], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    strats = [r.strategy.value for r in rows]
    rates = [r.cache_hit_token_rate for r in rows]
    bars = ax.bar(strats, rates, color=["#7a7a7a", "#3b6fa1", "#5b8d4a"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("token-level hit rate")
    ax.set_title("Cache hit rate by warming strategy")
    for b, r in zip(bars, rates, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.02, f"{r:.1%}", ha="center", fontsize=10)
    return _save(fig, out)


def bytes_saved_bar(rows: list[BenchResult], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    strats = [r.strategy.value for r in rows]
    saved_mb = [r.bytes_saved / (1024 * 1024) for r in rows]
    ax.bar(strats, saved_mb, color="#c25a4f")
    ax.set_ylabel("bytes saved (MB)")
    ax.set_title("KV-cache bytes avoided")
    for i, v in enumerate(saved_mb):
        ax.text(i, v, f"{v:.0f}MB", ha="center", va="bottom", fontsize=9)
    return _save(fig, out)


def reused_vs_total(rows: list[BenchResult], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    strats = [r.strategy.value for r in rows]
    total = [r.total_tokens_processed for r in rows]
    reused = [r.tokens_reused_from_cache for r in rows]
    x = np.arange(len(strats))
    w = 0.4
    ax.bar(x - w / 2, total, w, label="total tokens", color="#3b6fa1")
    ax.bar(x + w / 2, reused, w, label="reused (no recompute)", color="#5b8d4a")
    ax.set_xticks(x)
    ax.set_xticklabels(strats)
    ax.set_ylabel("tokens")
    ax.set_title("Total vs reused tokens")
    ax.legend()
    return _save(fig, out)


def unique_prefixes(rows: list[BenchResult], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    strats = [r.strategy.value for r in rows]
    n = [r.n_unique_prefixes for r in rows]
    ax.bar(strats, n, color="#3b6fa1")
    ax.set_ylabel("# unique trie branches")
    ax.set_title("Prefix-tree branch count")
    return _save(fig, out)


def depth_bar(rows: list[BenchResult], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    strats = [r.strategy.value for r in rows]
    d = [r.prefix_tree_depth for r in rows]
    ax.bar(strats, d, color="#5b8d4a")
    ax.set_ylabel("max trie depth (tokens)")
    ax.set_title("Prefix-tree max depth")
    return _save(fig, out)


def hit_rate_pie(rows: list[BenchResult], out: Path) -> Path:
    fig, axes = plt.subplots(1, len(rows), figsize=(5 * len(rows), 4))
    if len(rows) == 1:
        axes = [axes]
    for ax, r in zip(axes, rows, strict=True):
        ax.pie(
            [r.tokens_reused_from_cache, r.total_tokens_processed - r.tokens_reused_from_cache],
            labels=["reused", "recomputed"],
            autopct="%1.0f%%",
            colors=["#5b8d4a", "#c25a4f"],
        )
        ax.set_title(r.strategy.value)
    return _save(fig, out)
