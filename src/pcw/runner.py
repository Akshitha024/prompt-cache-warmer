"""End-to-end runner."""

from __future__ import annotations

import json
from pathlib import Path

from pcw.bench.run import run_one
from pcw.types import BenchResult, WarmingStrategy
from pcw.viz.charts import (
    bytes_saved_bar,
    depth_bar,
    hit_rate_bars,
    hit_rate_pie,
    reused_vs_total,
    unique_prefixes,
)
from pcw.workload.generator import generate


def run(
    out_dir: Path, n_prompts: int = 15_000, capacity_tokens: int = 32_768, seed: int = 17
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = Path("results/figures")
    prompts = generate(n=n_prompts, seed=seed)
    rows: list[BenchResult] = []
    for s in [WarmingStrategy.NONE, WarmingStrategy.LRU_PREFIX, WarmingStrategy.FREQUENCY_PREFIX]:
        rows.append(run_one(prompts, s, capacity_tokens=capacity_tokens))
    hit_rate_bars(rows, figs / "hit_rate.png")
    bytes_saved_bar(rows, figs / "bytes_saved.png")
    reused_vs_total(rows, figs / "reused_total.png")
    unique_prefixes(rows, figs / "unique_prefixes.png")
    depth_bar(rows, figs / "depth.png")
    hit_rate_pie(rows, figs / "hit_pie.png")
    summary: dict[str, object] = {
        "n_prompts": n_prompts,
        "capacity_tokens": capacity_tokens,
        "results": [r.model_dump() for r in rows],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
