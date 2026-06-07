"""Bench tests."""

from __future__ import annotations

from pcw.bench.run import run_one
from pcw.types import WarmingStrategy
from pcw.workload.generator import generate


def test_no_warming_zero_hit_rate() -> None:
    ps = generate(n=200, seed=3)
    r = run_one(ps, WarmingStrategy.NONE, capacity_tokens=10_000)
    assert r.cache_hit_token_rate == 0.0
    assert r.tokens_reused_from_cache == 0


def test_lru_warming_positive_hit_rate() -> None:
    ps = generate(n=500, seed=5)
    r = run_one(ps, WarmingStrategy.LRU_PREFIX, capacity_tokens=20_000)
    assert r.cache_hit_token_rate > 0.05


def test_frequency_beats_none() -> None:
    ps = generate(n=500, seed=7)
    none_r = run_one(ps, WarmingStrategy.NONE, capacity_tokens=20_000)
    freq_r = run_one(ps, WarmingStrategy.FREQUENCY_PREFIX, capacity_tokens=20_000)
    assert freq_r.cache_hit_token_rate > none_r.cache_hit_token_rate
