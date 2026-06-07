"""Types."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class WarmingStrategy(StrEnum):
    NONE = "none"
    LRU_PREFIX = "lru_prefix"
    FREQUENCY_PREFIX = "frequency_prefix"


class Prompt(BaseModel):
    pid: int
    tokens: list[int]  # token id sequence (synthetic)


class BenchResult(BaseModel):
    strategy: WarmingStrategy
    n_prompts: int
    total_tokens_processed: int
    tokens_reused_from_cache: int
    cache_hit_token_rate: float
    n_unique_prefixes: int
    bytes_saved: int
    prefix_tree_depth: int
