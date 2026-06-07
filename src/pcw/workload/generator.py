"""Synthesize a 15k-prompt trace with realistic prefix sharing.

In production, prompts come in clusters: a system prompt (5-15 shared tokens),
followed by a few-shot block (10-100 shared tokens), followed by a per-request
user prompt (50-500 unique tokens). The synthesizer matches this structure so
the prefix-tree dedup has something to find.
"""

from __future__ import annotations

import numpy as np

from pcw.types import Prompt


def generate(
    n: int = 15_000,
    n_systems: int = 8,
    n_few_shot: int = 30,
    seed: int = 17,
) -> list[Prompt]:
    rng = np.random.default_rng(seed)
    # Pre-generate the shared prefixes (system + few-shot).
    systems = [list(rng.integers(0, 50_000, size=10).tolist()) for _ in range(n_systems)]
    few_shots = [
        list(rng.integers(0, 50_000, size=rng.integers(20, 80)).tolist()) for _ in range(n_few_shot)
    ]
    out: list[Prompt] = []
    for i in range(n):
        sp = systems[int(rng.integers(0, n_systems))]
        fs = few_shots[int(rng.integers(0, n_few_shot))]
        user_len = int(rng.integers(50, 300))
        user = list(rng.integers(0, 50_000, size=user_len).tolist())
        out.append(Prompt(pid=i, tokens=sp + fs + user))
    return out
