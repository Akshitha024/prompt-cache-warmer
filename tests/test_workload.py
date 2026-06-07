"""Workload tests."""

from __future__ import annotations

import pytest

from pcw.workload.generator import generate


@pytest.mark.parametrize("seed", [11, 17, 23])
def test_deterministic(seed: int) -> None:
    a = generate(n=100, seed=seed)
    b = generate(n=100, seed=seed)
    assert [p.tokens for p in a] == [p.tokens for p in b]


def test_shared_prefixes_exist() -> None:
    ps = generate(n=200, n_systems=4, seed=5)
    # The first 10 tokens of every prompt should be one of 4 system prompts.
    firsts = {tuple(p.tokens[:10]) for p in ps}
    assert len(firsts) <= 4
