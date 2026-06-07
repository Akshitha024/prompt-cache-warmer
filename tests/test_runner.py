"""Runner smoke."""

from __future__ import annotations

from pathlib import Path

from pcw.runner import run


def test_runner_smoke(tmp_path: Path) -> None:
    s = run(tmp_path / "out", n_prompts=200, capacity_tokens=5_000, seed=1)
    assert len(s["results"]) == 3
    assert (tmp_path / "out" / "summary.json").exists()
