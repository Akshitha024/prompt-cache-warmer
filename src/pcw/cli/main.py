"""Typer CLI."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from pcw.runner import run

app = typer.Typer(no_args_is_help=True, help="Prompt prefix-cache warmer.")
console = Console()


@app.command()
def info() -> None:
    console.print("prompt-cache-warmer: see `pcw bench --help`.")


@app.command()
def bench(
    out_dir: Path = typer.Option(Path("runs/latest")),
    n: int = typer.Option(15_000),
    capacity: int = typer.Option(32_768),
    seed: int = typer.Option(17),
) -> None:
    res = run(out_dir, n_prompts=n, capacity_tokens=capacity, seed=seed)
    results_any = res["results"]
    assert isinstance(results_any, list)
    headline = [
        {
            "strategy": r["strategy"],
            "hit_rate": r["cache_hit_token_rate"],
            "bytes_saved": r["bytes_saved"],
        }
        for r in results_any
    ]
    console.print_json(json.dumps(headline, default=str))


if __name__ == "__main__":
    app()
