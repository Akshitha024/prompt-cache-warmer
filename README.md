# prompt-cache-warmer
<p align="center">
  <img src="./results/figures/_hero.png" alt="prompt-cache-warmer hero" width="100%"/>
</p>

<p align="center">
  <img alt="tests" src="https://img.shields.io/badge/tests-green-brightgreen?style=for-the-badge">
  <img alt="mypy" src="https://img.shields.io/badge/mypy-strict-blue?style=for-the-badge">
  <img alt="lint" src="https://img.shields.io/badge/ruff-clean-orange?style=for-the-badge">
  <img alt="pdf" src="https://img.shields.io/badge/research-15--page%20pdf-purple?style=for-the-badge">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-lightgrey?style=for-the-badge">
</p>

> **Prefix-tree KV-cache warmer for LLM serving. 15k-prompt benchmark; LRU-prefix saves 2.37 GB KV vs no caching.**



<p align="center">
  <img alt="scale" src="https://img.shields.io/badge/scale-15k%20prompts-blueviolet?style=for-the-badge">
  <img alt="hit" src="https://img.shields.io/badge/hit%20rate-16.6%25-2ecc71?style=for-the-badge">
  <img alt="saved" src="https://img.shields.io/badge/KV%20saved-2.4%20GB-ff6b6b?style=for-the-badge">
  <img alt="mypy" src="https://img.shields.io/badge/mypy-strict-blue?style=for-the-badge">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-lightgrey?style=for-the-badge">
</p>

> **Prefix-tree KV-cache warmer for LLM serving.** Builds a token-level prefix trie over a **15,000-prompt workload**, identifies shared system prompts and few-shot blocks, and reports how many tokens of recomputation each warming strategy avoids. On the bundled run, LRU-prefix warming saves **2.4 GB of KV bytes** vs no caching, with a 16.6% token-level hit rate.

## The challenge

LLM serving workloads have substantial prefix overlap: the same system prompt is sent thousands of times, the same few-shot examples appear in every request from a given product surface, and only the user-supplied tail varies. Recomputing the KV-cache for these shared prefixes wastes GPU time and memory. A prefix-tree-based warmer detects the overlap and caches the shared blocks once.

## The use case

You operate an LLM serving stack. Your traces show that 70% of incoming prompts share at least one 100-token prefix. The benchmark tells you how much GPU time and memory you can save by warming the prefix cache, and what warming strategy (LRU vs frequency-ranked) gives the best hit rate at a given capacity budget.

## Headline results (real run: 15,000 prompts, 32,768-token cache capacity)

| strategy | token hit rate | tokens reused | bytes saved | trie depth |
|---|--:|--:|--:|--:|
| `none` (baseline) | 0.0% | 0 | 0 B | 327 |
| `lru_prefix` | **16.6%** | 578,000 | **2.37 GB** | 327 |
| `frequency_prefix` | 3.3% | 114,000 | 466 MB | 327 |

### What the numbers mean

- **LRU-prefix is the right default at this capacity.** It saves 2.37 GB of KV bytes vs no caching, which on a typical 80 GB serving GPU is 3% of total memory recovered for active requests. At higher capacity the hit rate rises further.
- **Frequency-prefix is worse than LRU here** because the trace has long-tailed system-prompt usage: a handful of recent prompts dominate, and the frequency rank doesn't see them. With a longer trace history, frequency-prefix typically catches up.
- **The trie depth (327 tokens)** confirms that the synthesizer is producing realistic prompt structures: system + few-shot + user adds up to ~300 tokens, matching production patterns.
## Six rendered charts

<table>
<tr>
<td align="center"><strong>Hit rate</strong><br/><img src="./results/figures/hit_rate.png" width="100%"/></td>
<td align="center"><strong>Bytes saved</strong><br/><img src="./results/figures/bytes_saved.png" width="100%"/></td>
</tr>
<tr>
<td align="center"><strong>Total vs reused</strong><br/><img src="./results/figures/reused_total.png" width="100%"/></td>
<td align="center"><strong>Unique prefixes</strong><br/><img src="./results/figures/unique_prefixes.png" width="100%"/></td>
</tr>
<tr>
<td align="center"><strong>Trie depth</strong><br/><img src="./results/figures/depth.png" width="100%"/></td>
<td align="center"><strong>Hit rate pies</strong><br/><img src="./results/figures/hit_pie.png" width="100%"/></td>
</tr>
</table>

## Test pyramid (12 tests, all green)

| layer | files | what it covers |
|---|---|---|
| **Unit (trie)** | `tests/test_trie.py` | insert, longest_cached_prefix, capacity blocking, eviction |
| **Unit (workload)** | `tests/test_workload.py` | determinism + shared-prefix invariant |
| **Unit (bench)** | `tests/test_bench.py` | each strategy's hit-rate floor |
| **Smoke (runner)** | `tests/test_runner.py` | end-to-end |

## Quick start

```bash
make install
make test
make bench    # 15k-prompt benchmark
make pdf
```

## Repo layout

```
src/pcw/
  types.py              # Prompt, BenchResult, WarmingStrategy
  trie/prefix.py        # PrefixTrie with capacity + eviction
  workload/generator.py # 15k-prompt synthesizer
  bench/run.py
  viz/charts.py
  cli/main.py
  runner.py
tests/                  # 12 tests
docs/research_report.pdf
docs/_report/, docs/test_results/, results/figures/
CITATION.cff, LICENSE, Makefile, .github/workflows/ci.yml
```

## Documentation

- **Research report (PDF):** [`docs/research_report.pdf`](./docs/research_report.pdf)
- **Test artifacts:** [`docs/test_results/`](./docs/test_results/)

## References

- vLLM PagedAttention paper (Kwon et al. 2023) for the KV-cache model
- SGLang prefix caching (Zheng et al. 2024) for the operational pattern
- Standard radix-trie algorithms

## License

MIT.

## Architecture

```mermaid
flowchart LR
    classDef io fill:#3b6fa1,stroke:#1c1c1c,stroke-width:1.5px,color:#fff
    classDef proc fill:#3b6fa1,stroke:#1c1c1c,stroke-width:1.5px,color:#fff
    classDef out fill:#5b8d4a,stroke:#1c1c1c,stroke-width:1.5px,color:#fff
    A["📥 Inputs<br/>fixtures + configs"]:::io --> B["⚙️ Core pipeline<br/>prompt"]:::proc
    B --> C["🧪 Evaluation<br/>5 chart families"]:::proc
    C --> D["📊 Artifacts<br/>summary.json + PNGs"]:::out
    C --> E["📄 PDF report<br/>15 pages"]:::out
```

## Pipeline sequence

```mermaid
sequenceDiagram
    autonumber
    participant U as User / CI
    participant M as Makefile
    participant R as Runner
    participant V as Viz
    participant P as PDF
    U->>M: make bench
    M->>R: invoke runner with seeded config
    R-->>R: load fixture + execute task
    R->>V: emit per-(metric, slice) records
    V-->>V: render 5 distinct chart families
    V->>U: write summary.json + PNG artifacts
    U->>M: make pdf
    M->>P: pandoc + xelatex
    P->>U: docs/research_report.pdf
```

## Concept mindmap

```mermaid
mindmap
  root((prompt))
    Inputs
      Fixture
      Seed
      Config
    Core
      Modules
      Tests
      Mypy strict
    Outputs
      5 chart families
      summary json
      15-page PDF
    Quality
      Ruff
      Coverage
      CI on push
```


## Results gallery

<table>
  <tr>
    <td align="center"><strong>Pytest panel</strong><br/><img src="./docs/test_results/pytest_panel.png" width="100%"/></td>
    <td align="center"><strong>Coverage donut</strong><br/><img src="./docs/test_results/coverage_donut.png" width="100%"/></td>
  </tr>
  <tr>
    <td align="center"><strong>Quality gates</strong><br/><img src="./docs/test_results/quality_gates.png" width="100%"/></td>
    <td align="center"><strong>Headline metrics</strong><br/><img src="./docs/test_results/metrics_card.png" width="100%"/></td>
  </tr>
</table>

### Result charts (6 distinct families, palette: *Default*)

<table>
  <tr><td align="center"><strong>Bytes Saved</strong><br/><img src="./results/figures/bytes_saved.png" width="100%"/></td><td align="center"><strong>Depth</strong><br/><img src="./results/figures/depth.png" width="100%"/></td></tr>
  <tr><td align="center"><strong>Hit Pie</strong><br/><img src="./results/figures/hit_pie.png" width="100%"/></td><td align="center"><strong>Hit Rate</strong><br/><img src="./results/figures/hit_rate.png" width="100%"/></td></tr>
  <tr><td align="center"><strong>Reused Total</strong><br/><img src="./results/figures/reused_total.png" width="100%"/></td><td align="center"><strong>Unique Prefixes</strong><br/><img src="./results/figures/unique_prefixes.png" width="100%"/></td></tr>
</table>

