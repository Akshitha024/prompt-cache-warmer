"""End-to-end bench."""

from __future__ import annotations

from pcw.trie.prefix import PrefixTrie, TrieNode
from pcw.types import BenchResult, Prompt, WarmingStrategy

BYTES_PER_TOKEN = 4096


def run_one(
    prompts: list[Prompt], strategy: WarmingStrategy, capacity_tokens: int = 32_768
) -> BenchResult:
    trie = PrefixTrie(capacity_tokens=capacity_tokens)
    # Build the trie unconditionally so we can count unique prefixes.
    for p in prompts:
        trie.insert(p.tokens)

    if strategy == WarmingStrategy.FREQUENCY_PREFIX:
        # Mark the most-frequent prefixes as cached up to capacity.
        # We approximate: for each prompt, mark its full prefix path up to capacity.
        order = sorted(
            range(len(prompts)), key=lambda i: -_prefix_count(trie, prompts[i].tokens[:100])
        )
        for i in order:
            trie.mark_cached(prompts[i].tokens[:200])
            if trie.cached_token_count >= capacity_tokens:
                break

    total = 0
    reused = 0
    for p in prompts:
        total += len(p.tokens)
        if strategy == WarmingStrategy.LRU_PREFIX:
            reused += trie.longest_cached_prefix(p.tokens)
            trie.mark_cached(p.tokens[:200])
            if trie.cached_token_count > capacity_tokens:
                trie.evict_oldest(trie.cached_token_count - capacity_tokens)
        elif strategy == WarmingStrategy.FREQUENCY_PREFIX:
            reused += trie.longest_cached_prefix(p.tokens)
        # NONE: nothing reused
    unique_prefixes = _count_unique_prefixes(trie)
    return BenchResult(
        strategy=strategy,
        n_prompts=len(prompts),
        total_tokens_processed=total,
        tokens_reused_from_cache=reused,
        cache_hit_token_rate=reused / total if total else 0.0,
        n_unique_prefixes=unique_prefixes,
        bytes_saved=reused * BYTES_PER_TOKEN,
        prefix_tree_depth=trie.total_depth(),
    )


def _prefix_count(trie: PrefixTrie, tokens: list[int]) -> int:
    n = trie.root
    out = 0
    for t in tokens:
        if t not in n.children:
            break
        n = n.children[t]
        out += n.count
    return out


def _count_unique_prefixes(trie: PrefixTrie) -> int:
    counter = [0]

    def dfs(n: TrieNode) -> None:
        if len(n.children) > 1 or not n.children:
            counter[0] += 1
        for c in n.children.values():
            dfs(c)

    dfs(trie.root)
    return counter[0]
