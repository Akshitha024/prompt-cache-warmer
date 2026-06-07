"""Trie tests."""

from __future__ import annotations

from pcw.trie.prefix import PrefixTrie


def test_insert_and_longest_cached_prefix_zero_when_not_cached() -> None:
    t = PrefixTrie()
    t.insert([1, 2, 3])
    assert t.longest_cached_prefix([1, 2, 3]) == 0


def test_mark_cached_then_hit() -> None:
    t = PrefixTrie(capacity_tokens=10)
    t.insert([1, 2, 3])
    t.mark_cached([1, 2, 3])
    assert t.longest_cached_prefix([1, 2, 3]) == 3
    assert t.longest_cached_prefix([1, 2, 4]) == 2


def test_capacity_blocks_marking() -> None:
    t = PrefixTrie(capacity_tokens=2)
    t.insert([1, 2, 3, 4])
    t.mark_cached([1, 2, 3, 4])
    assert t.cached_token_count == 2


def test_evict_oldest_lowers_count() -> None:
    t = PrefixTrie(capacity_tokens=10)
    t.insert([1, 2, 3])
    t.mark_cached([1, 2, 3])
    before = t.cached_token_count
    t.evict_oldest(1)
    assert t.cached_token_count == before - 1
