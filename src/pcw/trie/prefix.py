"""Token-prefix trie used as the dedup index for prompt prefixes."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrieNode:
    children: dict[int, TrieNode] = field(default_factory=dict)
    count: int = 0
    is_cached: bool = False


@dataclass
class PrefixTrie:
    root: TrieNode = field(default_factory=TrieNode)
    cached_token_count: int = 0
    capacity_tokens: int = 32_768

    def insert(self, tokens: list[int]) -> None:
        node = self.root
        for t in tokens:
            if t not in node.children:
                node.children[t] = TrieNode()
            node = node.children[t]
            node.count += 1

    def longest_cached_prefix(self, tokens: list[int]) -> int:
        """Return the length of the longest fully-cached prefix in `tokens`."""
        node = self.root
        out = 0
        for t in tokens:
            if t not in node.children:
                break
            node = node.children[t]
            if not node.is_cached:
                break
            out += 1
        return out

    def mark_cached(self, tokens: list[int]) -> None:
        """Mark the path tokens[0:k] as cached up to the capacity bound."""
        node = self.root
        for t in tokens:
            if t not in node.children:
                return
            node = node.children[t]
            if not node.is_cached:
                if self.cached_token_count >= self.capacity_tokens:
                    return
                node.is_cached = True
                self.cached_token_count += 1

    def evict_oldest(self, drop: int) -> None:
        """Mark `drop` cached nodes as un-cached (DFS, evicting leaves first)."""
        evicted = [0]

        def dfs(n: TrieNode) -> None:
            for c in n.children.values():
                if evicted[0] >= drop:
                    return
                dfs(c)
                if (
                    c.is_cached
                    and evicted[0] < drop
                    and not any(g.is_cached for g in c.children.values())
                ):
                    c.is_cached = False
                    evicted[0] += 1
                    self.cached_token_count -= 1

        dfs(self.root)

    def total_depth(self) -> int:
        def d(n: TrieNode) -> int:
            if not n.children:
                return 0
            return 1 + max(d(c) for c in n.children.values())

        return d(self.root)
