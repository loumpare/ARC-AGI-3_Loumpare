"""Generic discrete state-graph with clustering, for mechanics ToolsAgent's
Cartesian self/BFS model can't represent (cd82's rotational selector,
ka59's mass-push blocks -- see llm_relay_agent_experiments memory,
2026-09-04 entry).

Instead of assuming a rigid avatar translating in x/y, this builds a graph
of (clustered state) --action--> (clustered state) edges purely from
observed transitions, then searches that graph for a path to any state
where a positive effect was observed. Works for ANY discrete mechanic, not
just movement -- the clustering step (state_signature) is what makes this
tractable: a raw per-pixel grid hash would treat any irrelevant visual noise
(a blinking decoration, an unrelated animated tile) as "a new state",
and would also make a Sokoban-scale configuration space
(ka59-style mass-push) explode combinatorially with no
generalization between near-identical configurations.
"""
from __future__ import annotations

from collections import Counter, deque

BIN_SIZE = 2  # round blob centroid/size coordinates to this granularity before
               # hashing -- absorbs rendering jitter/anti-aliasing noise without
               # conflating genuinely different configurations (which typically
               # differ by much more than a couple pixels for a discrete mechanic
               # like a rotated selector or a shifted block)


def state_signature(blobs: list[dict], bin_size: int = BIN_SIZE) -> tuple:
    """A hashable, order-independent signature for a set of blobs -- two
    frames that differ only in irrelevant noise hash identically; two frames
    with a genuinely different configuration (different rotation, different
    block arrangement) hash differently. Order-independent (sorted) so the
    same configuration signs identically regardless of _find_blobs's
    (arbitrary, scan-order-dependent) blob ordering."""
    items = []
    for b in blobs:
        r, c = b["centroid"]
        items.append((b["color"], round(r / bin_size), round(c / bin_size), round(b["size"] / bin_size)))
    return tuple(sorted(items))


class StateGraph:
    """Nodes are state signatures (see state_signature); edges are observed
    (signature, action) -> Counter[next_signature] transitions, reinforced by
    repetition so a one-off noisy observation doesn't corrupt the graph (the
    most-common observed outcome wins, matching the "require it to repeat
    before trusting it" pattern already used for action_deltas elsewhere in
    this project). Goal signatures are tracked separately -- any state
    reached at the moment a positive effect (levels_completed increasing, or
    a code-detected effect) was observed."""

    def __init__(self) -> None:
        self.edges: dict[tuple[tuple, str], Counter] = {}
        self.goal_signatures: set[tuple] = set()

    def record_transition(self, sig_before: tuple, action: str, sig_after: tuple) -> None:
        if sig_before == sig_after:
            return  # a no-op transition carries no connectivity information
        self.edges.setdefault((sig_before, action), Counter())[sig_after] += 1

    def mark_goal(self, sig: tuple) -> None:
        self.goal_signatures.add(sig)

    def neighbors(self, sig: tuple) -> list[tuple[str, tuple]]:
        """All (action, most-likely-resulting-signature) pairs observed from
        this exact state so far."""
        out = []
        for (s, action), counter in self.edges.items():
            if s == sig:
                out.append((action, counter.most_common(1)[0][0]))
        return out

    def find_path(self, start_sig: tuple, goal_sigs: set[tuple] | None = None,
                  max_depth: int = 60) -> list[str] | None:
        """BFS over OBSERVED transitions only, from start_sig to any signature
        in goal_sigs (defaults to self.goal_signatures). This is a search over
        experience already gathered, not a simulation -- it can only find a
        path the agent has, at some point, actually taken (same "learn by
        acting, then plan over what you learned" philosophy already used for
        action_deltas/blocked_values in llm_tools_agent.py)."""
        goals = goal_sigs if goal_sigs is not None else self.goal_signatures
        if not goals:
            return None
        if start_sig in goals:
            return []

        visited = {start_sig}
        queue = deque([(start_sig, [])])
        while queue:
            sig, path = queue.popleft()
            if len(path) >= max_depth:
                continue
            for action, next_sig in self.neighbors(sig):
                if next_sig in visited:
                    continue
                new_path = path + [action]
                if next_sig in goals:
                    return new_path
                visited.add(next_sig)
                queue.append((next_sig, new_path))
        return None
