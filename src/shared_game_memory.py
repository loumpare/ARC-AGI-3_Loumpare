"""Process-wide, thread-safe scratchpad shared across ALL concurrent game
threads within a single Swarm run (see data/ARC-AGI-3-Agents/agents/swarm.py:
one Thread per game, all in the same Python process -- a plain in-memory
singleton is enough, no files/IPC needed). Lets one game's empirically
CONFIRMED facts about the shared ACTION_SPACE's generic role (movement vs
non-spatial) inform which action a DIFFERENT, still-in-progress game tries
first or how the brain reasons about an untested action.

This is a prior about the ARC-AGI-3 action-space CONVENTION (all games share
the same ACTION1-7/RESET vocabulary), not about any single game's puzzle
logic -- the same kind of intuition a human who has played several ARC-AGI-3
games develops ("ACTION1-4 are usually directional movement") without that
telling them how to WIN any specific new game. Every fact surfaced from here
is explicitly labeled as coming from OTHER games and still requires each
game's own existing bootstrap/calibration logic to independently confirm it
-- this only nudges trial order / brain context, it never substitutes for
real per-game verification, so per-game correctness cannot regress even when
a prior turns out wrong for that particular game (see feedback_no_game_hacking:
nothing here is keyed by game_id, only by the generic action name).
"""
from __future__ import annotations

import threading
from collections import Counter


class SharedGameMemory:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._role_votes: dict[str, Counter] = {}
        self._reported: set[tuple] = set()  # (report_key) dedupe -- one vote per
                                             # (game instance, action, role) so a
                                             # single long-running game can't
                                             # dominate the tally just by calling
                                             # report_role repeatedly

    def report_role(self, game_key: object, action_name: str, role: str) -> None:
        key = (game_key, action_name, role)
        with self._lock:
            if key in self._reported:
                return
            self._reported.add(key)
            self._role_votes.setdefault(action_name, Counter())[role] += 1

    def digest(self, legal_names: list[str], exclude: set[str]) -> str:
        """One line per action in `legal_names` (skipping anything in `exclude`
        -- typically actions this game has already confirmed itself) that other
        games this session have reported a role for."""
        with self._lock:
            lines = []
            for name in legal_names:
                if name in exclude:
                    continue
                counter = self._role_votes.get(name)
                if not counter:
                    continue
                total = sum(counter.values())
                top_role, top_n = counter.most_common(1)[0]
                lines.append(f"- {name}: {top_n}/{total} OTHER games this session found this "
                             f"action to be {top_role} (unverified here -- just a hint)")
            return "\n".join(lines)


_shared_memory = SharedGameMemory()


def get_shared_memory() -> SharedGameMemory:
    return _shared_memory
