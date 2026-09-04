"""Comparison agent (2026-09-03), "tools+vision" variant: extends ToolsAgent
(src/llm_tools_agent.py) with two additions the user asked for, both aimed at
giving the brain real spatial grounding instead of a flat list of unlabeled
blobs, without reintroducing the failure mode that killed the earlier pure-
vision relay agents (Gemma mis-identifying "self" -- see
llm_relay_agent_experiments memory):

1. **Movement map**: the brain now sees exactly which actions have a
   confirmed spatial effect and by how much (`self.action_deltas`, already
   computed by ToolsAgent's code-only bootstrap -- just never surfaced in the
   prompt before). If NO action shows confirmed movement, that's itself a
   signal the mechanic isn't "walk around a maze" at all (e.g. a discrete
   selector or a button), which the brain can act on directly (see #3).

2. **Vision scene notes**: Gemma looks at the rendered grid ONCE per level
   and adds the qualitative interpretation code can't produce (what a shape
   looks like, whether several objects look arranged as a dial/ring/row of
   buttons/pushable blocks). Gemma is deliberately NOT asked to re-derive
   positions/sizes/colors -- code already computed those exactly via
   `_find_blobs`, and handing Gemma the same numbers as grounding avoids the
   coordinate-hallucination risk that a from-scratch vision description would
   have. This also avoids the self-identification failure mode: Gemma is
   never asked "which one is me", only to describe what's already there.

3. **Mixed planning**: the brain can now reply with either a landmark id
   (existing behavior -- a pathfinder gets you there deterministically) OR an
   action name directly, for the case where the movement map shows no action
   moving `self` at all -- there, "walk toward a landmark" is a meaningless
   plan, and code has no representation for whatever the real mechanic is
   (this is exactly the cd82/ka59 gap documented in llm_relay_agent_experiments
   memory). Pathfinding stays 100% deterministic code either way -- the brain
   never plans a multi-step path in text, only ever picks ONE landmark or ONE
   action per call, since LLMs are unreliable at exact-cell spatial reasoning.
"""
from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from arcengine import FrameData, GameAction, GameState

from llm_tools_agent import (
    ACTION_NAME_TO_ENUM,
    ACTION_NAMES,
    MAX_BRAIN_CALLS,
    OLLAMA_URL,
    ToolsAgent,
    _bbox_centroid,
    _bfs_path,
    _bulk_colors,
    _closest_blob,
    _find_blobs,
    _grid_of,
    _ollama_available,
    _query_brain_bounded,
)

# ACTION6 ("click") is deliberately NOT added to llm_tools_agent's shared
# ACTION_SPACE -- that module is also used by the plain ToolsAgent, and every
# game we've tested that mixes ACTION6 with movement actions would otherwise
# have it show up in ToolsAgent's bootstrap round-robin too, wasting a slot on
# a context-free click (x=0,y=0) that class has no way to aim. Kept local to
# this file, added only where explicitly handled (see _choose_click_action).
_CLICK_ACTION_NAMES = dict(ACTION_NAMES)
_CLICK_ACTION_NAMES[GameAction.ACTION6.value] = "ACTION6"
_CLICK_ACTION_NAME_TO_ENUM = dict(ACTION_NAME_TO_ENUM)
_CLICK_ACTION_NAME_TO_ENUM["ACTION6"] = GameAction.ACTION6

EYES_MODEL = "gemma4:e4b"
UPSCALE_SIZE = 512
MAX_EYES_CALLS = 1  # one static scene description per level is the design ask --
                     # cheap (not per-turn) and re-triggered only on a level change
MODEL_TRIGGER_INTERVAL = 15  # periodic check-in, regardless of bootstrap/goal state --
                              # the "ask only once a fresh goal is needed" trigger can
                              # only ever fire AFTER self-identification succeeds, so on
                              # a game where it never does (ka59/vc33-style: no single
                              # rigid avatar, or click-only actions our bootstrap can't
                              # interpret), the models were never being consulted at all
                              # (confirmed empirically: brain_call_count/eyes_call_count
                              # stayed 0 across a full 300-action ka59/vc33 run -- see
                              # feedback_verify_before_asserting). A periodic trigger
                              # fires during bootstrap too, giving Gemma+Qwen a chance to
                              # suggest an action to try even when code alone is stuck in
                              # a plain round-robin exploration loop.

EYES_SYSTEM_PROMPT = """\
You are the vision module for a game-playing agent that already has EXACT, \
code-computed data about every distinct-colored object on the grid (their \
pixel bounding boxes, colors, and sizes) -- you do not need to re-derive \
positions, sizes, or colors, they are given to you already and are certain, \
not guesses.

Your job is to add what the exact numeric data cannot capture: what each \
object actually LOOKS like (its shape, e.g. arrow/dial/door/block/dot/bar), \
and any structural pattern across multiple objects that hints at how they \
might behave -- e.g. objects arranged in a ring or arc (possibly a rotating \
dial or selector), a cluster of identical small blocks (possibly Sokoban-style \
pushable objects), a row of similar small shapes (possibly buttons or a \
selector). Do not guess what the game's objective is or what any action does. \
Be concise (4-6 sentences), and reference the object list's ids (e.g. blob_2) \
when describing something.
"""

BRAIN_SYSTEM_PROMPT = """\
You are the decision module for a game-playing agent. A perception layer has \
already computed, with certainty (not guesswork): your own position (if \
found yet), a list of distinct landmarks visible on the grid, which actions \
have a CONFIRMED spatial movement effect on your position (and by exactly \
how much), and a log of any cause-and-effect discovered so far. A vision \
module has also added qualitative notes about what things look like and any \
structural patterns it noticed (this part is a description, not certain \
fact -- weigh it accordingly).

Normally, pick a landmark to move toward -- a pathfinder handles getting you \
there using the confirmed movement laws, you do not need to plan the route \
yourself. But if the movement laws show NO action reliably moves anything \
(or your own position hasn't even been found yet, meaning no single object \
is reliably tracking as "you"), "move toward a landmark" is not a meaningful \
plan -- the real mechanic probably isn't about walking around a maze (it \
could be a selector/dial you cycle through, a button you press, or something \
the vision notes hint at). In that case, suggest trying a specific ACTION \
directly instead of a landmark.

You do not know what any landmark does until you've touched it or the effects \
log tells you. Prefer landmarks you haven't visited yet, unless the effects \
log suggests a specific landmark is useful to revisit. Your goal is to reach \
a WIN state / complete a level.

You keep a short, persistent scratchpad of your own best current hypothesis \
about how THIS SPECIFIC game works -- what the goal seems to be, which kind \
of landmark is worth prioritizing, anything that tripped you up -- carried \
forward across calls instead of being rederived from scratch each time. You \
MUST write a concrete, specific sentence describing what THIS game actually \
seems to be about, based on what you see below (including the vision \
module's notes) -- not a generic restatement of these instructions and not a \
placeholder. Revise it as you learn more; only repeat the exact same wording \
if it is still fully accurate and there is truly nothing new to add.

Respond in exactly three parts, each on its own line:
1. A short reasoning (1 sentence).
2. A line starting with "NOTES:" followed by your specific, concrete \
one-sentence hypothesis about this game -- refine or extend your previous \
notes, don't discard them unless they turned out wrong.
3. On the LAST line, write EITHER one landmark id (e.g. "blob_2") OR one \
available action name (e.g. "ACTION3") -- whichever you're recommending -- \
and nothing else.
"""


def _grid_to_image_b64(grid: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=UPSCALE_SIZE // 6)
    ax.imshow(grid.astype(float), vmin=0, vmax=15, cmap="tab20", interpolation="nearest")
    ax.axis("off")
    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _query_eyes(image_b64: str, blob_lines: list[str]) -> str:
    # Local-dev only for now: Ollama-served Gemma. Porting this to a bundled
    # CPU GGUF (llama-cpp-python supports multimodal via an mmproj file) is a
    # separate, later decision -- see feedback_verify_before_asserting, don't
    # assume it'll port cleanly without testing that path directly first.
    if not _ollama_available():
        raise RuntimeError("Ollama not available -- vision scene notes need a local Ollama+Gemma "
                            "for now, not yet ported to the bundled-GGUF Kaggle path")
    import requests
    user_content = (
        "Exact object list (already computed, certain):\n" + "\n".join(blob_lines) +
        "\n\nDescribe what these look like and any structural pattern you notice."
    )
    resp = requests.post(OLLAMA_URL, json={
        "model": EYES_MODEL,
        "messages": [
            {"role": "system", "content": EYES_SYSTEM_PROMPT},
            {"role": "user", "content": user_content, "images": [image_b64]},
        ],
        "stream": False,
        "options": {"temperature": 0.2},
    }, timeout=180)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


class VisionToolsAgent(ToolsAgent):
    """ToolsAgent + a movement-law summary and a one-shot Gemma scene
    description in the brain's prompt, plus letting the brain suggest a
    direct action instead of a landmark when movement looks non-spatial.

    PERIODIC_MODE controls how the periodic (every MODEL_TRIGGER_INTERVAL
    actions) check-in competes for brain-call budget against the normal
    "fresh goal needed" trigger -- found empirically to matter: on ls20
    (where the fresh-goal trigger alone already worked well, ~5/5 wins),
    sharing one budget between both triggers measurably dropped the win rate
    (~2/5) because periodic call-outs could spend the shared budget on a
    less pivotal moment than a later fresh-goal pick would have used it on.
    Three modes, compared directly against each other on ls20 (win rate) and
    ka59 (does the periodic path still rescue an otherwise-idle brain):
      - "shared": periodic and fresh-goal triggers draw from the same
        MAX_BRAIN_CALLS pool (original, naive version).
      - "separate": periodic gets its own MAX_PERIODIC_CALLS budget, on top
        of MAX_BRAIN_CALLS -- doesn't cannibalize fresh-goal calls, but
        raises the worst-case total call count (and thus worst-case latency
        contribution) per game.
      - "stuck_only": periodic only actually consults when goal_fail_count
        has exceeded STUCK_THRESHOLD (i.e. we're genuinely stuck, not just
        at a 15-action boundary), still drawing from MAX_BRAIN_CALLS -- no
        extra worst-case latency, but only helps once things have already
        gone wrong for a while.
    """

    PERIODIC_MODE = "shared"
    MAX_PERIODIC_CALLS = 2
    STUCK_THRESHOLD = 3

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.scene_notes = ""
        self.eyes_call_count = 0
        self.periodic_call_count = 0

    def _periodic_due(self) -> bool:
        if self.action_counter <= 0 or self.action_counter % MODEL_TRIGGER_INTERVAL != 0:
            return False
        if self.PERIODIC_MODE == "separate":
            return self.periodic_call_count < self.MAX_PERIODIC_CALLS
        if self.PERIODIC_MODE == "stuck_only":
            return self.goal_fail_count > self.STUCK_THRESHOLD and self.brain_call_count < MAX_BRAIN_CALLS
        return self.brain_call_count < MAX_BRAIN_CALLS  # "shared" (default)

    def _periodic_counter_attr(self) -> str:
        return "periodic_call_count" if self.PERIODIC_MODE == "separate" else "brain_call_count"

    def _choose_click_action(self, grid: np.ndarray, blobs: list[dict], legal_names: list[str],
                              periodic_due: bool, periodic_counter_attr: str) -> GameAction:
        """ACTION6 ("click") has completely different semantics from every
        other action here: there's no self-avatar to move, no BFS path to
        execute -- each turn IS just "click somewhere". Reuses the same
        blob-detection + brain/vision consultation as the movement path, but
        the brain's chosen blob becomes a click TARGET directly instead of a
        BFS goal.
        NOTE, not yet fixed -- known issue before this can run multi-threaded:
        GameAction enum members are process-wide singletons, so `.set_data()`
        mutates shared state on the single `GameAction.ACTION6` object itself.
        Harmless for this single-threaded local test, but a real race
        condition once reused across Swarm's ~110 concurrent per-game threads
        on the real Kaggle submission -- two games choosing ACTION6 around the
        same time could interleave `.set_data()` calls and one could send the
        other's click coordinates. Needs a fix (e.g. constructing a fresh
        ComplexAction instead of mutating the singleton) before real use."""
        target = None
        if blobs and self.brain_call_count < MAX_BRAIN_CALLS and (periodic_due or not self.tried_actions):
            target, _ = self._consult_models(grid, blobs, legal_names, self_pos=None,
                                              counter_attr=periodic_counter_attr)
        if target is None and blobs:
            unvisited = [b for b in blobs
                         if (b["color"], b["bbox"][0], b["bbox"][2]) not in self.visited_blob_keys]
            target = unvisited[0] if unvisited else blobs[self.action_counter % len(blobs)]
        if target is not None:
            self.visited_blob_keys.add((target["color"], target["bbox"][0], target["bbox"][2]))
            cy, cx = target["centroid"]
        else:
            cy, cx = grid.shape[0] / 2, grid.shape[1] / 2  # no blobs at all -- click center as a fallback
        action = GameAction.ACTION6
        action.set_data({"x": int(round(cx)), "y": int(round(cy))})
        self.tried_actions.add("ACTION6")
        return action

    # _movement_summary: inherited from ToolsAgent unchanged (was a byte-identical
    # duplicate override here before 2026-09-04's diff-magnitude-stats addition --
    # removed so both classes automatically share one implementation instead of
    # needing every future improvement ported to two places)

    def _consult_models(self, grid: np.ndarray, blobs: list[dict], legal_names: list[str],
                         self_pos: tuple[float, float] | None,
                         counter_attr: str = "brain_call_count") -> tuple[dict | None, str | None]:
        """Ask Gemma (scene notes, cached per-game) + Qwen (goal-or-action choice).
        Returns (target_blob_or_None, direct_action_name_or_None). Works whether or
        not `self` has been identified yet (self_pos may be None) -- see
        MODEL_TRIGGER_INTERVAL's comment for why this must not assume self is known.
        `counter_attr` selects which budget counter this call draws down -- see
        PERIODIC_MODE's docstring for why "separate" mode needs a distinct counter
        from the normal fresh-goal trigger."""
        id_map = {}
        blob_lines = []
        for i, b in enumerate(blobs):
            bid = f"blob_{i}"
            id_map[bid] = b
            visited = "visited" if (b["color"], b["bbox"][0], b["bbox"][2]) in self.visited_blob_keys else "unvisited"
            blob_lines.append(f"{bid}: color={b['color']} pos={b['centroid']} size={b['size']} ({visited})")
        blob_lines = blob_lines or ["(none detected)"]

        if not self.scene_notes and self.eyes_call_count < MAX_EYES_CALLS:
            self.eyes_call_count += 1
            try:
                image_b64 = _grid_to_image_b64(grid)
                self.scene_notes = _query_eyes(image_b64, blob_lines)
            except Exception as e:
                print(f"[VisionToolsAgent] eyes query failed: {e!r}")
                self.scene_notes = "(vision module unavailable this run)"

        effects_lines = []
        for e in self.effects_log:
            if e.get("effect_type") == "resource_bar_grew":
                effects_lines.append(
                    f"- touching color {e['touched_color']} at {e['touched_pos']} GREW a resource/time "
                    f"bar (color {e['bar_color']}) from {e['fragments_before']} to {e['fragments_after']} "
                    f"segments -- this likely grants more moves/time, worth prioritizing if you have "
                    f"other unvisited landmarks like this one")
            else:
                effects_lines.append(
                    f"- touching color {e['touched_color']} at {e['touched_pos']} changed region "
                    f"rows{e['effect_region'][0]}-{e['effect_region'][1]} cols{e['effect_region'][2]}-{e['effect_region'][3]}")
        effects_lines = effects_lines or ["(none discovered yet)"]

        position_line = (f"Your position: {self_pos}" if self_pos is not None else
                          "Your position: NOT YET FOUND -- no single object has been reliably "
                          "tracked as \"you\" despite trying multiple actions (see movement laws "
                          "below). \"Move toward a landmark\" is likely not applicable here.")

        prompt = (
            f"{position_line}\n\n"
            f"Your notes from earlier turns:\n{self.brain_notes or '(none yet -- this is your first consult this game)'}\n\n"
            f"Movement laws discovered so far:\n{self._movement_summary(legal_names)}\n\n"
            f"Vision module's scene notes:\n{self.scene_notes or '(not available this run)'}\n\n"
            f"Landmarks visible now:\n" + "\n".join(blob_lines) + "\n\n"
            f"Effects discovered so far:\n" + "\n".join(effects_lines) + "\n\n"
            f"Available actions: {', '.join(legal_names)}\n"
            f"Which landmark should you move toward next, or which action should you try directly?"
        )
        # count the ATTEMPT, not just a successful reply -- see llm_tools_agent's identical note
        setattr(self, counter_attr, getattr(self, counter_attr) + 1)
        try:
            reply = _query_brain_bounded(prompt, BRAIN_SYSTEM_PROMPT)
        except Exception as e:
            reply = ""
            print(f"[VisionToolsAgent] brain query failed/timed out: {e!r}")
        self.last_raw_response = reply
        self._update_brain_notes(reply)  # inherited from ToolsAgent, same NOTES: parsing

        chosen = None
        for line in reversed(reply.strip().splitlines()):
            line = line.strip()
            if line in id_map or line in legal_names:
                chosen = line
                break
        if chosen in id_map:
            return id_map[chosen], None
        if chosen in legal_names:
            return None, chosen
        return None, None

    def _choose_action_impl(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.prev_grid = None
            self.prev_action_name = None
            self.self_bbox = None
            self.current_path = []
            self.current_goal_key = None
            return GameAction.RESET

        legal = [a for a in latest_frame.available_actions if a in _CLICK_ACTION_NAMES]
        if not legal:
            return GameAction.RESET
        legal_names = [_CLICK_ACTION_NAMES[a] for a in legal]

        grid = _grid_of(latest_frame)
        level_changed = (self.prev_levels_completed is not None and
                          latest_frame.levels_completed != self.prev_levels_completed)
        if level_changed:
            self.prev_grid = None
            # deliberately NOT clearing self.scene_notes here: MAX_EYES_CALLS is a
            # per-GAME budget (not per-level), so blanking it on every level change
            # could permanently lose the only description this game ever gets once
            # the budget is spent. A slightly-stale description from an earlier
            # level is still more useful context than none.
        self.prev_levels_completed = latest_frame.levels_completed
        self._update_from_last_transition(grid)

        bulk = _bulk_colors(grid)
        blobs = _find_blobs(grid, bulk | self.self_colors | self.attached_colors)

        # periodic model check-in, independent of bootstrap/goal state -- see
        # MODEL_TRIGGER_INTERVAL's comment: without this, a game where
        # self-identification never succeeds (ka59/vc33-style mechanics) NEVER
        # reaches the brain/eyes at all, confirmed empirically (both stayed at 0
        # calls across a full 300-action run on each). See PERIODIC_MODE's
        # docstring for why which counter this draws down is configurable.
        periodic_due = self._periodic_due()
        periodic_counter_attr = self._periodic_counter_attr()

        if not self.self_colors:
            if "ACTION6" in legal_names:
                # a click-capable game where no movement-based self has ever been
                # identified (generic condition -- true for any click-only game,
                # not specifically vc33, see feedback_no_game_hacking) -- the
                # whole self/BFS model doesn't apply, route to the click handler
                action = self._choose_click_action(grid, blobs, legal_names, periodic_due, periodic_counter_attr)
                self.prev_grid = grid
                self.prev_action_name = action.name
                return action
            if periodic_due:
                _, direct_action_name = self._consult_models(
                    grid, blobs, legal_names, self_pos=None, counter_attr=periodic_counter_attr)
                if direct_action_name is not None:
                    self.tried_actions.add(direct_action_name)
                    self.prev_grid = grid
                    self.prev_action_name = direct_action_name
                    return _CLICK_ACTION_NAME_TO_ENUM[direct_action_name]
            untried = [a for a in legal_names if a not in self.tried_actions]
            chosen_name = untried[0] if untried else legal_names[self.action_counter % len(legal_names)]
            action = _CLICK_ACTION_NAME_TO_ENUM[chosen_name]
            self.tried_actions.add(chosen_name)
            self.prev_grid = grid
            self.prev_action_name = chosen_name
            return action

        self_pos = _bbox_centroid(self.self_bbox) if self.self_bbox else self.prev_self_pos
        self.prev_self_pos = self_pos
        self_pos_int = (int(round(self_pos[0])), int(round(self_pos[1]))) if self_pos else (0, 0)

        if not self.current_path:
            # NOTE: this consult must NOT be gated on `if blobs:` -- empty blobs
            # (e.g. ka59: pushable objects fragment past MAX_FRAGMENTS_PER_COLOR
            # and get excluded as "bar-like" clutter) is exactly one of the stuck
            # states the periodic trigger exists to break out of. Confirmed
            # empirically: gating this on `blobs` left periodic_due=True doing
            # nothing for the entire second half of a ka59 run once blobs hit 0.
            fresh_goal_due = bool(blobs) and self.goal_fail_count == 0 and self.brain_call_count < MAX_BRAIN_CALLS
            ask_brain = fresh_goal_due or periodic_due
            target = None
            direct_action_name = None
            if ask_brain:
                # fresh-goal-need always draws from the main budget when both
                # triggers coincide -- it's the primary reason, periodic is the
                # opportunistic extra
                counter_attr = "brain_call_count" if fresh_goal_due else periodic_counter_attr
                target, direct_action_name = self._consult_models(
                    grid, blobs, legal_names, self_pos, counter_attr=counter_attr)

            if direct_action_name is not None:
                # brain judged the mechanic isn't spatial pathing -- try its
                # suggested action directly instead of routing through BFS
                self.tried_actions.add(direct_action_name)
                self.prev_grid = grid
                self.prev_action_name = direct_action_name
                return _CLICK_ACTION_NAME_TO_ENUM[direct_action_name]

            if target is None and blobs:
                unvisited = [b for b in blobs
                             if (b["color"], b["bbox"][0], b["bbox"][2]) not in self.visited_blob_keys]
                target = _closest_blob(unvisited, self_pos) or _closest_blob(blobs, self_pos)
            if target is not None:
                self.current_goal_key = (target["color"], target["bbox"][0], target["bbox"][2])
                self.visited_blob_keys.add(self.current_goal_key)
                path = _bfs_path(grid, self_pos_int, target["bbox"], self.blocked_values, self.action_deltas)
                self.current_path = path or []
            if not self.current_path:
                untried = [a for a in legal_names if a not in self.tried_actions]
                chosen_name = untried[0] if untried else legal_names[self.action_counter % len(legal_names)]
                self.tried_actions.add(chosen_name)
                self.prev_grid = grid
                self.prev_action_name = chosen_name
                self.goal_fail_count += 1
                return _CLICK_ACTION_NAME_TO_ENUM[chosen_name]

        action = self.current_path.pop(0)
        self.prev_grid = grid
        self.prev_action_name = action.name
        if not self.current_path:
            self.goal_fail_count = 0
        return action
