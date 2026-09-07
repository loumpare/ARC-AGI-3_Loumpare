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
import queue
import threading
from collections import deque
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from arcengine import FrameData, GameAction, GameState

from llm_tools_agent import (
    ACTION_NAME_TO_ENUM,
    ACTION_NAMES,
    BAR_HISTORY_WINDOW,
    DIRECT_ACTION_REPEAT,
    EXTRA_STEP_BUDGET,
    NOTES_MAX_CHARS,
    OLLAMA_URL,
    STUCK_MAX_DISTINCT_BBOX,
    ToolsAgent,
    _bbox_centroid,
    _bfs_path,
    _bulk_colors,
    _closest_blob,
    _find_blobs,
    _GGUF_SEARCH_ROOTS,
    _gpu_offload_available,
    _grid_of,
    _ollama_available,
    _pending_brain_threads,
    _pending_brain_threads_lock,
    _query_brain_bounded,
    _raw_mask_bbox,
    _structural_fact_lines,
    _usable_cpu_count,
)
from state_graph import state_signature

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

EYES_MODEL = "gemma4:e4b"  # local Ollama dev only -- see _query_eyes
UPSCALE_SIZE = 512
MAX_EYES_CALLS = 1  # one static scene description per level is the design ask --
                     # cheap (not per-turn) and re-triggered only on a level change

# Kaggle GPU path: our own qwen3.8-27B GGUF + mmproj (same weights already used for
# the text-only brain's model family, extracted from local Ollama blobs -- not a
# third-party model), run in-process via llama-cpp-python's MTMDChatHandler with
# full GPU offload. Deliberately GPU-ONLY: a CPU-only run of this same model
# measured ~170s/call locally (see llm_relay_agent_experiments memory,
# 2026-09-06), which would blow the per-game latency budget many times over
# under real ~110-way Kaggle thread contention -- confirmed via a real Kaggle
# benchmark kernel (notebooks/gpu_bench/) that full GPU offload on the
# competition's RTX PRO 6000 brings the SAME model+prompt down to ~2-6s/call, so
# unlike the brain's CPU-only path this one is only worth attempting with a real
# GPU present; if the GPU isn't recognized (offload unsupported), _query_eyes
# skips straight to "vision unavailable" rather than eating a 170s CPU fallback.
_VISION_GGUF_FILENAME = "qwen3.8-27b.gguf"
_VISION_MMPROJ_FILENAME = "qwen3.8-27b-mmproj.gguf"
_vision_llama_singleton = None  # (Llama, MTMDChatHandler) tuple, lazily built -- see _get_local_vision_llama
_vision_llama_lock = threading.Lock()
EYES_CALL_TIMEOUT_S = 60  # generous vs the ~2-6s measured on the real RTX PRO 6000 (see
                           # notebooks/gpu_bench/), but still bounded so one stuck native
                           # call can't hang a game thread indefinitely -- same rationale as
                           # BRAIN_CALL_TIMEOUT_S in llm_tools_agent.py


def _find_vision_file(filename: str) -> Path:
    # same search-by-filename approach as llm_tools_agent._find_gguf, for the
    # same reason: Kaggle's exact model_sources mount path varies by
    # owner/slug/framework/instance/version and shouldn't be hardcoded (see
    # feedback_verify_before_asserting)
    for root in _GGUF_SEARCH_ROOTS:
        if not root.exists():
            continue
        for match in root.rglob(filename):
            return match
    raise FileNotFoundError(f"{filename} not found under any of: {_GGUF_SEARCH_ROOTS}")


def _get_local_vision_llama():
    global _vision_llama_singleton
    if _vision_llama_singleton is not None:
        return _vision_llama_singleton
    with _vision_llama_lock:
        if _vision_llama_singleton is None:  # re-check: another thread may have won the race
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import MTMDChatHandler
            mmproj_path = _find_vision_file(_VISION_MMPROJ_FILENAME)
            handler = MTMDChatHandler(clip_model_path=str(mmproj_path), verbose=False)
            model_path = _find_vision_file(_VISION_GGUF_FILENAME)
            llm = Llama(model_path=str(model_path), chat_handler=handler, n_ctx=4096,
                        n_gpu_layers=-1, n_threads=_usable_cpu_count(), verbose=False)
            _vision_llama_singleton = (llm, handler)
        return _vision_llama_singleton


def _query_eyes_gguf(image_b64: str, blob_lines: list[str]) -> str:
    """GPU-GGUF path for _query_eyes -- see the module-level comment above
    _VISION_GGUF_FILENAME for why this is GPU-only. Mirrors llm_tools_agent's
    `_query_brain`'s local-GGUF branch: one shared Llama instance, serialized
    behind a lock since llama-cpp-python inference isn't safe for concurrent
    calls on one instance (same reasoning as _llama_lock there)."""
    llm, handler = _get_local_vision_llama()
    user_text = (
        "Exact object list (already computed, certain):\n" + "\n".join(blob_lines) +
        "\n\nDescribe what these look like and any structural pattern you notice."
    )
    with _vision_llama_lock:
        resp = handler(
            llama=llm,
            messages=[
                {"role": "system", "content": EYES_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ]},
            ],
            max_tokens=300,
            temperature=0.2,
            # "low" was found, empirically, to answer just as completely as the
            # default "xhigh" (shorter <think> block, not a truncated one) while
            # being meaningfully faster on the real GPU (2.2s vs 6.2s measured on
            # the same prompt -- see notebooks/gpu_bench/, 2026-09-06) -- unlike
            # on CPU, where reasoning_effort barely moved total wall-clock time.
            reasoning_effort="low",
        )
    return resp["choices"][0]["message"]["content"]


def _query_eyes_bounded(image_b64: str, blob_lines: list[str]) -> str:
    """Same bounded-daemon-thread pattern as llm_tools_agent._query_brain_bounded
    (see that function's docstring for why an unbounded call is a whole-submission
    hang risk, not just a slow one) -- reuses that module's SAME pending-thread
    list/lock so the one atexit drain hook there covers eyes calls too instead of
    needing a second copy of that exit-time-segfault-avoidance machinery."""
    outcome: queue.Queue = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            outcome.put(("ok", _query_eyes_gguf(image_b64, blob_lines)))
        except Exception as e:  # noqa: BLE001 -- forwarded to the caller below
            outcome.put(("err", e))

    t = threading.Thread(target=worker, daemon=True)
    with _pending_brain_threads_lock:
        _pending_brain_threads.append(t)
    t.start()
    try:
        status, payload = outcome.get(timeout=EYES_CALL_TIMEOUT_S)
    except queue.Empty:
        raise TimeoutError(f"eyes call exceeded {EYES_CALL_TIMEOUT_S}s") from None
    if status == "err":
        raise payload
    return payload
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

Each object's id label ("blob_0", "blob_1", ...) is drawn directly ON the \
image next to it -- use that to check which id you actually mean before \
naming it, instead of guessing which shape an id refers to.

Before calling two same-colored objects "separate" or "parallel", check \
their given bounding boxes: if they're nearly touching (close row/col \
ranges) with a THIRD, differently-colored object sitting in the gap between \
them, that's much more likely to be ONE object interrupted by that other \
object (e.g. a bar with a position marker on it) than two independent ones. \
Say so explicitly when you notice this pattern.

Keep uncertain interpretation clearly separate from what the object list \
already guarantees as fact. Phrase guesses as "possibly"/"likely"/"could be" \
-- do not state an interpretation (what something IS or DOES) with the same \
confidence as a given position/color/size.
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

REFLECTION_SYSTEM_PROMPT = """\
You are periodically reviewing an in-progress ARC-AGI-3 game session, using \
the WHOLE trajectory so far (several sampled frames across the whole episode, \
in order, plus the complete action log) instead of just the single current \
frame a normal turn-by-turn decision is based on.

Based on this wider view, write an updated, concrete hypothesis about this \
game's actual goal and mechanic, and what you'd recommend trying differently \
going forward. Look across the whole sequence for patterns a single frame \
can't reveal: what changed together, what stayed constant, whether the same \
action ever produced different effects, whether progress correlates with a \
specific region/object/color, and whether recent actions have actually been \
making a difference or just repeating without effect.

Respond with 2-4 concrete sentences only (this replaces the agent's working \
notes for all future turns) -- no restating these instructions, no \
placeholder text, no preamble.
"""
N_REFLECTION_FRAMES = 8
REFLECTION_TIMEOUT_S = 240


def _sample_frame_indices(n_frames: int, n_sample: int) -> list[int]:
    if n_frames <= n_sample:
        return list(range(n_frames))
    return sorted(set(round(i * (n_frames - 1) / (n_sample - 1)) for i in range(n_sample)))


def _grid_to_image_b64(grid: np.ndarray, blobs: list[dict] | None = None) -> str:
    """Renders the grid, and if `blobs` is given, labels each one "blob_N"
    directly on the image next to its centroid -- see EYES_SYSTEM_PROMPT: this
    lets Gemma check which shape an id actually refers to instead of having to
    match a separate text list to the image by eye, which was a source of
    hallucinated blob-to-shape mappings (confirmed by inspecting real replies
    against the actual grid, 2026-09-04 -- e.g. "blob_3, blob_4, blob_5 grouped
    in the lower right" on ls20 didn't match anything actually in that region)."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=UPSCALE_SIZE // 6)
    ax.imshow(grid.astype(float), vmin=0, vmax=15, cmap="tab20", interpolation="nearest")
    ax.axis("off")
    if blobs:
        for i, b in enumerate(blobs):
            y, x = b["centroid"]
            ax.annotate(f"blob_{i}", (x, y), color="white", fontsize=7, fontweight="bold",
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6, edgecolor="none"))
    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _query_eyes(image_b64: str, blob_lines: list[str]) -> str:
    # Local dev: Ollama-served Gemma, unchanged. Kaggle (no Ollama, no internet):
    # falls through to our own bundled qwen3.8 GGUF+mmproj, GPU-offloaded -- see
    # _query_eyes_gguf's docstring for why this branch is GPU-only (a CPU
    # fallback here would blow the per-game latency budget, unlike the brain's
    # CPU-viable path in llm_tools_agent.py).
    if _ollama_available():
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
    if not _gpu_offload_available():
        raise RuntimeError("neither Ollama nor GPU offload available -- vision scene notes need "
                            "either a local Ollama+Gemma (dev) or a real GPU for the bundled "
                            "qwen3.8 GGUF (Kaggle); CPU-only fallback deliberately not attempted "
                            "(see _query_eyes_gguf docstring)")
    return _query_eyes_bounded(image_b64, blob_lines)


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
    # Whole-GAME brain-call ceiling for this class -- deliberately separate from
    # ToolsAgent's MAX_BRAIN_CALLS (a PER-LEVEL cap there, see llm_tools_agent.py).
    # Every gate below checks this against brain_call_count (whole-game running
    # total), so reusing the imported MAX_BRAIN_CALLS=4 here effectively meant
    # "4 calls for the entire 150-action game" -- confirmed via a real 25-game
    # benchmark (2026-09-07, results/unified_calib_25game_20260907.json) that
    # brain_call_count was EXACTLY 4 in every one of the 12 games where self was
    # found but the level was never completed, with action_counter=151 -- i.e.
    # the last ~90 actions of every one of those runs were spent on autopilot
    # with zero further brain consultation, even when brain_notes showed clear
    # unresolved confusion about the game's mechanic. Raised well above 4 so the
    # periodic trigger (MODEL_TRIGGER_INTERVAL=15) can keep firing through
    # roughly the whole action budget instead of exhausting itself in the first
    # ~60 actions.
    MAX_BRAIN_CALLS_GAME = 10
    REFLECTION_MODEL = EYES_MODEL  # UnifiedVisionAgent overrides this to its own single model

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.scene_notes = ""
        self.eyes_call_count = 0
        self.periodic_call_count = 0

    def _maybe_reflect(self) -> None:
        """Overrides ToolsAgent's no-op: periodically (see REFLECTION_INTERVAL)
        show the model several sampled frames across the WHOLE episode so far
        (via self.frames, populated by the base Agent class every turn -- no
        new frame storage needed) plus the full action log, and let it rewrite
        brain_notes with a trajectory-grounded hypothesis. Added 2026-09-07
        after a retrospective test (results/reflection_20260907/) showed the
        model reasons much better given the whole trajectory at once than one
        frame at a time -- this brings a (cheaper, periodic) version of that
        into live play instead of only as an after-the-fact diagnostic.
        Local-dev (Ollama) only for now -- silently skipped when only the
        Kaggle GGUF path is available (single-image only currently, see
        _query_eyes_gguf/_query_unified_gguf); this budget-gated (
        MAX_REFLECTION_CALLS) side channel is additive, so skipping it just
        means brain_notes keeps evolving the normal single-frame way, exactly
        today's behavior."""
        if not _ollama_available():
            return
        self.reflection_call_count += 1
        try:
            import requests
            # self.frames[0] is a placeholder FrameData with no `.frame` populated at all (see
            # agents/agent.py's Agent.__init__: `self.frames = [FrameData(levels_completed=0)]`) --
            # real frames start at index 1, one per action taken so far.
            real_frames = self.frames[1:]
            if not real_frames:
                return
            idxs = _sample_frame_indices(len(real_frames), N_REFLECTION_FRAMES)
            images_b64 = [_grid_to_image_b64(np.array(real_frames[i].frame[0], dtype=int))
                          for i in idxs]
            frame_labels = "\n".join(
                f"Image {k + 1} = the grid after step {idx + 1}."
                for k, idx in enumerate(idxs))
            action_log = "\n".join(f"step {i + 1}: {a}" for i, a in enumerate(self.action_history))
            user_prompt = (
                f"You are shown {len(idxs)} sampled frames from THIS episode so far "
                f"({len(self.action_history)} actions taken), in order:\n{frame_labels}\n\n"
                f"Full action log so far:\n{action_log}\n\n"
                f"Your current notes (from single-frame turns):\n{self.brain_notes or '(none yet)'}\n\n"
                f"Write your updated hypothesis as instructed."
            )
            resp = requests.post(OLLAMA_URL, json={
                "model": self.REFLECTION_MODEL,
                "messages": [
                    {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt, "images": images_b64},
                ],
                "stream": False,
                "options": {"temperature": 0.3},
            }, timeout=REFLECTION_TIMEOUT_S)
            resp.raise_for_status()
            reflection = resp.json()["message"]["content"].strip()
            if reflection:
                self.brain_notes = reflection[:NOTES_MAX_CHARS]
        except Exception as e:
            print(f"[VisionToolsAgent] reflection failed (non-fatal, notes unchanged): {e!r}")

    def _periodic_due(self) -> bool:
        if self.action_counter <= 0 or self.action_counter % MODEL_TRIGGER_INTERVAL != 0:
            return False
        if self.PERIODIC_MODE == "separate":
            return self.periodic_call_count < self.MAX_PERIODIC_CALLS
        if self.PERIODIC_MODE == "stuck_only":
            return self.goal_fail_count > self.STUCK_THRESHOLD and self.brain_call_count < self.MAX_BRAIN_CALLS_GAME
        return self.brain_call_count < self.MAX_BRAIN_CALLS_GAME  # "shared" (default)

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
        if blobs and self.brain_call_count < self.MAX_BRAIN_CALLS_GAME and (periodic_due or not self.tried_actions):
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
        structural_lines = _structural_fact_lines(blobs, grid, _bulk_colors(grid))

        if not self.scene_notes and self.eyes_call_count < MAX_EYES_CALLS:
            self.eyes_call_count += 1
            try:
                image_b64 = _grid_to_image_b64(grid, blobs)
                # structural facts (e.g. "blob_2/blob_3 are one bar split by blob_7")
                # are GIVEN, computed exactly from positions -- pass them alongside
                # the object list so Gemma doesn't waste effort re-guessing (and
                # getting wrong, see llm_relay_agent_experiments memory 2026-09-04)
                # something code already knows for certain.
                eyes_lines = list(blob_lines)
                if structural_lines:
                    eyes_lines.append("Also already computed, certain (not guesses):")
                    eyes_lines.extend(structural_lines)
                self.scene_notes = _query_eyes(image_b64, eyes_lines)
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

        cross_game_hints = self._sync_shared_action_memory(legal_names)
        progress_signal = self._progress_signal_summary()
        prompt = (
            f"{position_line}\n\n"
            f"Your notes from earlier turns:\n{self.brain_notes or '(none yet -- this is your first consult this game)'}\n\n"
            f"Movement laws discovered so far:\n{self._movement_summary(legal_names)}\n\n"
            f"Progress signal (from actual gameplay, not a guess):\n{progress_signal}\n\n"
            f"Hints from OTHER games played this session (same action vocabulary, "
            f"DIFFERENT game -- may not apply here, treat as a weak prior only):\n{cross_game_hints}\n\n"
            f"Vision module's scene notes:\n{self.scene_notes or '(not available this run)'}\n\n"
            f"Landmarks visible now:\n" + "\n".join(blob_lines) + "\n\n" +
            (f"Structural facts computed exactly from positions (not guesses):\n"
             + "\n".join(structural_lines) + "\n\n" if structural_lines else "") +
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
        # ====================================================================
        # REPÈRE FR -- C'EST ICI la boucle de décision/déplacement pour
        # VisionToolsAgent, ET pour UnifiedVisionAgent (qui hérite cette
        # méthode SANS la réécrire -- voir scratchpad/vision_compare/
        # unified_agent.py, il ne modifie QUE _consult_models plus haut dans
        # ce fichier, pas le déplacement). C'est une réécriture COMPLÈTE de
        # ToolsAgent._choose_action_impl (src/llm_tools_agent.py, ~ligne 1151)
        # -- pas un appel à la version parente -- donc tout correctif fait sur
        # la version de base doit être reporté ici À LA MAIN (c'est justement
        # le bug trouvé le 2026-09-05 : le correctif anti-boucle du
        # 2026-09-04 n'avait jamais été copié ici).
        #
        # Déroulé d'un tour :
        #   1. RESET/GAME_OVER -> on relance
        #   2. calcul des blobs (objets détectés) via _find_blobs
        #   3. si le joueur n'est pas encore identifié -> phase bootstrap
        #      (self.self_colors vide, voir plus bas)
        #   4. sinon -> si un chemin est déjà en cours (self.current_path),
        #      on continue de le suivre ; sinon on consulte le brain/vision
        #      (_consult_models) pour choisir une nouvelle cible, puis
        #      _bfs_path (src/llm_tools_agent.py) calcule le chemin.
        # ====================================================================
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.prev_grid = None
            self.prev_action_name = None
            self.self_bbox = None
            self.current_path = []
            self.current_goal_key = None
            self.current_goal_bbox = None  # ported 2026-09-05, see pending_arrival_check below
            self.pending_arrival_check = None
            self.pending_arrival_action = None
            self.extra_step_budget = 0
            self.extra_step_action = None
            self.recent_action_log.clear()
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
            self.pending_arrival_check = None  # ported 2026-09-05 -- see llm_tools_agent's
            self.pending_arrival_action = None  # identical resets, same reasoning (new level,
            self.extra_step_budget = 0          # any in-flight check/streak/retry-budget is stale
            self.extra_step_action = None
            self.recent_action_log.clear()
        self.prev_levels_completed = latest_frame.levels_completed
        self._update_from_last_transition(grid)

        # PORTED 2026-09-06 from llm_tools_agent.ToolsAgent's calibration phase
        # (added there same day, user-requested) -- see this class's docstring
        # warning above and llm_relay_agent_experiments memory: this method is a
        # FULL override, not an extension, so base-class fixes never propagate
        # here automatically (already bit us once, 2026-09-05, anti-loop fix).
        # Deterministically try every non-click legal action TWICE right at
        # game start, then RESET, before any goal-directed/vision-consulted
        # play begins -- verified via arcengine's own source (see the base
        # class's identical comment) that a mid-game RESET here cannot lose
        # levels_completed/level progress, only replay the current level fresh.
        if not self.calibration_done:
            if not self.calibration_actions:
                self.calibration_actions = [n for n in legal_names if n != "ACTION6"]
            if not self.calibration_actions:
                self.calibration_done = True
            elif self.calibration_reset_pending:
                self.prev_grid = None
                self.prev_action_name = None
                # see llm_tools_agent's identical fix/comment: self_bbox must NOT be
                # nulled to None here (found the hard way via a real Kaggle Phase A
                # crash, 2026-09-07) -- recompute it fresh from the just-arrived
                # post-reset grid instead, so self_pos is never None on the very
                # first post-calibration goal-directed turn.
                self.self_bbox = _raw_mask_bbox(grid, self.self_colors) if self.self_colors else None
                self.prev_self_pos = _bbox_centroid(self.self_bbox) if self.self_bbox else None
                self.current_path = []
                self.current_goal_key = None
                self.current_goal_bbox = None
                self.pending_arrival_check = None
                self.pending_arrival_action = None
                self.extra_step_budget = 0
                self.extra_step_action = None
                self.recent_action_log.clear()
                self.direct_action_repeat_remaining = 0
                self.prev_state_sig = None
                self.state_graph_path = []
                self.calibration_reset_pending = False
                self.calibration_done = True
            elif self.calibration_step < 2 * len(self.calibration_actions):
                chosen_name = self.calibration_actions[self.calibration_step % len(self.calibration_actions)]
                self.calibration_step += 1
                self.tried_actions.add(chosen_name)
                self.prev_grid = grid
                self.prev_action_name = chosen_name
                return _CLICK_ACTION_NAME_TO_ENUM[chosen_name]
            else:
                self.calibration_reset_pending = True
                self.prev_grid = grid
                self.prev_action_name = None
                return GameAction.RESET

        if not level_changed:
            # PORTED 2026-09-05 from llm_tools_agent.ToolsAgent (added there 2026-09-04,
            # never mirrored here until a real 4-config vision comparison showed all
            # four configs stuck in the identical ping-pong loop -- see
            # llm_relay_agent_experiments memory). This class's _choose_action_impl is a
            # full override, not an extension, so base-class fixes never propagated here
            # automatically -- confirmed the hard way via an instrumented ls20 replay:
            # 60/145 turns (41%) were spent re-targeting ls20's own draining step-budget
            # bar (rows 61-62) as a "new" landmark every tick it shrank, immediately
            # failing BFS (not walkable) and falling back to a round-robin
            # ACTION1->2->3->4->1 cycle -- exactly the reported "haut bas gauche droite"
            # loop, on every one of the 4 compared configs since they all share this
            # method. See the _find_blobs call below for the actual missing-exclusion
            # half of this fix.
            if self.pending_arrival_check is not None:
                arrived = False
                if self.self_bbox is not None:
                    r, c = _bbox_centroid(self.self_bbox)
                    tr0, tr1, tc0, tc1 = self.pending_arrival_check
                    arrived = tr0 - 2 <= r <= tr1 + 2 and tc0 - 2 <= c <= tc1 + 2
                if arrived:
                    self.goal_fail_count = 0
                    if self.current_goal_key is not None:
                        self.visited_blob_keys.add(self.current_goal_key)
                    # ported 2026-09-05 -- see EXTRA_STEP_BUDGET's comment in
                    # llm_tools_agent.py: arrival confirmed but no level transition
                    # followed, so try continuing the same action a few more times
                    # before picking a brand new target (many ARC mechanics need
                    # walking PAST a marker's edge, not just touching it).
                    if not level_changed and self.pending_arrival_action is not None:
                        self.extra_step_budget = EXTRA_STEP_BUDGET
                        self.extra_step_action = self.pending_arrival_action
                else:
                    self.goal_fail_count += 1
                self.pending_arrival_check = None
                self.pending_arrival_action = None

            if self.self_colors and self.prev_action_name is not None:
                self.recent_action_log.append((self.prev_action_name, self.self_bbox))
                if (len(self.recent_action_log) == self.recent_action_log.maxlen
                        and len({e[1] for e in self.recent_action_log}) <= STUCK_MAX_DISTINCT_BBOX):
                    if self.current_goal_key is not None:
                        self.blob_attempt_count[self.current_goal_key] = (
                            self.blob_attempt_count.get(self.current_goal_key, 0) + 1)
                        if self.blob_attempt_count[self.current_goal_key] >= 3:
                            self.visited_blob_keys.add(self.current_goal_key)
                    self.current_path = []
                    self.current_goal_key = None
                    self.current_goal_bbox = None
                    self.goal_fail_count = max(self.goal_fail_count, 1)
                    self.recent_action_log.clear()

        bulk = _bulk_colors(grid)
        # `| self.confirmed_bar_colors` was the missing half of the port above: this
        # attribute was already being computed correctly (inherited
        # _update_from_last_transition fills it unconditionally) but never actually
        # consulted here, so a draining bar/gauge always looked like a fresh landmark
        # regardless of the fix above.
        blobs = _find_blobs(grid, bulk | self.self_colors | self.attached_colors | self.confirmed_bar_colors)

        # Generic clustered state-graph (see state_graph.py) -- ported from
        # llm_tools_agent.py 2026-09-07 (this class never had it wired in at all
        # before, not even in bootstrap). Records EVERY transition every turn
        # regardless of self-ID/branch, and replays a known path to a previously
        # observed goal state (a levels_completed increase) the moment one is
        # available -- purely additive: a no-op until some win has actually
        # happened on this game (find_path returns None with an empty goal set),
        # so it cannot make an already-working game worse.
        sig = state_signature(blobs)
        if self.prev_state_sig is not None and self.prev_action_name is not None:
            self.state_graph.record_transition(self.prev_state_sig, self.prev_action_name, sig)
        if level_changed:
            self.state_graph.mark_goal(sig)
            self.state_graph_path = []  # a level transition invalidates any in-flight plan
            self.recent_state_sigs.clear()
            self.bar_coverage_history = {}
        self.prev_state_sig = sig

        # Generic progress/stall signal -- ported from llm_tools_agent.py 2026-09-07
        # (see PROGRESS_SIG_WINDOW/BAR_HISTORY_WINDOW and _progress_signal_summary
        # there for the full rationale). Bookkeeping only; surfaced via
        # self._progress_signal_summary() (inherited, unchanged) when consulting.
        self.cycle_detected_turns_ago = None
        for i, s in enumerate(reversed(self.recent_state_sigs)):
            if s == sig:
                self.cycle_detected_turns_ago = i + 1
                break
        self.recent_state_sigs.append(sig)
        for bar_color in self.confirmed_bar_colors:
            coverage = int(np.count_nonzero(grid == bar_color))
            self.bar_coverage_history.setdefault(bar_color, deque(maxlen=BAR_HISTORY_WINDOW)).append(coverage)

        if not self.state_graph_path:
            self.state_graph_path = self.state_graph.find_path(sig) or []
        if self.state_graph_path:
            chosen_name = self.state_graph_path.pop(0)
            if chosen_name in legal_names:
                self.tried_actions.add(chosen_name)
                self.prev_grid = grid
                self.prev_action_name = chosen_name
                return _CLICK_ACTION_NAME_TO_ENUM[chosen_name]
            self.state_graph_path = []  # stale (action no longer legal) -- replan next time

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

        # ported 2026-09-05 -- see EXTRA_STEP_BUDGET's comment in llm_tools_agent.py
        if not self.current_path and self.extra_step_budget > 0 and self.extra_step_action in legal_names:
            self.extra_step_budget -= 1
            self.tried_actions.add(self.extra_step_action)
            self.prev_grid = grid
            self.prev_action_name = self.extra_step_action
            return _CLICK_ACTION_NAME_TO_ENUM[self.extra_step_action]

        if not self.current_path:
            if self.direct_action_repeat_remaining > 0 and self.direct_action_repeat_name in legal_names:
                # mid-commitment to a brain-suggested non-spatial action -- ported
                # from llm_tools_agent.py 2026-09-07 (same fix, see that file's
                # comment): this branch previously tried a direct_action_name
                # suggestion exactly ONCE, then re-entered this block next turn
                # and immediately re-consulted (goal_fail_count still 0), which a
                # 13-game benchmark showed burned the whole per-game brain-call
                # budget testing single untested actions one at a time (cd82/
                # re86/sk48/tr87 all exhausted MAX_BRAIN_CALLS_GAME well before
                # 150 actions, then degraded to blind landmark-picking for the
                # rest of the game). Many puzzle mechanics (toggles, state
                # cycles) need several presses of the SAME action to show an
                # effect at all, not just one.
                self.direct_action_repeat_remaining -= 1
                chosen_name = self.direct_action_repeat_name
                self.tried_actions.add(chosen_name)
                self.prev_grid = grid
                self.prev_action_name = chosen_name
                return _CLICK_ACTION_NAME_TO_ENUM[chosen_name]
            # NOTE: this consult must NOT be gated on `if blobs:` -- empty blobs
            # (e.g. ka59: pushable objects fragment past MAX_FRAGMENTS_PER_COLOR
            # and get excluded as "bar-like" clutter) is exactly one of the stuck
            # states the periodic trigger exists to break out of. Confirmed
            # empirically: gating this on `blobs` left periodic_due=True doing
            # nothing for the entire second half of a ka59 run once blobs hit 0.
            fresh_goal_due = bool(blobs) and self.goal_fail_count == 0 and self.brain_call_count < self.MAX_BRAIN_CALLS_GAME
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
                self.direct_action_repeat_name = direct_action_name
                self.direct_action_repeat_remaining = DIRECT_ACTION_REPEAT - 1
                self.prev_grid = grid
                self.prev_action_name = direct_action_name
                return _CLICK_ACTION_NAME_TO_ENUM[direct_action_name]

            if target is None and blobs:
                unvisited = [b for b in blobs
                             if (b["color"], b["bbox"][0], b["bbox"][2]) not in self.visited_blob_keys]
                target = _closest_blob(unvisited, self_pos) or _closest_blob(blobs, self_pos)
            if target is not None:
                self.direct_action_repeat_remaining = 0  # a fresh spatial plan supersedes any
                                                           # stale non-spatial repeat commitment
                self.current_goal_key = (target["color"], target["bbox"][0], target["bbox"][2])
                self.current_goal_bbox = target["bbox"]  # ported 2026-09-05, see pending_arrival_check
                # PORTED 2026-09-05 (see llm_tools_agent, 2026-09-04): NOT marked visited
                # here anymore on mere selection -- only on confirmed arrival below, via
                # pending_arrival_check. blob_attempt_count is the bounded-retry safety
                # net so a target BFS can never actually reach still gets excluded after
                # 3 tries instead of looping forever.
                self.blob_attempt_count[self.current_goal_key] = (
                    self.blob_attempt_count.get(self.current_goal_key, 0) + 1)
                if self.blob_attempt_count[self.current_goal_key] >= 3:
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
        if not self.current_path and self.current_goal_bbox is not None:
            # ported 2026-09-05 -- defer the arrival verdict to next turn's top-of-call
            # resolution block instead of declaring "arrived" unconditionally right here
            self.pending_arrival_check = self.current_goal_bbox
            self.pending_arrival_action = action.name  # see extra_step_budget
        return action
