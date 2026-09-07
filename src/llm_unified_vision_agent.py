"""Unified vision+brain agent: ONE multimodal model call does both scene
perception AND decision (no separate Gemma-describes/Qwen-decides relay).
Originally a diagnostic-only comparison variant (built to answer "could one
capable VLM just do everything" empirically instead of by assumption) --
promoted to Kaggle-deployable 2026-09-06 once a CUDA-enabled llama-cpp-python
build + our own bundled qwen3.8-27B GGUF+mmproj made this model's ~170s/call
CPU latency (measured locally) drop to ~2-6s/call on the competition's real
RTX PRO 6000 GPU (see notebooks/gpu_bench/ and llm_relay_agent_experiments
memory) -- fast enough to actually use, unlike the CPU-only path this was
stuck behind before.

REPÈRE FR -- CE FICHIER NE CONTIENT AUCUN CODE DE DÉPLACEMENT.
`UnifiedVisionAgent` hérite de `VisionToolsAgent` (src/llm_tools_vision_agent.py)
et ne réécrit QUE `_consult_models` ci-dessous (comment on interroge le
modèle vision+décision). Toute la logique de déplacement -- boucle
`_choose_action_impl`, planification `_bfs_path`, apprentissage des lois de
mouvement -- vient sans modification de la classe parente. Si tu cherches un
bug de déplacement/boucle sur cet agent, va voir :
  - `VisionToolsAgent._choose_action_impl` dans src/llm_tools_vision_agent.py
  - `_bfs_path` / `_update_from_last_transition` dans src/llm_tools_agent.py
Ce fichier était dans un scratchpad temporaire (session Claude) -- copié ici
le 2026-09-05 pour que ce soit consultable/versionnable dans le repo."""
import queue
import threading

from llm_tools_agent import (
    OLLAMA_URL,
    _gpu_offload_available,
    _pending_brain_threads,
    _pending_brain_threads_lock,
    _structural_fact_lines,
)
from llm_tools_vision_agent import (
    VisionToolsAgent,
    _get_local_vision_llama,
    _grid_to_image_b64,
    _ollama_available,
    _vision_llama_lock,
)

UNIFIED_SYSTEM_PROMPT = """\
You are the full perception+decision module for a game-playing agent. You are shown \
the grid DIRECTLY as an image (each object labeled blob_N on the image itself), plus \
exact code-computed data about every distinct-colored object (bounding boxes, colors, \
sizes) and any structural facts -- the image and the object list describe the SAME \
thing: use the image to judge shape/layout/visual pattern, and the object list for exact \
positions, which is certain, not a guess.

You do not know what any landmark does until you've touched it or the effects log tells \
you. Prefer landmarks you haven't visited yet, unless the effects log suggests a specific \
landmark is useful to revisit.

If NO action shows a CONFIRMED movement effect (see movement laws below) or your own \
position hasn't been found yet, "move toward a landmark" is not a meaningful plan -- the \
real mechanic probably isn't about walking around a maze (it could be a selector/dial you \
cycle through, a button, or something the image hints at). In that case, suggest trying a \
specific ACTION directly instead of a landmark.

You keep a short, persistent scratchpad of your own best current hypothesis about how \
THIS SPECIFIC game works -- what the goal seems to be, which kind of landmark is worth \
prioritizing, anything that tripped you up -- carried forward across calls instead of \
being rederived from scratch each time. You MUST write a concrete, specific sentence \
describing what THIS game actually seems to be about, based on what you see (including \
the image) -- not a generic restatement of these instructions and not a placeholder. \
Revise it as you learn more; only repeat the exact same wording if it is still fully \
accurate and there is truly nothing new to add.

Respond in exactly three parts, each on its own line:
1. A short reasoning (1-2 sentences), referencing what you actually see in the image.
2. A line starting with "NOTES:" followed by your specific, concrete one-sentence \
hypothesis about this game -- refine or extend your previous notes, don't discard them \
unless they turned out wrong.
3. On the LAST line, write EITHER one landmark id (e.g. "blob_2") OR one available action \
name (e.g. "ACTION3") -- whichever you're recommending -- and nothing else.
"""


def _query_unified(image_b64: str, user_prompt: str, model_name: str, timeout_s: int) -> str:
    # Local dev only: Ollama-served qwen3.8. Kaggle (no Ollama, no internet)
    # uses _query_unified_gguf below instead -- see _query_unified_bounded.
    import requests
    resp = requests.post(OLLAMA_URL, json={
        "model": model_name,
        "messages": [
            {"role": "system", "content": UNIFIED_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt, "images": [image_b64]},
        ],
        "stream": False,
        "options": {"temperature": 0.3},
    }, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _query_unified_gguf(image_b64: str, user_prompt: str) -> str:
    """Kaggle GPU path: same bundled qwen3.8-27B GGUF+mmproj singleton as
    llm_tools_vision_agent._query_eyes_gguf (one shared Llama instance --
    perception and decision are the SAME model call here, so there's only ever
    one vision-capable model loaded regardless of which agent variant is
    running). GPU-only by the same reasoning as that function's docstring: a
    CPU run of this model measured ~170s/call locally, vs ~2-6s/call on the
    real RTX PRO 6000 (notebooks/gpu_bench/, 2026-09-06)."""
    llm, handler = _get_local_vision_llama()
    with _vision_llama_lock:
        resp = handler(
            llama=llm,
            messages=[
                {"role": "system", "content": UNIFIED_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ]},
            ],
            max_tokens=300,
            temperature=0.3,
            reasoning_effort="low",  # see llm_tools_vision_agent._query_eyes_gguf's comment --
                                      # measured as fast as (not slower than) "xhigh" on the real GPU
        )
    return resp["choices"][0]["message"]["content"]


def _query_unified_bounded(image_b64: str, user_prompt: str, model_name: str, timeout_s: int) -> str:
    """Dispatches to Ollama (local dev) or the bundled GGUF (Kaggle GPU),
    same pattern as llm_tools_vision_agent._query_eyes -- and reuses THAT
    module's pending-thread list/lock so one atexit drain hook (see
    llm_tools_agent._drain_pending_brain_threads) covers brain, eyes, and
    unified calls alike instead of a third copy of that machinery."""
    if _ollama_available():
        def call():
            return _query_unified(image_b64, user_prompt, model_name, timeout_s)
    else:
        if not _gpu_offload_available():
            raise RuntimeError("neither Ollama nor GPU offload available -- UnifiedVisionAgent "
                                "needs either a local Ollama+qwen3.8 (dev) or a real GPU for the "
                                "bundled qwen3.8 GGUF (Kaggle); CPU-only fallback deliberately not "
                                "attempted (see _query_unified_gguf docstring)")
        def call():
            return _query_unified_gguf(image_b64, user_prompt)

    outcome: queue.Queue = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            outcome.put(("ok", call()))
        except Exception as e:  # noqa: BLE001
            outcome.put(("err", e))

    t = threading.Thread(target=worker, daemon=True)
    with _pending_brain_threads_lock:
        _pending_brain_threads.append(t)
    t.start()
    try:
        status, payload = outcome.get(timeout=timeout_s + 5)
    except queue.Empty:
        raise TimeoutError(f"unified call exceeded {timeout_s}s") from None
    if status == "err":
        raise payload
    return payload


class UnifiedVisionAgent(VisionToolsAgent):
    UNIFIED_MODEL = "qwen3.8:latest"
    UNIFIED_TIMEOUT_S = 180

    def _consult_models(self, grid, blobs, legal_names, self_pos, counter_attr="brain_call_count"):
        id_map = {}
        blob_lines = []
        for i, b in enumerate(blobs):
            bid = f"blob_{i}"
            id_map[bid] = b
            visited = "visited" if (b["color"], b["bbox"][0], b["bbox"][2]) in self.visited_blob_keys else "unvisited"
            blob_lines.append(f"{bid}: color={b['color']} pos={b['centroid']} size={b['size']} ({visited})")
        blob_lines = blob_lines or ["(none detected)"]
        structural_lines = _structural_fact_lines(blobs)

        effects_lines = []
        for e in self.effects_log:
            if e.get("effect_type") == "resource_bar_grew":
                effects_lines.append(
                    f"- touching color {e['touched_color']} at {e['touched_pos']} GREW a resource/time "
                    f"bar (color {e['bar_color']}) from {e['fragments_before']} to {e['fragments_after']} "
                    f"segments")
            else:
                effects_lines.append(
                    f"- touching color {e['touched_color']} at {e['touched_pos']} changed region "
                    f"rows{e['effect_region'][0]}-{e['effect_region'][1]} cols{e['effect_region'][2]}-{e['effect_region'][3]}")
        effects_lines = effects_lines or ["(none discovered yet)"]

        position_line = (f"Your position: {self_pos}" if self_pos is not None else
                          "Your position: NOT YET FOUND -- no single object has been reliably "
                          "tracked as \"you\" despite trying multiple actions.")

        image_b64 = _grid_to_image_b64(grid, blobs)
        cross_game_hints = self._sync_shared_action_memory(legal_names)
        user_prompt = (
            f"{position_line}\n\n"
            f"Your notes from earlier turns:\n{self.brain_notes or '(none yet -- first consult this game)'}\n\n"
            f"Movement laws discovered so far:\n{self._movement_summary(legal_names)}\n\n"
            f"Hints from OTHER games played this session (same action vocabulary, "
            f"DIFFERENT game -- may not apply here, treat as a weak prior only):\n{cross_game_hints}\n\n"
            f"Landmarks visible now (also labeled blob_N on the image):\n" + "\n".join(blob_lines) + "\n\n" +
            (f"Structural facts computed exactly from positions (not guesses):\n"
             + "\n".join(structural_lines) + "\n\n" if structural_lines else "") +
            f"Effects discovered so far:\n" + "\n".join(effects_lines) + "\n\n"
            f"Available actions: {', '.join(legal_names)}\n"
            f"Which landmark should you move toward next, or which action should you try directly?"
        )
        setattr(self, counter_attr, getattr(self, counter_attr) + 1)
        try:
            reply = _query_unified_bounded(image_b64, user_prompt, self.UNIFIED_MODEL, self.UNIFIED_TIMEOUT_S)
        except Exception as e:
            reply = ""
            print(f"[UnifiedVisionAgent] query failed/timed out: {e!r}")
        self.last_raw_response = reply
        self._update_brain_notes(reply)

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
