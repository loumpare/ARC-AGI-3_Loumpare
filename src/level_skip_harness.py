"""Local-only testing helper: jump an agent straight to a specific level of a
game instead of making it play through the earlier ones first, to isolate/
debug behavior on a harder level without spending a whole run reaching it.

Mechanism (verified empirically 2026-09-06, not guessed): `arcengine`'s
`ARCBaseGame.set_level(index)` already exists and correctly switches which
level's sprites/logic are active, but `env.observation_space` (an
`arc_agi.LocalEnvironmentWrapper` property) is just a cached `_last_response`
-- only `reset()`/`step()` ever refresh it, so `set_level()` alone is
invisible to the agent until something re-renders and re-publishes a frame.
`jump_to_level` does exactly that render+publish step, reusing the SAME
`camera.render(...)` call `ARCBaseGame.perform_action` uses internally for
every real step -- not a reimplementation of frame rendering.

LOCAL TESTING ONLY. The real competition rerun always starts every game at
level 0 with no equivalent bypass -- this doesn't change reward/scoring
logic, just which level a local test observes from first (see
feedback_no_game_hacking memory: this is a test harness, not per-game reward
shaping in the agent itself).
"""
from __future__ import annotations

import time
from typing import Any

from arcengine import ActionInput, FrameDataRaw, GameAction


def jump_to_level(env: Any, level_index: int) -> None:
    """Mutates `env` (an already-`reset()` arc_agi LocalEnvironmentWrapper) so
    its NEXT `observation_space` read reflects `level_index` instead of
    wherever `reset()` left it. Call this after `env.reset()` and before
    handing the env to an agent -- agents read `arc_env.observation_space`
    fresh on every `choose_action` call (see `agents/agent.py`'s `main()`
    loop), so there's no need to touch the agent itself."""
    game = env._game
    game.set_level(level_index)
    frame_arr = game.camera.render(game.current_level.get_sprites())
    frame_raw = FrameDataRaw(
        game_id=game.game_id, state=game._state,
        levels_completed=game._score, win_levels=game.win_score,
        action_input=ActionInput(id=GameAction.RESET), full_reset=False,
        available_actions=game._available_actions,
    )
    frame_raw.frame = [frame_arr]  # `.frame` is a property backed by a PrivateAttr,
                                    # not a constructor field -- must be set after
                                    # construction (confirmed empirically: passing
                                    # frame=... to FrameDataRaw(...) is silently
                                    # dropped, no error, no exception)
    frame_raw.guid = env._guid
    frame_raw.game_id = env.environment_info.game_id
    env._set_last_response(frame_raw)


def play_from_level(agent_cls: type, game_id: str, level_index: int, arcade: Any,
                     max_actions: int = 150) -> Any:
    """Runs `agent_cls` on `game_id` starting from `level_index` (0-based)
    instead of level 0. Returns the finished agent instance -- same shape as
    calling `agent.main()` directly (`agent.frames[-1]` has the final frame,
    `agent.action_counter` the action count)."""
    env = arcade.make(game_id)
    env.reset()
    jump_to_level(env, level_index)
    agent = agent_cls(card_id="level-skip-test", game_id=game_id, agent_name="level-skip-test",
                       ROOT_URL="http://offline", record=False, arc_env=env, tags=[])
    agent.MAX_ACTIONS = max_actions
    t0 = time.time()
    agent.main()
    agent.level_skip_elapsed = round(time.time() - t0, 1)  # convenience, not used elsewhere
    return agent
