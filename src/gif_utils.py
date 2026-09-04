"""GIF recording utility for comparing agent variants -- renders each frame
with the standard ARC 16-color palette (same mapping used in
notebooks/simplified-submission-approach's ARC_COLORS) and assembles a GIF
via PIL (no imageio dependency needed)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

ARC_COLORS = {
    0: (0, 0, 0), 1: (0, 116, 217), 2: (255, 65, 54), 3: (46, 204, 64),
    4: (255, 220, 0), 5: (170, 170, 170), 6: (240, 18, 190), 7: (255, 133, 27),
    8: (127, 219, 255), 9: (135, 12, 37), 10: (57, 204, 204), 11: (177, 13, 201),
    12: (1, 255, 112), 13: (133, 20, 75), 14: (61, 153, 112), 15: (221, 221, 221),
}
UPSCALE = 8


def grid_to_image(grid: np.ndarray) -> Image.Image:
    h, w = grid.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for color, rgb_val in ARC_COLORS.items():
        rgb[grid == color] = rgb_val
    img = Image.fromarray(rgb, mode="RGB")
    return img.resize((w * UPSCALE, h * UPSCALE), Image.NEAREST)


def run_and_record(agent, out_gif_path: str, max_actions: int | None = None) -> dict:
    """Drives `agent.main()`'s exact loop manually (same pattern as
    Agent.main(), see agents/agent.py) so a frame image can be captured after
    every action, then writes the sequence as a GIF. Returns the same summary
    stats the rest of this session's test scripts have used throughout."""
    import time

    from arcengine import FrameData

    if max_actions is not None:
        agent.MAX_ACTIONS = max_actions

    frames_imgs = []
    agent.timer = time.time()
    latest = agent._convert_raw_frame_data(agent.arc_env.observation_space)
    frames_imgs.append(grid_to_image(np.array(latest.frame[0], dtype=int)))

    while not agent.is_done(agent.frames, latest) and agent.action_counter <= agent.MAX_ACTIONS:
        action = agent.choose_action(agent.frames, latest)
        frame = agent.take_action(action)
        if frame:
            agent.append_frame(frame)
            latest = frame
            frames_imgs.append(grid_to_image(np.array(latest.frame[0], dtype=int)))
        agent.action_counter += 1

    agent.cleanup()

    Path(out_gif_path).parent.mkdir(parents=True, exist_ok=True)
    frames_imgs[0].save(
        out_gif_path, save_all=True, append_images=frames_imgs[1:],
        duration=120, loop=0, optimize=True,
    )

    return dict(
        actions=agent.action_counter,
        levels=agent.frames[-1].levels_completed,
        state=agent.frames[-1].state.name,
        n_frames=len(frames_imgs),
        gif_path=out_gif_path,
    )
