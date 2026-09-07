"""Builds submission.ipynb for UnifiedVisionAgent (ONE qwen3.8-27B multimodal
call does both scene perception AND decision per consult -- see
src/llm_unified_vision_agent.py) on the competition's GPU accelerator
(RTX PRO 6000). Mirrors notebooks/submission_kernel_tools/build_notebook.py's
structure (gateway sidecar, two-phase submit, dummy-parquet fallback -- see
kaggle_submission_pipeline memory for why that structure exists), but inlines
the agent source directly from src/ at build time instead of keeping a second,
manually-synced copy like my_agent_tools.py -- src/llm_tools_agent.py,
src/llm_tools_vision_agent.py, and src/llm_unified_vision_agent.py are the
source of truth, this script is the only place that needs to know how to
flatten them into one Kaggle-writable file.
"""
import re

import nbformat as nbf


def _strip_future_import(src: str) -> str:
    # Only one `from __future__ import annotations` is allowed per file, and it
    # must be the first statement -- state_graph.py and each llm_*.py module
    # has its own; strip all of them here and re-add exactly one at the very
    # top of the concatenated bundle below.
    return re.sub(r"^from __future__ import annotations\n", "", src, count=1, flags=re.MULTILINE)


def _strip_internal_import(src: str, module_name: str) -> str:
    # llm_tools_vision_agent.py / llm_unified_vision_agent.py normally do
    # `from llm_tools_agent import (...)` / `from llm_tools_vision_agent import (...)`
    # -- once all files are concatenated into one module, those names are
    # already in scope, and there's no separate module on Kaggle's filesystem
    # for that import to resolve against.
    return re.sub(rf"from {module_name} import \([^)]*\)\n", "", src)


STATE_GRAPH_SOURCE = _strip_future_import(open("../../src/state_graph.py").read())
TOOLS_SOURCE = _strip_future_import(open("../../src/llm_tools_agent.py").read()).replace(
    "from state_graph import StateGraph, state_signature\n", ""
).replace(
    # same bump the plain-ToolsAgent bundle (my_agent_tools.py) applies for the
    # real submission -- src/'s MAX_ACTIONS=80 is the local-testing default
    "    MAX_ACTIONS = 80\n", "    MAX_ACTIONS = 150\n"
)
VISION_SOURCE = _strip_internal_import(
    _strip_future_import(open("../../src/llm_tools_vision_agent.py").read()), "llm_tools_agent"
)
UNIFIED_SOURCE = _strip_internal_import(
    _strip_internal_import(
        _strip_future_import(open("../../src/llm_unified_vision_agent.py").read()), "llm_tools_agent"
    ), "llm_tools_vision_agent"
)
# UnifiedVisionAgent -> MyAgent, same rename my_agent_tools.py does for plain
# ToolsAgent -- the ARC-AGI-3-Agents framework's agent registry expects a class
# named MyAgent (see agents/__init__.py rewrite below). VisionToolsAgent is
# kept as-is (a base class UnifiedVisionAgent inherits from, not the concrete
# agent this kernel runs).
UNIFIED_SOURCE = UNIFIED_SOURCE.replace(
    "class UnifiedVisionAgent(VisionToolsAgent):", "class MyAgent(VisionToolsAgent):"
)
AGENT_SOURCE = ("from __future__ import annotations\n\n" + STATE_GRAPH_SOURCE + "\n\n" +
                TOOLS_SOURCE + "\n\n" + VISION_SOURCE + "\n\n" + UNIFIED_SOURCE)

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# ARC Prize 2026 -- ARC-AGI-3 Submission (UnifiedVisionAgent, GPU)\n\n"
    "ToolsAgent's code-grounded self/wall/path tools, but perception AND "
    "decision are ONE qwen3.8-27B multimodal call per consult (no separate "
    "brain/eyes relay) -- see `src/llm_unified_vision_agent.py`. Runs locally "
    "via a CUDA-enabled `llama-cpp-python` build (dataset "
    "`loumitrmas/llama-cpp-python-cuda-blackwell`) fully offloaded to the "
    "competition's RTX PRO 6000 accelerator -- confirmed on real hardware "
    "(notebooks/gpu_bench/) to bring qwen3.8's ~170s/call CPU latency down to "
    "~2-6s/call. GPU-only by design (see `_query_unified_gguf`'s docstring): "
    "if the accelerator isn't recognized, the call raises rather than falling "
    "back to a 170s CPU call. Do not edit cells directly -- edit "
    "src/llm_tools_agent.py / src/llm_tools_vision_agent.py / "
    "src/llm_unified_vision_agent.py and rebuild."
))

cells.append(nbf.v4.new_code_cell(
    "# Mount paths for dataset/model sources vary run to run (see\n"
    "# feedback_verify_before_asserting) -- diagnose before assuming a fixed path.\n"
    "import glob\n"
    "print('--- /kaggle/input tree ---')\n"
    "for p in sorted(glob.glob('/kaggle/input/**/*', recursive=True))[:200]:\n"
    "    print(p)"
))

cells.append(nbf.v4.new_code_cell(
    "import subprocess\n"
    "print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)"
))

cells.append(nbf.v4.new_code_cell(
    "import glob, subprocess\n"
    "\n"
    "def pip_install_from_wheels(pkgs, name_hint):\n"
    "    candidates = set()\n"
    "    for pkg in pkgs:\n"
    "        for whl in glob.glob(f'/kaggle/input/**/{pkg.replace(\"-\", \"_\").replace(\"-\", \"*\")}*', recursive=True):\n"
    "            candidates.add('/'.join(whl.split('/')[:-1]))\n"
    "    for pattern in ['arc_agi_3_wheels', 'llama-cpp-python-cuda-blackwell', 'wheels']:\n"
    "        for d in glob.glob(f'/kaggle/input/**/{pattern}', recursive=True):\n"
    "            candidates.add(d)\n"
    "    print(f'{name_hint}: candidate wheel dirs =', candidates)\n"
    "    for d in candidates:\n"
    "        r = subprocess.run(['pip', 'install', '--no-index', '--find-links', d] + pkgs,\n"
    "                            capture_output=True, text=True)\n"
    "        print(r.stdout[-1500:], r.stderr[-1500:])\n"
    "        if r.returncode == 0:\n"
    "            print(f'{name_hint}: installed from {d}')\n"
    "            return\n"
    "    raise RuntimeError(f'{name_hint}: could not install {pkgs} from any candidate dir')\n"
    "\n"
    "pip_install_from_wheels(['arc-agi', 'python-dotenv'], 'competition wheels')\n"
    "pip_install_from_wheels(['llama-cpp-python', 'diskcache'], 'CUDA llama-cpp-python wheel')\n"
    "\n"
    "import llama_cpp\n"
    "print('llama_cpp version:', llama_cpp.__version__)\n"
    "print('supports_gpu_offload:', llama_cpp.llama_cpp.llama_supports_gpu_offload())"
))

cells.append(nbf.v4.new_code_cell(
    "# The GGUF+mmproj are served via a Kaggle Model source -- the agent locates\n"
    "# them itself by filename search under /kaggle/input (mount path/casing isn't\n"
    "# fully predictable across model_sources, see feedback_verify_before_asserting),\n"
    "# so just confirm they're visible here rather than copying them anywhere.\n"
    "from pathlib import Path\n"
    "for name in ['qwen3.8-27b.gguf', 'qwen3.8-27b-mmproj.gguf']:\n"
    "    matches = list(Path('/kaggle/input').rglob(name))\n"
    "    print(name, '->', matches if matches else 'NOT FOUND')"
))

cells.append(nbf.v4.new_code_cell(f"%%writefile /tmp/my_agent.py\n{AGENT_SOURCE}"))

cells.append(nbf.v4.new_code_cell(
    "import os, shutil, subprocess, sys\n"
    "from pathlib import Path\n\n"
    "def _find_under_input(name):\n"
    "    # /kaggle/input mount paths for the competition's own bundled dirs are NOT\n"
    "    # stable -- search by name instead of hardcoding one guessed path (same\n"
    "    # reasoning as notebooks/submission_kernel_tools/build_notebook.py).\n"
    "    for match in Path('/kaggle/input').rglob(name):\n"
    "        if match.is_dir():\n"
    "            return match\n"
    "    raise FileNotFoundError(f'{name} not found anywhere under /kaggle/input')\n\n"
    "def prepare_framework():\n"
    "    agents_wd = Path('/kaggle/working/ARC-AGI-3-Agents')\n"
    "    if agents_wd.exists():\n"
    "        shutil.rmtree(agents_wd)\n"
    "    shutil.copytree(_find_under_input('ARC-AGI-3-Agents'), agents_wd)\n"
    "    shutil.copyfile('/tmp/my_agent.py', agents_wd / 'agents' / 'templates' / 'my_agent.py')\n"
    "    # Register MyAgent in the framework's agent registry. We rewrite\n"
    "    # __init__.py because the upstream version eagerly imports\n"
    "    # templates with deps we don't ship (langgraph, smolagents, etc.).\n"
    "    (agents_wd / 'agents' / '__init__.py').write_text(\n"
    "        \"from typing import Type\\n\"\n"
    "        \"from dotenv import load_dotenv\\n\"\n"
    "        \"from .agent import Agent, Playback\\n\"\n"
    "        \"from .swarm import Swarm\\n\"\n"
    "        \"from .templates.random_agent import Random\\n\"\n"
    "        \"from .templates.my_agent import MyAgent\\n\\n\"\n"
    "        \"load_dotenv()\\n\\n\"\n"
    "        \"AVAILABLE_AGENTS: dict[str, Type[Agent]] = {\\n\"\n"
    "        \"    'random': Random,\\n\"\n"
    "        \"    'myagent': MyAgent,\\n\"\n"
    "        \"}\\n\"\n"
    "    )\n"
    "    return agents_wd\n\n"
    "def run_via_main(agents_wd):\n"
    "    # subprocess.run(check=True), deliberately NOT a `!shell` magic -- see\n"
    "    # notebooks/submission_kernel_tools/build_notebook.py's identical comment\n"
    "    # for why a silent `!cmd` failure was the leading hypothesis for earlier\n"
    "    # opaque 'Kaggle Error' results.\n"
    "    env = os.environ.copy()\n"
    "    env['MPLBACKEND'] = 'agg'\n"
    "    subprocess.run([sys.executable, 'main.py', '--agent', 'myagent'],\n"
    "                   cwd=str(agents_wd), check=True, env=env)\n\n"
    "if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):\n"
    "    subprocess.run(\n"
    "        ['curl', '--fail', '--retry', '999', '--retry-all-errors', '--retry-delay', '5',\n"
    "         '--retry-max-time', '600', 'http://gateway:8001/api/games'],\n"
    "        check=True,\n"
    "    )\n"
    "    agents_wd = prepare_framework()\n"
    "    (agents_wd / '.env').write_text(\n"
    "        \"SCHEME=http\\nHOST=gateway\\nPORT=8001\\nARC_API_KEY=test-key-123\\n\"\n"
    "        \"ARC_BASE_URL=http://gateway:8001/\\nOPERATION_MODE=online\\n\"\n"
    "        \"ENVIRONMENTS_DIR=\\nRECORDINGS_DIR=/kaggle/working/server_recording\\n\"\n"
    "    )\n"
    "    run_via_main(agents_wd)"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Local validation (Phase A only)\n\n"
    "Same rationale/technique as notebooks/submission_kernel_tools/build_notebook.py: "
    "actually plays the agent against the 25 bundled public games before ever "
    "spending Phase B quota, importing `MyAgent` directly against "
    "`arc_agi.Arcade` in OFFLINE mode (main.py always resolves `--game` against "
    "the online API regardless of OPERATION_MODE, and this kernel has no "
    "internet access)."
))

cells.append(nbf.v4.new_code_cell(
    "import os\n\n"
    "LOCAL_VALIDATION_MAX_ACTIONS = 60  # short budget -- this is a correctness/crash\n"
    "                                    # check, not a full play-through\n\n"
    "if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):\n"
    "    try:\n"
    "        import importlib.util\n"
    "        import time\n\n"
    "        import arc_agi\n\n"
    "        agents_wd = prepare_framework()\n"
    "        sys.path.insert(0, str(agents_wd))\n\n"
    "        spec = importlib.util.spec_from_file_location('my_agent_validation', '/tmp/my_agent.py')\n"
    "        module = importlib.util.module_from_spec(spec)\n"
    "        spec.loader.exec_module(module)\n"
    "        print(f'Local validation: imported {module.MyAgent.__name__} OK')\n\n"
    "        os.environ['OPERATION_MODE'] = 'offline'\n"
    "        os.environ['ENVIRONMENTS_DIR'] = str(_find_under_input('environment_files'))\n"
    "        arcade = arc_agi.Arcade()\n"
    "        games = [env_info.game_id for env_info in arcade.available_environments]\n"
    "        print(f'Local validation: {len(games)} public games found: {games}')\n\n"
    "        summary = []\n"
    "        for game_id in games:\n"
    "            env = arcade.make(game_id)\n"
    "            env.reset()\n"
    "            agent = module.MyAgent(\n"
    "                card_id='local-validation', game_id=game_id, agent_name='validation',\n"
    "                ROOT_URL='http://offline', record=False, arc_env=env, tags=[],\n"
    "            )\n"
    "            agent.MAX_ACTIONS = LOCAL_VALIDATION_MAX_ACTIONS\n"
    "            t0 = time.time()\n"
    "            agent.main()\n"
    "            summary.append((game_id, agent.action_counter, agent.frames[-1].levels_completed,\n"
    "                             round(time.time() - t0, 1)))\n"
    "            print(f'  {game_id}: actions={agent.action_counter} '\n"
    "                  f'levels_completed={agent.frames[-1].levels_completed} '\n"
    "                  f'elapsed={round(time.time() - t0, 1)}s')\n\n"
    "        print('--- local validation summary ---')\n"
    "        print(f'{len(summary)}/{len(games)} games completed without an uncaught exception')\n"
    "        print(f'total levels completed across all games: {sum(s[2] for s in summary)}')\n"
    "    except Exception:\n"
    "        import traceback\n"
    "        print('LOCAL VALIDATION FAILED -- fix this before spending Phase B quota. '\n"
    "              'Phase A commit will still proceed with a dummy parquet below.')\n"
    "        traceback.print_exc()"
))

cells.append(nbf.v4.new_code_cell(
    "from pathlib import Path\n\n"
    "if not Path('/kaggle/working/submission.parquet').exists():\n"
    "    import pandas as pd\n"
    "    submission = pd.DataFrame(\n"
    "        data=[['1_0', '1', True, 1]],\n"
    "        columns=['row_id', 'game_id', 'end_of_game', 'score'])\n"
    "    submission.to_parquet('/kaggle/working/submission.parquet', index=False)\n"
    "    submission.head()"
))

nb["cells"] = cells
with open("submission.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote submission.ipynb")
