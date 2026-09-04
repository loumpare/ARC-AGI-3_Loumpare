"""Builds submission.ipynb for the ToolsAgent (code-grounded self/wall/path
tools + Qwen2.5-7B run locally via llama-cpp-python from a bundled GGUF,
since the competition rerun has no internet/no Ollama server). Mirrors the
exact structure of notebooks/submission_kernel/submission.ipynb (the GRPO
agent's notebook) -- see kaggle_submission_pipeline memory for why this
structure exists (gateway sidecar, two-phase submit, dummy-parquet fallback).
"""
import re

import nbformat as nbf


def _strip_future_import(src: str) -> str:
    # Only one `from __future__ import annotations` is allowed per file, and it
    # must be the first statement -- both src/state_graph.py and
    # my_agent_tools.py have their own, so strip it here and re-add exactly
    # one at the very top of the concatenated bundle below.
    return re.sub(r"^from __future__ import annotations\n", "", src, count=1, flags=re.MULTILINE)


# my_agent_tools.py now does `from state_graph import StateGraph, state_signature`
# (src/llm_tools_agent.py 2026-09-04). Kaggle's kernel only gets this one file
# written via %%writefile below -- there's no second importable module on its
# filesystem -- so inline state_graph.py's source directly instead of shipping
# a separate file the import would fail to find.
STATE_GRAPH_SOURCE = _strip_future_import(open("../../src/state_graph.py").read())
AGENT_SOURCE_RAW = _strip_future_import(open("my_agent_tools.py").read()).replace(
    "from state_graph import StateGraph, state_signature\n", ""
)
AGENT_SOURCE = "from __future__ import annotations\n\n" + STATE_GRAPH_SOURCE + "\n\n" + AGENT_SOURCE_RAW

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# ARC Prize 2026 -- ARC-AGI-3 Submission (ToolsAgent)\n\n"
    "Code-grounded self/wall/path tools (matrix diffing, BFS pathfinding, a "
    "discovered-effects log) + Qwen2.5-7B for goal selection only -- see "
    "`src/llm_tools_agent.py`. Qwen runs locally via `llama-cpp-python` from a "
    "bundled GGUF (dataset `loumitrmas/arc-agi-3-qwen-tools-bundle`), since the "
    "competition rerun has no internet access and no local Ollama server. "
    "Do not edit cells directly -- edit the source file and rebuild."
))

cells.append(nbf.v4.new_code_cell(
    "# Mount paths have shifted since the last submission (verified empirically\n"
    "# this run, see feedback_verify_before_asserting) -- diagnose before\n"
    "# assuming a fixed path, then install from wherever the wheels actually are.\n"
    "import glob\n"
    "print('--- /kaggle/input tree ---')\n"
    "for p in sorted(glob.glob('/kaggle/input/**/*', recursive=True))[:200]:\n"
    "    print(p)"
))

cells.append(nbf.v4.new_code_cell(
    "import glob, subprocess\n"
    "\n"
    "def pip_install_from_wheels(pkgs, name_hint):\n"
    "    candidates = set()\n"
    "    for pkg in pkgs:\n"
    "        for whl in glob.glob(f'/kaggle/input/**/{pkg.replace(\"-\", \"_\").replace(\"-\", \"*\")}*', recursive=True):\n"
    "            candidates.add('/'.join(whl.split('/')[:-1]))\n"
    "    for pattern in ['arc_agi_3_wheels', 'arc-agi-3-qwen-tools-bundle/wheels', 'wheels']:\n"
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
    "pip_install_from_wheels(['llama-cpp-python', 'diskcache'], 'qwen tools bundle wheels')"
))

cells.append(nbf.v4.new_code_cell(
    "# GGUF is served via a Kaggle Model source -- the agent locates it itself\n"
    "# by filename search under /kaggle/input (mount path/casing isn't fully\n"
    "# predictable across model_sources, see feedback_verify_before_asserting),\n"
    "# so just confirm it's visible here rather than copying it anywhere.\n"
    "from pathlib import Path\n"
    "matches = list(Path(\"/kaggle/input\").rglob(\"qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf\"))\n"
    "print(\"GGUF found at:\", matches if matches else \"NOT FOUND\")"
))

cells.append(nbf.v4.new_code_cell(f"%%writefile /tmp/my_agent.py\n{AGENT_SOURCE}"))

cells.append(nbf.v4.new_code_cell(
    "import os, shutil, subprocess, sys\n"
    "from pathlib import Path\n\n"
    "def _find_under_input(name):\n"
    "    # /kaggle/input mount paths for the competition's own bundled dirs are NOT\n"
    "    # stable -- confirmed by a real Phase A run finding no\n"
    "    # /kaggle/input/competitions/<slug>/ARC-AGI-3-Agents (the nested path the\n"
    "    # official starter notebook hardcodes) even though the SAME directory name\n"
    "    # existed elsewhere under /kaggle/input. A `!shell` magic silently ignoring\n"
    "    # that FileNotFoundError (see run_via_main's docstring) is the leading\n"
    "    # hypothesis for all three of our unexplained \"Kaggle Error\" results --\n"
    "    # search by name instead of hardcoding one guessed path, matching how\n"
    "    # the wheel-install and GGUF-discovery steps already do this defensively.\n"
    "    for match in Path('/kaggle/input').rglob(name):\n"
    "        if match.is_dir():\n"
    "            return match\n"
    "    raise FileNotFoundError(f'{name} not found anywhere under /kaggle/input')\n\n"
    "def prepare_framework():\n"
    "    # Copy the framework into a writable location.\n"
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
    "    # subprocess.run(check=True), deliberately NOT a `!shell` magic -- a `!cmd`\n"
    "    # cell in Jupyter does not check the exit code or raise on failure, so a\n"
    "    # crash inside main.py was previously entirely invisible: the notebook would\n"
    "    # just silently continue to the next cell with no Python exception at all,\n"
    "    # indistinguishable from Kaggle's own opaque \"Kaggle Error\" (which is what\n"
    "    # all three of our real submissions got, with zero traceback surfaced --\n"
    "    # see llm_relay_agent_experiments memory). subprocess.run(check=True) turns\n"
    "    # any such failure into a real, visible \"Notebook Threw Exception\" instead.\n"
    "    # Pattern confirmed working in a public reference submission (mbmmurad's\n"
    "    # 3rd-place-candidate milestone notebook uses the identical approach).\n"
    "    env = os.environ.copy()\n"
    "    env['MPLBACKEND'] = 'agg'\n"
    "    subprocess.run([sys.executable, 'main.py', '--agent', 'myagent'],\n"
    "                   cwd=str(agents_wd), check=True, env=env)\n\n"
    "if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):\n"
    "    # Wait for the gateway sidecar to be ready -- also subprocess.run(check=True)\n"
    "    # now, for the same reason as above (a failed `!curl` would otherwise be silent).\n"
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
    "    # Run it. The gateway records every action and emits submission.parquet.\n"
    "    run_via_main(agents_wd)"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Local validation (Phase A only)\n\n"
    "Actually PLAYS the agent against the competition's 25 bundled public games "
    "before ever spending the daily Phase B quota -- previously Phase A only ever "
    "wrote a dummy parquet and never ran the agent at all, so a real bug (or the "
    "GGUF/Qwen path) was never exercised until the graded rerun, which gives no "
    "traceback on failure. Matches the \"verify basic fundamentals with a "
    "submission\" / \"build sanity checks into your pipeline\" advice from Kaggle's "
    "own code-competition debugging guide, and the pattern used by public reference "
    "notebooks (jeroencottaar/simplified-submission-approach; mbmmurad's "
    "3rd-place-candidate milestone's `RUN_ARC_LOCAL_VALIDATION` path).\n\n"
    "`main.py` cannot be reused directly here: it always resolves `--game` against "
    "the ONLINE `/api/games` endpoint regardless of `OPERATION_MODE` (documented "
    "gotcha, see GUIDELINE.md) and this kernel has no internet access, so it would "
    "just fail immediately with a 401 before ever touching our agent. Instead, "
    "import `MyAgent` directly and drive it against `arc_agi.Arcade` in OFFLINE "
    "mode ourselves -- the same technique `jeroencottaar/simplified-submission-approach` "
    "and the commit-mode guardrail in mbmmurad's notebook both use to sidestep this."
))

cells.append(nbf.v4.new_code_cell(
    "import os\n\n"
    "LOCAL_VALIDATION_MAX_ACTIONS = 60  # short budget -- this is a correctness/crash\n"
    "                                    # check, not a full 150-action play-through\n\n"
    "if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):\n"
    "    # Wrapped in try/except so a real bug here is LOUD (full traceback below)\n"
    "    # but still lets Phase A's commit finish (writes the dummy parquet next).\n"
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
    "    # Save-and-run-all (commit) mode: emit a dummy submission so the\n"
    "    # commit succeeds. The real submission.parquet is produced by the\n"
    "    # gateway during competition rerun.\n"
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
