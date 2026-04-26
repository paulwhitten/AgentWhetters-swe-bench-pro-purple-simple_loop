"""
SWE-bench Purple Agent — simple single-loop solver.

Architecture: mini-SWE-agent style.  One flat agent loop using the native
OpenAI shell tool (local mode) + a done function tool.  The LLM decides
when to read, grep, edit, and test.  No phases, no phase budgets.

Receives:
  - A JSON message from the green agent containing:
    instance_id, problem_statement, docker_image, base_commit, repo, hints

Usage:
  uv run src/purple/server.py --host 127.0.0.1 --port 9022
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import os
import re
import sys
import textwrap
import time
from pathlib import Path

import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Artifact,
    Part,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)
from uuid import uuid4
from a2a.utils import new_agent_text_message
from openai import AsyncOpenAI, AsyncAzureOpenAI

# Ensure the package root (src/) is on sys.path
_src_dir = str(Path(__file__).resolve().parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from purple.docker_runner import DockerRunner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STEP_LIMIT = 50               # Global step limit
QA_BUDGET = 15                # Extra steps for post-done test-gate fix phase
TOOL_RESULT_LIMIT = 30_000    # Max characters per tool result
COMMAND_TIMEOUT = 300          # Per-command timeout in seconds
TEST_FAILURE_EXTRACT_LIMIT = 3000
LOG_DIR = Path("logs")
COMPACT_THRESHOLD = 200_000   # Server-side compaction threshold (tokens)

# ---------------------------------------------------------------------------
# OpenAI client factory
# ---------------------------------------------------------------------------

def _make_openai_client(api_key: str, base_url: str | None = None) -> AsyncOpenAI:
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    if azure_endpoint:
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
        return AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


# ---------------------------------------------------------------------------
# Model classification
# ---------------------------------------------------------------------------

_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")

def _is_reasoning_model(model_name: str) -> bool:
    """Return True if *model_name* is a reasoning model."""
    return any(model_name.startswith(p) for p in _REASONING_MODEL_PREFIXES)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

# --- Reasoning model prompt (GPT-5.x) ---
SYSTEM_PROMPT_REASONING = textwrap.dedent("""\
    You are an expert software engineer. Solve the coding task in the problem
    statement correctly and efficiently.

    You have a LIMITED budget of {step_limit} shell calls (each assistant response
    that runs a command = 1 call). Each call costs real money. Your goal is to
    solve the problem correctly while using as few calls as possible. A good
    solution typically needs 8-12 calls.

    <grading>
    After you submit, a rigorous hidden test patch will be applied that adds
    targeted tests for every requirement — including edge cases (null,
    undefined, empty arrays, falsy values, missing keys, exact return types).
    Your code must handle all of these correctly. This is what you are graded
    on — not cosmetic improvements, not documentation, not peripheral cleanup.

    Write code as if it is safety-critical. Every new function must be hardened
    against: null/undefined/falsy inputs, empty arrays, missing object keys,
    wrong types, and must preserve exact ordering and return types. If a
    function takes an array, it must handle [], null, false, undefined. If it
    returns an array, element order must match input order, with null for
    missing values.

    PRIORITIZE: requirements implementation and robust edge-case handling FIRST.
    Only address secondary concerns (schemas, docs, linting) if you have calls
    to spare after the core requirements and their edge cases are solid.
    </grading>

    <editing>
    When using Python string .replace() to edit files, the old string must
    match EXACTLY or the replacement silently does nothing. After every edit,
    VERIFY it landed by piping to grep or diff. If your edit didn't apply,
    diagnose why and fix it immediately.
    Make MINIMAL changes. Change only the lines you need. Do NOT rewrite entire
    functions when a one-line fix suffices. Smaller diffs are more reliable.
    </editing>

    <efficiency>
    Minimize calls by batching work. Examples:

    Batched read (1 call to read 4 files):
      sed -n '1,80p' src/user/email.js && printf '\\n===FILE2===\\n' && \\
      sed -n '1,120p' src/api/users.js && printf '\\n===FILE3===\\n' && \\
      sed -n '400,500p' src/database/redis/hash.js && printf '\\n===FILE4===\\n' && \\
      grep -rn 'canSendValidation' src/

    Batched edit + verify (1 call to edit 2 files and confirm):
      python <<'PY'
      import pathlib
      f1 = pathlib.Path('src/user/email.js')
      f1.write_text(f1.read_text().replace('OLD1', 'NEW1'))
      f2 = pathlib.Path('src/api/users.js')
      f2.write_text(f2.read_text().replace('OLD2', 'NEW2'))
      PY
      grep -n 'NEW1' src/user/email.js && grep -n 'NEW2' src/api/users.js

    General rules:
    - Batch file reads into one call using sed/cat/grep with printf separators
    - Batch multiple edits into one python heredoc, then verify each
    - Run the full test suite at most 1-2 times total
    - Quick behavior checks with node -e '...' or python -c '...'
    </efficiency>

    <validation>
    Before calling done, review your changes in your reasoning:
    - Did you address ALL requirements in the problem statement?
    - For every new function: does it handle null, undefined, [], falsy inputs?
    - For every function returning an array: is element order preserved?
    - Did your test run pass? If not, fix failures before calling done.
    - Are your edits minimal? No unnecessary rewrites or unrelated changes?
    If any check fails, fix the issue first. Only call done when confident.
    </validation>

    <rules>
    - Read a file before modifying it
    - Do NOT modify test files unless required
    - After EVERY file edit, verify the change landed (grep, diff, or sed -n)
    - Call done when finished
    </rules>
""")

# --- Non-reasoning model prompt (gpt-4o-mini, gpt-4o, gpt-4.1-mini) ---
SYSTEM_PROMPT_CLASSIC = textwrap.dedent("""\
    You are an expert software engineer. Your task: fix a bug or implement a
    feature described in the problem statement below.

    <critical_rules>
    THESE RULES ARE MANDATORY — violations cause grading failures:
    1. Read a file BEFORE modifying it. Never guess file contents.
    2. After EVERY edit, verify it landed: grep -n 'expected_text' file
    3. Do NOT modify test files unless the problem explicitly requires it.
    4. Make MINIMAL changes. One-line fix > function rewrite.
    5. Call done when finished — do not keep exploring after fixing.
    </critical_rules>

    <workflow>
    Follow these steps IN ORDER. Before each action, write a brief
    <thought> explaining what you will do and why.

    Step 1 — UNDERSTAND: Read the problem statement carefully. Count every
             distinct requirement. Identify which files are likely involved.
    Step 2 — EXPLORE: Use run_command to find and read the relevant files.
             Batch reads: cat file1 && echo '====' && cat file2
    Step 3 — DIAGNOSE: In a <thought> block, reason step-by-step about:
             - What is the root cause or core requirement?
             - Which function(s) need to change?
             - What edge cases does the problem imply?
    Step 4 — EDIT: Make the minimal code changes. Use python heredocs for
             precise edits:
               python3 <<'PY'
               import pathlib
               p = pathlib.Path('src/file.js')
               p.write_text(p.read_text().replace('OLD', 'NEW'))
               PY
    Step 5 — VERIFY: Confirm each edit landed with grep or cat.
    Step 6 — TEST: Run the test suite (at most 1-2 times total).
    Step 7 — FIX: If tests fail, read the error, diagnose, and fix.
             Repeat Steps 4-6 until tests pass.
    Step 8 — Call done with a brief summary of what you changed.
    </workflow>

    <grading>
    After you submit, a hidden test patch adds targeted tests for every
    requirement — edge cases included (null, undefined, empty arrays, falsy
    values, missing keys, exact return types). Your code must handle ALL of
    these.

    Every new function must be hardened against: null/undefined/falsy inputs,
    empty arrays, missing object keys, wrong types. If it returns an array,
    element order must match input order with null for missing values.
    </grading>

    <efficiency>
    You have {step_limit} tool calls. A good solution needs 8-15 calls.
    - Batch file reads: cat f1 && echo '====' && cat f2
    - Batch edits into one python heredoc, then verify
    - Run the full test suite at most 1-2 times
    </efficiency>

    <self_check>
    Before calling done, verify in a <thought> block:
    - Did you address ALL requirements from the problem statement?
    - Did your edits actually apply? (verified with grep)
    - For new functions: do they handle null, undefined, [], falsy inputs?
    - Did tests pass? If not, fix before calling done.
    </self_check>
""")


def _get_system_prompt(model_name: str) -> str:
    """Return the appropriate system prompt template for the model."""
    if _is_reasoning_model(model_name):
        return SYSTEM_PROMPT_REASONING
    return SYSTEM_PROMPT_CLASSIC

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

# Native shell tool — only works with reasoning models (GPT-5.x)
SHELL_TOOL: dict = {
    "type": "shell",
    "environment": {"type": "local"},
}

# Function-based run_command — for non-reasoning models (gpt-4o-mini, etc.)
RUN_COMMAND_TOOL: dict = {
    "type": "function",
    "name": "run_command",
    "description": (
        "Execute a shell command in the repository directory. "
        "Returns stdout, stderr, and exit code. Use for reading files, "
        "searching, editing, running tests, and any other shell operations."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute (bash).",
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    },
    "strict": True,
}

DONE_TOOL: dict = {
    "type": "function",
    "name": "done",
    "description": "Signal that the fix is complete. Call after all changes are made and verified.",
    "parameters": {
        "type": "object",
        "properties": {
            "explanation": {
                "type": "string",
                "description": "Brief summary of what was changed and why.",
            },
        },
        "required": ["explanation"],
        "additionalProperties": False,
    },
    "strict": True,
}

# Tool sets per model type
TOOLS_REASONING: list[dict] = [SHELL_TOOL, DONE_TOOL]
TOOLS_CLASSIC: list[dict] = [RUN_COMMAND_TOOL, DONE_TOOL]


def _get_tools(model_name: str) -> list[dict]:
    """Return the appropriate tool set for the model."""
    if _is_reasoning_model(model_name):
        return TOOLS_REASONING
    return TOOLS_CLASSIC

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_instance_message(text: str) -> dict:
    """Parse the JSON payload sent by the green agent."""
    return json.loads(text)


def _start_required_services(runner: DockerRunner) -> list[str]:
    """Detect and start background services the test suite needs."""
    started: list[str] = []

    # --- Redis ---
    need_redis = False
    for cfg in ("config.json", "docker-compose.yml", "docker-compose.yaml"):
        r = runner.run(f"grep -qi redis {cfg} 2>/dev/null && echo yes")
        if "yes" in r.output:
            need_redis = True
            break
    if not need_redis:
        r = runner.run("grep -qi redis package.json 2>/dev/null && echo yes")
        if "yes" in r.output:
            need_redis = True
    if need_redis:
        r = runner.run("which redis-server 2>/dev/null")
        if r.exit_code == 0:
            r = runner.run("redis-cli ping 2>/dev/null")
            if r.exit_code != 0 or "PONG" not in r.output:
                runner.run("redis-server --daemonize yes --protected-mode no --appendonly yes")
                runner.run("for i in 1 2 3 4 5; do redis-cli ping 2>/dev/null | grep -q PONG && break; done")
                started.append("Started redis-server")

    # --- MongoDB ---
    need_mongo = False
    for cfg in ("config.json", "docker-compose.yml", "docker-compose.yaml"):
        r = runner.run(f"grep -qi mongo {cfg} 2>/dev/null && echo yes")
        if "yes" in r.output:
            need_mongo = True
            break
    if need_mongo:
        r = runner.run("which mongod 2>/dev/null")
        if r.exit_code == 0:
            r = runner.run("pgrep mongod")
            if r.exit_code != 0:
                runner.run("mongod --fork --logpath /tmp/mongod.log --dbpath /data/db 2>/dev/null || mkdir -p /data/db && mongod --fork --logpath /tmp/mongod.log --dbpath /data/db")
                started.append("Started mongod")

    # --- PostgreSQL ---
    need_pg = False
    for cfg in ("config.json", "docker-compose.yml", "docker-compose.yaml"):
        r = runner.run(f"grep -qi postgres {cfg} 2>/dev/null && echo yes")
        if "yes" in r.output:
            need_pg = True
            break
    if need_pg:
        r = runner.run("which pg_isready 2>/dev/null")
        if r.exit_code == 0:
            r = runner.run("pg_isready")
            if r.exit_code != 0:
                runner.run("su - postgres -c 'pg_ctl start -D /var/lib/postgresql/data -l /tmp/pg.log' 2>/dev/null || pg_ctlcluster 14 main start 2>/dev/null")
                started.append("Started PostgreSQL")

    return started


def _discover_test_command(runner: DockerRunner) -> str | None:
    """Probe the container to find a working test command."""
    # Node.js
    r = runner.run("cat package.json 2>/dev/null")
    if r.exit_code == 0 and r.output.strip():
        try:
            pkg = json.loads(r.output)
            scripts = pkg.get("scripts", {})
            if "test" in scripts:
                return "npm test"
            if "check" in scripts:
                return "npm run check"
        except json.JSONDecodeError:
            pass

    # Python
    r = runner.run("test -f pytest.ini -o -f setup.cfg -o -f pyproject.toml && echo found")
    if r.exit_code == 0 and "found" in r.output:
        r2 = runner.run("python -m pytest --version 2>/dev/null")
        if r2.exit_code == 0:
            return "python -m pytest -x --tb=short -q"
        r2 = runner.run("python -m unittest discover --help 2>/dev/null")
        if r2.exit_code == 0:
            return "python -m unittest discover -s tests"

    # Go
    r = runner.run("test -f go.mod && echo found")
    if r.exit_code == 0 and "found" in r.output:
        return "go test ./..."

    # Makefile
    r = runner.run("grep -q '^test:' Makefile 2>/dev/null && echo found")
    if r.exit_code == 0 and "found" in r.output:
        return "make test"

    # Rust
    r = runner.run("test -f Cargo.toml && echo found")
    if r.exit_code == 0 and "found" in r.output:
        return "cargo test"

    # Ruby
    r = runner.run("test -f Gemfile && grep -q 'rspec\\|minitest' Gemfile 2>/dev/null && echo found")
    if r.exit_code == 0 and "found" in r.output:
        return "bundle exec rake test"

    return None


def _extract_test_failures(output: str) -> str:
    """Extract a focused failure summary from test runner output."""
    lines = output.splitlines()
    failures: list[str] = []
    passing = 0
    failing = 0

    in_failure_block = False
    current_failure: list[str] = []
    for line in lines:
        stripped = line.strip()
        if "passing" in stripped and not stripped.startswith("#"):
            try:
                passing = int(stripped.split()[0])
            except (ValueError, IndexError):
                pass
        if "failing" in stripped and not stripped.startswith("#"):
            try:
                failing = int(stripped.split()[0])
            except (ValueError, IndexError):
                pass
        if re.match(r"^\s+\d+\)", line):
            if current_failure:
                failures.append("\n".join(current_failure))
            current_failure = [stripped]
            in_failure_block = True
        elif in_failure_block:
            if stripped.startswith("at ") or stripped.startswith("Error:") or stripped.startswith("AssertionError"):
                current_failure.append("  " + stripped)
            elif stripped == "" or re.match(r"^\s+\d+\)", stripped):
                if current_failure:
                    failures.append("\n".join(current_failure))
                    current_failure = []
                in_failure_block = stripped != ""
            elif "expected" in stripped.lower() or "actual" in stripped.lower() or "assert" in stripped.lower():
                current_failure.append("  " + stripped)

    if current_failure:
        failures.append("\n".join(current_failure))

    for line in lines:
        if "FAILED" in line and "::" in line:
            failures.append(line.strip())

    if not failures and not failing:
        return output

    parts = [f"=== TEST SUMMARY: {passing} passing, {failing} failing ==="]
    if failures:
        parts.append("FAILING TESTS:")
        for f in failures[:20]:
            parts.append(f)
    parts.append("=== END TEST SUMMARY ===")
    parts.append("")
    summary = "\n".join(parts)

    remaining = TEST_FAILURE_EXTRACT_LIMIT - len(summary)
    if remaining > 200:
        summary += output[:remaining]
    return summary


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    # Keep head + tail so test results at the end aren't lost
    head = limit * 2 // 3
    tail = limit - head
    return text[:head] + f"\n... (truncated, {len(text)} total chars) ...\n" + text[-tail:]


def _execute_shell(runner: DockerRunner, commands: list[str]) -> list[dict]:
    """Execute shell commands inside the Docker container.

    Returns a list of result dicts in shell_call_output format:
    [{"stdout": ..., "stderr": ..., "outcome": {"type": "exit", "exit_code": N}}]
    """
    results = []
    for command in commands:
        try:
            r = runner.run(command, timeout=COMMAND_TIMEOUT)
            stdout = r.output or ""
            stderr = ""
            # DockerRunner combines stdout+stderr; split on common patterns
            # The runner returns combined output, so put it all in stdout
            exit_code = r.exit_code

            # Auto-recover: if a service is down, try to restart and retry
            if exit_code != 0 and ("ECONNREFUSED" in stdout or "Connection refused" in stdout):
                recovery_attempted = False
                if "6379" in stdout or "redis" in stdout.lower():
                    runner.run("redis-server --daemonize yes --protected-mode no --appendonly yes")
                    runner.run("for i in 1 2 3 4 5; do redis-cli ping 2>/dev/null | grep -q PONG && break; done")
                    recovery_attempted = True
                elif "27017" in stdout or "mongo" in stdout.lower():
                    runner.run("mongod --fork --logpath /tmp/mongod.log --dbpath /data/db 2>/dev/null || mkdir -p /data/db && mongod --fork --logpath /tmp/mongod.log --dbpath /data/db")
                    recovery_attempted = True
                elif "5432" in stdout or "postgres" in stdout.lower():
                    runner.run("su - postgres -c 'pg_ctl start -D /var/lib/postgresql/data -l /tmp/pg.log' 2>/dev/null || pg_ctlcluster 14 main start 2>/dev/null")
                    recovery_attempted = True
                if recovery_attempted:
                    r = runner.run(command, timeout=COMMAND_TIMEOUT)
                    stdout = r.output or ""
                    exit_code = r.exit_code

            results.append({
                "stdout": _truncate(stdout, TOOL_RESULT_LIMIT),
                "stderr": _truncate(stderr, TOOL_RESULT_LIMIT),
                "outcome": {"type": "exit", "exit_code": exit_code},
            })
        except Exception as exc:
            results.append({
                "stdout": "",
                "stderr": f"Error: {exc}",
                "outcome": {"type": "timeout"},
            })
    return results


# ---------------------------------------------------------------------------
# Conversation logger
# ---------------------------------------------------------------------------

class ConversationLogger:
    """Logs each turn of the agentic loop to a JSONL file."""

    def __init__(self, instance_id: str):
        run_ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_id = re.sub(r"[^\w.-]", "_", instance_id)[:80]
        self._dir = LOG_DIR / run_ts
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"{safe_id}.jsonl"
        self._fh = self._path.open("a", encoding="utf-8")

    def log(self, event: str, **data) -> None:
        entry = {
            "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
            "event": event,
            **data,
        }
        self._fh.write(json.dumps(entry, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    @property
    def path(self) -> Path:
        return self._path


# ---------------------------------------------------------------------------
# Core solve logic — single flat loop
# ---------------------------------------------------------------------------

async def solve_instance(
    instance: dict,
    client: AsyncOpenAI,
    model: str,
    on_status: callable | None = None,
) -> str:
    """Solve a SWE-bench instance using a single flat agent loop.

    The LLM gets a bash shell and decides what to do — read files, grep,
    edit with sed/cat, run tests — all in one loop.  Returns the final
    git diff.
    """
    instance_id = instance["instance_id"]
    problem = instance["problem_statement"]
    image_uri = instance["docker_image"]
    base_commit = instance["base_commit"]
    repo = instance.get("repo", "")
    hints = instance.get("hints", "")
    is_reasoning = _is_reasoning_model(model)

    async def status(msg: str) -> None:
        if on_status:
            await on_status(msg)
        logger.info("[%s] %s", instance_id, msg)

    await status("Pulling image and starting container...")
    loop = asyncio.get_event_loop()
    runner = DockerRunner(image_uri, base_commit)
    clog = ConversationLogger(instance_id)

    try:
        await loop.run_in_executor(None, runner.start)

        # Start required services
        svc_msgs = await loop.run_in_executor(None, lambda: _start_required_services(runner))
        for svc_msg in svc_msgs:
            await status(svc_msg)

        # Get repo overview
        await status("Exploring repository structure...")
        tree = await loop.run_in_executor(
            None, lambda: runner.list_files(".", max_depth=2)
        )
        tree = _truncate(tree, 10_000)

        # Discover test commands
        test_cmd = await loop.run_in_executor(
            None, lambda: _discover_test_command(runner)
        )
        if test_cmd:
            await status(f"Detected test command: {test_cmd}")

        # Run baseline tests to capture current failures (skip if all pass)
        baseline_failures = ""
        if test_cmd:
            await status("Running baseline tests...")
            baseline_result = await loop.run_in_executor(
                None, lambda: runner.run(test_cmd, timeout=COMMAND_TIMEOUT)
            )
            if baseline_result.exit_code != 0:
                baseline_failures = _extract_test_failures(baseline_result.output)
                await status(f"Baseline tests found failures (exit code {baseline_result.exit_code})")
            else:
                await status("Baseline tests all pass — skipping injection")

        # Build the initial user message
        user_content = f"## Repository: {repo}\n\n"
        user_content += f"## File listing (depth 2):\n```\n{tree}\n```\n\n"
        user_content += f"## Problem statement:\n{_truncate(problem, 8000)}\n"
        if hints:
            user_content += f"\n## Hints:\n{_truncate(hints, 2000)}\n"
        if test_cmd:
            user_content += f"\n## Test command:\n```\n{test_cmd}\n```\n"
        if baseline_failures:
            user_content += (
                f"\n## Baseline test failures (before any changes):\n"
                f"```\n{_truncate(baseline_failures, 4000)}\n```\n"
                f"\nThese are the tests you must make pass. Fix these specific failures.\n"
            )
        user_content += (
            "\nSolve this problem. Read relevant files, understand the root cause, "
            "make the necessary code changes, and verify with tests."
        )

        # Initialize conversation
        system_prompt = _get_system_prompt(model).format(step_limit=STEP_LIMIT)
        tools = _get_tools(model)
        items: list = [{"role": "user", "content": user_content}]
        clog.log("system", content=system_prompt)
        clog.log("user", content=user_content)

        # -----------------------------------------------------------
        # SINGLE FLAT LOOP
        # -----------------------------------------------------------
        done_signalled = False
        qa_gate_failed = False          # True when done was rejected by test gate
        qa_steps_used = 0               # Steps consumed in QA fix phase
        total_steps = 0                 # Tracks total steps across main + QA
        step = 0

        for step in range(STEP_LIMIT):
            await status(f"Step {step + 1}")

            # Build API request
            api_kwargs: dict = {
                "model": model,
                "instructions": system_prompt,
                "input": items,
                "tools": tools,
                "parallel_tool_calls": False,
                "store": False,
            }
            if is_reasoning:
                api_kwargs["include"] = ["reasoning.encrypted_content"]
                api_kwargs["context_management"] = [
                    {"type": "compaction", "compact_threshold": COMPACT_THRESHOLD},
                ]
                api_kwargs["reasoning"] = {
                    "effort": "high",
                    "summary": "auto",
                }
                api_kwargs["max_output_tokens"] = 16_000
            else:
                api_kwargs["temperature"] = 0.0
                api_kwargs["max_output_tokens"] = 4096

            response = await client.responses.create(**api_kwargs)
            usage = response.usage

            # Parse response
            shell_calls = []
            function_calls = []
            text_content = None
            for item in response.output:
                if item.type == "shell_call":
                    shell_calls.append(item)
                elif item.type == "function_call":
                    function_calls.append(item)
                elif item.type == "message":
                    for part in (item.content or []):
                        if hasattr(part, "text"):
                            text_content = part.text

            # Append all output items (including reasoning + compaction)
            items.extend(response.output)

            # Drop items before the latest compaction item to save tokens
            last_compaction_idx = None
            for i, it in enumerate(items):
                if hasattr(it, "type") and it.type == "compaction":
                    last_compaction_idx = i
            if last_compaction_idx is not None and last_compaction_idx > 0:
                items = items[last_compaction_idx:]
                clog.log("compaction", dropped_before=last_compaction_idx)

            has_tool_calls = bool(shell_calls or function_calls)
            clog.log(
                "assistant",
                step=step,
                content=text_content,
                shell_calls=len(shell_calls),
                function_calls=[fc.name for fc in function_calls],
                usage={
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                } if usage else None,
            )

            # No tool calls = model is done (text-only response)
            if not has_tool_calls:
                logger.info("[%s] No tool calls at step %d — ending", instance_id, step)
                break

            # Process shell_call items (native shell tool)
            for sc in shell_calls:
                commands = []
                if hasattr(sc, "action") and sc.action:
                    commands = sc.action.commands if hasattr(sc.action, "commands") else []
                if not commands:
                    commands = ["echo '(no command)'"]

                preview = commands[0][:80] if commands else ""
                await status(f"[{step + 1}] $ {preview}{'...' if len(commands[0]) > 80 else ''}")

                results = await loop.run_in_executor(
                    None, lambda cmds=commands: _execute_shell(runner, cmds),
                )

                max_output_length = TOOL_RESULT_LIMIT
                if hasattr(sc, "action") and hasattr(sc.action, "max_output_length") and sc.action.max_output_length:
                    max_output_length = sc.action.max_output_length

                is_error = any(r["outcome"].get("exit_code", 0) != 0 for r in results)

                clog.log(
                    "tool",
                    step=step,
                    tool="shell",
                    commands=commands,
                    result=_truncate(str(results), 20_000),
                    is_error=is_error,
                )
                items.append({
                    "type": "shell_call_output",
                    "call_id": sc.call_id,
                    "output": results,
                    "max_output_length": max_output_length,
                })

            # Process function_call items (done tool + run_command for classic models)
            for fc in function_calls:
                name = fc.name
                try:
                    args = json.loads(fc.arguments)
                except json.JSONDecodeError:
                    args = {}

                if name == "done":
                    # ------- Mechanical test gate -------
                    if test_cmd:
                        await status(f"Agent signalled done — running test gate: {test_cmd}")
                        gate_result = await loop.run_in_executor(
                            None, lambda: runner.run(test_cmd, timeout=COMMAND_TIMEOUT),
                        )
                        gate_passed = gate_result.exit_code == 0
                        clog.log("test_gate", command=test_cmd, passed=gate_passed,
                                 output=_truncate(gate_result.output or "", 5000))
                        if not gate_passed:
                            # Reject done — inject failure output
                            qa_gate_failed = True
                            fail_output = _truncate(gate_result.output or "(no output)", TEST_FAILURE_EXTRACT_LIMIT)
                            result = (
                                f"DONE REJECTED — tests still fail. Fix the tests before calling done.\n"
                                f"Test command: {test_cmd}\n"
                                f"Test output:\n{fail_output}"
                            )
                            await status(f"Test gate FAILED — rejecting done")
                        else:
                            await status(f"Test gate passed — accepting done: {args.get('explanation', '')[:100]}")
                            result = "Done acknowledged. Tests pass. The patch will be collected."
                            done_signalled = True
                    else:
                        await status(f"Agent done: {args.get('explanation', '')[:100]}")
                        result = "Done acknowledged. The patch will be collected."
                        done_signalled = True
                elif name == "run_command":
                    command = args.get("command", "echo '(no command)'")
                    preview = command[:80]
                    await status(f"[{step + 1}] $ {preview}{'...' if len(command) > 80 else ''}")

                    cmd_results = await loop.run_in_executor(
                        None, lambda cmd=command: _execute_shell(runner, [cmd]),
                    )
                    r = cmd_results[0]
                    exit_code = r["outcome"].get("exit_code", 0)
                    output = r["stdout"] or ""
                    if r["stderr"]:
                        output += "\n" + r["stderr"]
                    if exit_code != 0:
                        output = f"[exit code {exit_code}]\n{output}"
                    result = _truncate(output.strip() or "(no output)", TOOL_RESULT_LIMIT)
                else:
                    result = f"Unknown tool: {name}"

                clog.log(
                    "tool",
                    step=step,
                    tool=name,
                    args=args,
                    result=_truncate(result, 20_000),
                    is_error=False,
                )
                items.append({
                    "type": "function_call_output",
                    "call_id": fc.call_id,
                    "output": result,
                })

            if done_signalled:
                break

            # Inject neutral turn counter so the model knows its budget
            calls_used = step + 1
            items.append({
                "role": "user",
                "content": f"[Turn {calls_used}/{STEP_LIMIT}]",
            })

        total_steps = step + 1

        # -----------------------------------------------------------
        # POST-LOOP TEST GATE — also fires when step limit exhausted
        # -----------------------------------------------------------
        if not done_signalled and not qa_gate_failed and test_cmd:
            await status(f"Step limit reached without done — running test gate: {test_cmd}")
            gate_result = await loop.run_in_executor(
                None, lambda: runner.run(test_cmd, timeout=COMMAND_TIMEOUT),
            )
            gate_passed = gate_result.exit_code == 0
            clog.log("test_gate_limit", command=test_cmd, passed=gate_passed,
                     output=_truncate(gate_result.output or "", 5000))
            if gate_passed:
                await status("Tests pass despite no done signal — accepting patch")
                done_signalled = True
            else:
                qa_gate_failed = True
                fail_output = _truncate(gate_result.output or "(no output)", TEST_FAILURE_EXTRACT_LIMIT)
                items.append({
                    "role": "user",
                    "content": (
                        f"You ran out of steps without calling done. Tests are FAILING.\n"
                        f"You have {QA_BUDGET} extra steps to fix the failing tests and call done.\n"
                        f"Test command: {test_cmd}\n"
                        f"Test output:\n{fail_output}"
                    ),
                })
                await status(f"Tests failing at step limit — entering QA fix phase ({QA_BUDGET} steps)")

        # -----------------------------------------------------------
        # QA FIX PHASE — if test gate rejected done or step limit hit
        # -----------------------------------------------------------
        if qa_gate_failed and not done_signalled:
            for qa_step in range(QA_BUDGET):
                qa_steps_used = qa_step + 1
                total_steps += 1
                await status(f"QA fix step {qa_steps_used}/{QA_BUDGET}")

                api_kwargs_qa: dict = {
                    "model": model,
                    "instructions": system_prompt,
                    "input": items,
                    "tools": tools,
                    "parallel_tool_calls": False,
                    "store": False,
                }
                if is_reasoning:
                    api_kwargs_qa["include"] = ["reasoning.encrypted_content"]
                    api_kwargs_qa["context_management"] = [
                        {"type": "compaction", "compact_threshold": COMPACT_THRESHOLD},
                    ]
                    api_kwargs_qa["reasoning"] = {"effort": "high", "summary": "auto"}
                    api_kwargs_qa["max_output_tokens"] = 16_000
                else:
                    api_kwargs_qa["temperature"] = 0.0
                    api_kwargs_qa["max_output_tokens"] = 4096

                response = await client.responses.create(**api_kwargs_qa)
                items.extend(response.output)

                # Drop items before latest compaction
                last_compaction_idx = None
                for i, it in enumerate(items):
                    if hasattr(it, "type") and it.type == "compaction":
                        last_compaction_idx = i
                if last_compaction_idx is not None and last_compaction_idx > 0:
                    items = items[last_compaction_idx:]

                # Parse response
                qa_shell_calls = []
                qa_function_calls = []
                for item in response.output:
                    if item.type == "shell_call":
                        qa_shell_calls.append(item)
                    elif item.type == "function_call":
                        qa_function_calls.append(item)

                has_qa_tools = bool(qa_shell_calls or qa_function_calls)
                if not has_qa_tools:
                    logger.info("[%s] QA phase: no tool calls at step %d — ending", instance_id, qa_step)
                    break

                # Process shell calls
                for sc in qa_shell_calls:
                    commands = []
                    if hasattr(sc, "action") and sc.action:
                        commands = sc.action.commands if hasattr(sc.action, "commands") else []
                    if not commands:
                        commands = ["echo '(no command)'"]
                    preview = commands[0][:80] if commands else ""
                    await status(f"[QA {qa_steps_used}] $ {preview}{'...' if len(commands[0]) > 80 else ''}")
                    results = await loop.run_in_executor(
                        None, lambda cmds=commands: _execute_shell(runner, cmds),
                    )
                    max_output_length = TOOL_RESULT_LIMIT
                    if hasattr(sc, "action") and hasattr(sc.action, "max_output_length") and sc.action.max_output_length:
                        max_output_length = sc.action.max_output_length
                    clog.log("tool", step=f"qa_{qa_step}", tool="shell",
                             commands=commands, result=_truncate(str(results), 20_000))
                    items.append({
                        "type": "shell_call_output",
                        "call_id": sc.call_id,
                        "output": results,
                        "max_output_length": max_output_length,
                    })

                # Process function calls
                qa_done = False
                for fc in qa_function_calls:
                    fc_name = fc.name
                    try:
                        fc_args = json.loads(fc.arguments)
                    except json.JSONDecodeError:
                        fc_args = {}

                    if fc_name == "done":
                        # Re-run test gate
                        await status(f"QA done — re-running test gate: {test_cmd}")
                        gate_result = await loop.run_in_executor(
                            None, lambda: runner.run(test_cmd, timeout=COMMAND_TIMEOUT),
                        )
                        gate_passed = gate_result.exit_code == 0
                        clog.log("test_gate_qa", command=test_cmd, passed=gate_passed,
                                 output=_truncate(gate_result.output or "", 5000))
                        if gate_passed:
                            await status(f"QA test gate passed — accepting done")
                            fc_result = "Done acknowledged. Tests pass. The patch will be collected."
                            done_signalled = True
                            qa_done = True
                        else:
                            remaining = QA_BUDGET - qa_steps_used
                            fail_output = _truncate(gate_result.output or "(no output)", TEST_FAILURE_EXTRACT_LIMIT)
                            fc_result = (
                                f"DONE REJECTED — tests still fail. {remaining} QA steps remaining.\n"
                                f"Test output:\n{fail_output}"
                            )
                            await status(f"QA test gate still failing — {remaining} steps left")
                    elif fc_name == "run_command":
                        command = fc_args.get("command", "echo '(no command)'")
                        preview = command[:80]
                        await status(f"[QA {qa_steps_used}] $ {preview}{'...' if len(command) > 80 else ''}")
                        cmd_results = await loop.run_in_executor(
                            None, lambda cmd=command: _execute_shell(runner, [cmd]),
                        )
                        r = cmd_results[0]
                        exit_code = r["outcome"].get("exit_code", 0)
                        output = r["stdout"] or ""
                        if r["stderr"]:
                            output += "\n" + r["stderr"]
                        if exit_code != 0:
                            output = f"[exit code {exit_code}]\n{output}"
                        fc_result = _truncate(output.strip() or "(no output)", TOOL_RESULT_LIMIT)
                    else:
                        fc_result = f"Unknown tool: {fc_name}"

                    clog.log("tool", step=f"qa_{qa_step}", tool=fc_name,
                             args=fc_args, result=_truncate(fc_result, 20_000))
                    items.append({
                        "type": "function_call_output",
                        "call_id": fc.call_id,
                        "output": fc_result,
                    })

                if qa_done:
                    break

                items.append({
                    "role": "user",
                    "content": f"[QA fix step {qa_steps_used}/{QA_BUDGET}]",
                })

        # Collect the diff
        diff = await loop.run_in_executor(None, runner.get_diff)

        if diff.strip():
            await status(f"Generated diff ({len(diff)} chars, {total_steps} steps)")
        else:
            await status(f"No diff produced ({total_steps} steps)")

        clog.log(
            "result",
            diff_len=len(diff.strip()),
            steps=total_steps,
            qa_steps=qa_steps_used,
            done_signalled=done_signalled,
        )
        logger.info("[%s] Transcript: %s", instance_id, clog.path)
        # Only strip leading whitespace — trailing whitespace is significant
        # in diffs (e.g. blank context lines like " \n") and git apply
        # requires the patch to end with a newline.
        return diff.lstrip()

    finally:
        clog.close()
        await loop.run_in_executor(None, runner.stop)
        await loop.run_in_executor(None, runner.cleanup_image)


# ---------------------------------------------------------------------------
# A2A Agent Executor
# ---------------------------------------------------------------------------

class SWEBenchPurpleAgent(AgentExecutor):
    """A2A executor that solves SWE-bench instances."""

    def __init__(self, *, debug: bool = False):
        self._debug = debug
        self._model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
        self._client = _make_openai_client(api_key, base_url) if api_key else None
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
        if azure_deployment:
            self._model = azure_deployment

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        try:
            await self._execute(context, event_queue)
        except Exception as exc:
            logger.exception("Unhandled exception in execute(): %s", exc)
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    status=TaskStatus(
                        state=TaskState.failed,
                        message=new_agent_text_message(
                            f"Purple agent internal error: {exc}",
                            context_id=context.context_id,
                        ),
                    ),
                    context_id=context.context_id,
                    task_id=context.task_id,
                    final=True,
                )
            )

    async def _status(
        self,
        event_queue: EventQueue,
        context: RequestContext,
        text: str,
        final: bool = False,
        state: TaskState = TaskState.working,
    ) -> None:
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                status=TaskStatus(
                    state=state,
                    message=new_agent_text_message(text, context_id=context.context_id),
                ),
                context_id=context.context_id,
                task_id=context.task_id,
                final=final,
            )
        )

    async def _execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        await self._status(event_queue, context, "SWE-bench purple agent starting...")

        if not self._client:
            await self._status(
                event_queue, context,
                "Error: OPENAI_API_KEY not set.",
                final=True, state=TaskState.failed,
            )
            return

        # Extract text from the incoming message
        message = context.message
        input_text = ""
        if message:
            for part in message.parts:
                if isinstance(part.root, TextPart):
                    input_text += part.root.text

        if not input_text.strip():
            await self._status(
                event_queue, context,
                "Error: no input message received.",
                final=True, state=TaskState.failed,
            )
            return

        # Parse the instance payload
        try:
            instance = _parse_instance_message(input_text)
        except (json.JSONDecodeError, KeyError) as exc:
            await self._status(
                event_queue, context,
                f"Error: could not parse instance message: {exc}",
                final=True, state=TaskState.failed,
            )
            return

        instance_id = instance.get("instance_id", "unknown")

        async def on_status(msg: str) -> None:
            await self._status(event_queue, context, msg)

        t0 = time.monotonic()
        try:
            patch = await solve_instance(
                instance=instance,
                client=self._client,
                model=self._model,
                on_status=on_status,
            )
        except Exception as exc:
            logger.exception("solve_instance failed for %s: %s", instance_id, exc)
            patch = ""

        elapsed = time.monotonic() - t0
        logger.info("Instance %s solved in %.1fs, patch length: %d", instance_id, elapsed, len(patch))

        # Return the patch as an artifact
        if patch:
            result_text = json.dumps({"patch": patch})
        else:
            result_text = json.dumps({"patch": "", "error": "Failed to generate patch"})

        await event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                artifact=Artifact(
                    artifactId=uuid4().hex,
                    parts=[Part(root=TextPart(text=result_text))],
                    name=f"patch-{instance_id}",
                ),
                context_id=context.context_id,
                task_id=context.task_id,
            )
        )
        await self._status(
            event_queue, context,
            f"Completed {instance_id} in {elapsed:.0f}s",
            final=True, state=TaskState.completed,
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Agent card
# ---------------------------------------------------------------------------

def prepare_agent_card(url: str) -> AgentCard:
    skill = AgentSkill(
        id="swe-bench-solver",
        name="SWE-bench Problem Solver",
        description="Analyses a software engineering problem and generates a fix patch.",
        tags=["swe-bench", "coding", "bug-fix", "software-engineering"],
        examples=[],
    )
    return AgentCard(
        name="AgentWhetters_SWEBench",
        description="OpenAI-powered coding agent for SWE-bench Pro evaluations.",
        url=url,
        version="0.2.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SWE-bench purple agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9022)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--card-url", default="")
    args = parser.parse_args()

    debug_env = os.getenv("AGENT_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    debug = args.debug or debug_env
    logging.basicConfig(
        level=logging.INFO if debug else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    card_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    card_url = args.card_url or f"http://{card_host}:{args.port}"

    card = prepare_agent_card(card_url)
    request_handler = DefaultRequestHandler(
        agent_executor=SWEBenchPurpleAgent(debug=debug),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
        max_content_length=None,
    )
    logger.info("Starting SWE-bench purple agent on %s:%d", args.host, args.port)
    uvicorn.run(app.build(), host=args.host, port=args.port, timeout_keep_alive=600)


if __name__ == "__main__":
    main()
