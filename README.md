# SWE-bench Pro Purple Agent

A2A coding agent that solves [SWE-bench Pro](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro) problems. Receives issue descriptions from the green agent, analyses the repository in a Docker container, and returns a fix patch.

## How it works (simple_loop architecture)

This agent runs a single flat agent loop —
no strategy abstraction, no provider hierarchy, no separate prompt
modules. Full reference: [`docs/architecture-simple_loop.md`](../docs/architecture-simple_loop.md).

1. **Receive** a JSON message from the green agent containing the
   problem statement, Docker image URI, base commit, repo name,
   and optional hints.
2. **Start container** — pull the image, start it with
   `tail -f /dev/null`, and `git checkout` the base commit so the
   repo is in a clean state.
3. **Auto-start services** — heuristically launch Redis / MongoDB /
   PostgreSQL when the repo's config files reference them.
4. **Discover test command** — probe `package.json`, `pytest.ini`,
   `go.mod`, `Cargo.toml`, etc. in a fixed order; pick the first
   working invocation (e.g. `npm test`, `python -m pytest -x`,
   `go test ./...`). Run it once to capture pre-existing failures.
5. **Build the initial user message** — repo header, depth-2 file
   listing, problem statement, hints, test command, and the
   captured baseline failure output.
6. **Run the main loop** for up to `STEP_LIMIT = 50` turns, with the
   model driving a bash shell (native `shell_call` tool for
   reasoning models, `run_command` + `done` function-calling for
   classic models). Each turn appends a `[Turn N/50]` reminder.
7. **Mechanical test gate** — when the model emits `done`, run the
   discovered test command. If it passes, accept the patch. If it
   fails, inject the failure output and reject `done`.
8. **Post-loop test gate** — if the agent ran out of steps without
   ever calling `done`, run the test command anyway. Pass → accept.
   Fail → continue into the QA fix phase.
9. **QA fix phase** — up to `QA_BUDGET = 15` extra turns to fix
   failing tests; each `done` re-runs the gate.
10. **Return** the working tree's `git diff` to the green agent as
    an A2A artifact.

The system prompt and tool set are chosen by a prefix check on the configured model name. Reasoning-class models receive a system prompt tuned for native shell tool use and have provider-side reasoning effort enabled. Other models receive a classic chat-style system prompt with explicit `run_command` and `done` function-calling tools. The exact prefix list and prompts live in `src/purple/server.py`.

## Quick start

```bash
# Install dependencies
uv sync

# Set your OpenAI key
cp .env.example .env
# Edit .env with your actual key

# Run locally
source .env
uv run src/purple/server.py --host 127.0.0.1 --port 9022 --debug
```

## Docker

```bash
docker build -t swe-bench-purple-agent .

docker run -d -p 9022:9022 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  swe-bench-purple-agent

curl http://localhost:9022/.well-known/agent-card.json
```

The Docker socket mount is required because the purple agent starts sibling containers to inspect repositories.

### Port 9022 (note)

The upstream baseline purple agent at
[RDI-Foundation/swe-bench-purple-agent](https://github.com/RDI-Foundation/swe-bench-purple-agent)
defaults to port **9009**. This fork defaults to **9022** so the
purple agent can be run concurrently with the green agent (also
on 9009) on a single development host without a port collision.

The change is internal-only:

- Production grading runs the container in isolation, so the bind
  port is irrelevant — Amber routes traffic via the
  `endpoints[].port` value declared in `amber/amber-manifest-purple.json5`,
  which is also `9022`.
- The Dockerfile `EXPOSE`, the `CMD --port` flag, the amber
  manifest endpoint, the test conftest, and the CI workflow all
  agree on `9022`. Verify with:

  ```bash
  grep -rnE "\b9022\b|\b9009\b" Dockerfile src/ amber/ tests/ .github/
  ```

If you want to match upstream exactly, change `9022` to `9009` in
all four places (`Dockerfile`, `src/purple/server.py` argparse
default, `amber/amber-manifest-purple.json5`,
`.github/workflows/test-and-publish.yml`, plus `tests/conftest.py`)
and stop running green and purple on the same host.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | (required) | OpenAI API key |
| `OPENAI_MODEL` | `gpt-XX` | Model name passed to the LLM provider; override per deployment |
| `OPENAI_BASE_URL` | | Optional base URL for proxy or alternative providers |
| `AZURE_OPENAI_ENDPOINT` | | Azure OpenAI endpoint (mutually exclusive with `OPENAI_BASE_URL`) |
| `AZURE_OPENAI_API_VERSION` | `2024-10-21` | Azure API version |
| `AZURE_OPENAI_DEPLOYMENT` | | Azure deployment name (overrides `OPENAI_MODEL`) |
| `AGENT_DEBUG` | `0` | Set to `1` for verbose logging |

## Architecture

```
src/purple/
  server.py          – A2A server, agent executor, LLM prompts, solve pipeline
  docker_runner.py   – Docker container lifecycle, exec commands, read files, apply patches
```

**Solve pipeline:**

- **Localization**: LLM analyses the file tree + problem statement to pick 3-8 files
- **Generation**: LLM reads the file contents and produces a unified diff
- **Repair loop**: if `git apply` fails, the error is fed back to the LLM (up to 2 retries)
- **Verification**: the patch is applied in the container and `git diff` confirms the result
