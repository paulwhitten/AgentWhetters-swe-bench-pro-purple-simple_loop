"""
Starts all agents defined in a scenario TOML then drives the evaluation
using the client CLI.

Usage (from the project root):
    uv run python -m agentbeats.run_scenario scenario.toml --show-logs
"""

import argparse
import asyncio
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import httpx
from dotenv import load_dotenv
from a2a.client import A2ACardResolver

load_dotenv(override=True)


# ---------------------------------------------------------------------------
# TOML parsing
# ---------------------------------------------------------------------------

def _host_port(endpoint: str) -> tuple[str, int]:
    s = endpoint.replace("http://", "").replace("https://", "").split("/", 1)[0]
    host, port = s.split(":", 1)
    return host, int(port)


def parse_toml(scenario_path: str) -> dict:
    path = Path(scenario_path)
    if not path.exists():
        print(f"Error: Scenario file not found: {path}")
        sys.exit(1)

    data = tomllib.loads(path.read_text())

    green_ep = data.get("green_agent", {}).get("endpoint", "")
    g_host, g_port = _host_port(green_ep)
    green_cmd = data.get("green_agent", {}).get("cmd", "")
    green_cwd = data.get("green_agent", {}).get("cwd", "")

    parts = []
    for p in data.get("participants", []):
        if isinstance(p, dict) and "endpoint" in p:
            h, pt = _host_port(p["endpoint"])
            parts.append(
                {
                    "role": str(p.get("role", "")),
                    "host": h,
                    "port": pt,
                    "cmd": p.get("cmd", ""),
                    "cwd": p.get("cwd", ""),
                }
            )

    return {
        "green_agent": {"host": g_host, "port": g_port, "cmd": green_cmd, "cwd": green_cwd},
        "participants": parts,
        "config": data.get("config", {}),
    }


# ---------------------------------------------------------------------------
# Health-check
# ---------------------------------------------------------------------------

async def wait_for_agents(cfg: dict, timeout: int = 60) -> bool:
    endpoints: list[str] = []

    for p in cfg["participants"]:
        if p.get("cmd"):
            endpoints.append(f"http://{p['host']}:{p['port']}")

    if cfg["green_agent"].get("cmd"):
        endpoints.append(
            f"http://{cfg['green_agent']['host']}:{cfg['green_agent']['port']}"
        )

    if not endpoints:
        return True

    print(f"Waiting for {len(endpoints)} agent(s) to be ready…")
    start = time.time()

    async def ok(ep: str) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2) as c:
                await A2ACardResolver(httpx_client=c, base_url=ep).get_agent_card()
            return True
        except Exception:
            return False

    ready_count = 0
    while time.time() - start < timeout:
        ready_count = sum([await ok(ep) for ep in endpoints])
        if ready_count == len(endpoints):
            print("All agents ready.")
            return True
        print(f"  {ready_count}/{len(endpoints)} agents ready, retrying…")
        await asyncio.sleep(2)

    print(f"Timeout: only {ready_count}/{len(endpoints)} agents ready after {timeout}s")
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent scenario")
    parser.add_argument("scenario", help="Path to scenario TOML file")
    parser.add_argument("--show-logs", action="store_true", help="Show agent stdout/stderr")
    parser.add_argument(
        "--serve-only",
        action="store_true",
        help="Start agent servers only without running evaluation",
    )
    args = parser.parse_args()

    cfg = parse_toml(args.scenario)

    sink = None if args.show_logs or args.serve_only else subprocess.DEVNULL
    parent_bin = str(Path(sys.executable).parent)
    base_env = os.environ.copy()
    base_env["PATH"] = parent_bin + os.pathsep + base_env.get("PATH", "")

    procs: list[subprocess.Popen] = []
    try:
        # Start participant agents (purple, etc.)
        for p in cfg["participants"]:
            cmd_args = shlex.split(p.get("cmd", ""))
            if cmd_args:
                cwd = p.get("cwd") or None
                print(f"Starting {p['role']} agent at {p['host']}:{p['port']}")
                procs.append(
                    subprocess.Popen(
                        cmd_args,
                        cwd=cwd,
                        env=base_env,
                        stdout=sink,
                        stderr=sink,
                        text=True,
                        start_new_session=True,
                    )
                )

        # Start the green agent
        green_cmd_args = shlex.split(cfg["green_agent"].get("cmd", ""))
        if green_cmd_args:
            green_cwd = cfg["green_agent"].get("cwd") or None
            print(
                f"Starting green agent at "
                f"{cfg['green_agent']['host']}:{cfg['green_agent']['port']}"
            )
            procs.append(
                subprocess.Popen(
                    green_cmd_args,
                    cwd=green_cwd,
                    env=base_env,
                    stdout=sink,
                    stderr=sink,
                    text=True,
                    start_new_session=True,
                )
            )

        if not asyncio.run(wait_for_agents(cfg)):
            print("Error: not all agents became ready. Exiting.")
            return

        print("All agents started. Press Ctrl+C to stop.")

        if args.serve_only:
            while True:
                for proc in procs:
                    if proc.poll() is not None:
                        print(f"An agent exited with code {proc.returncode}")
                        return
                time.sleep(0.5)
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"logs/{ts}/results.json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            client_proc = subprocess.Popen(
                [sys.executable, "-m", "agentbeats.client_cli", args.scenario, output_file],
                env=base_env,
                start_new_session=True,
            )
            # Monitor participant health while client runs; restart crashed agents.
            while client_proc.poll() is None:
                for idx, p in enumerate(cfg["participants"]):
                    proc = procs[idx]
                    if proc.poll() is not None:
                        print(
                            f"\n⚠ {p['role']} agent (PID {proc.pid}) died "
                            f"with code {proc.returncode} — restarting…"
                        )
                        cmd_args = shlex.split(p.get("cmd", ""))
                        cwd = p.get("cwd") or None
                        new_proc = subprocess.Popen(
                            cmd_args,
                            cwd=cwd,
                            env=base_env,
                            stdout=sink,
                            stderr=sink,
                            text=True,
                            start_new_session=True,
                        )
                        procs[idx] = new_proc
                        print(
                            f"  Restarted {p['role']} agent "
                            f"(new PID {new_proc.pid})"
                        )
                time.sleep(1)
            if client_proc.returncode != 0:
                print(f"\nClient exited with code {client_proc.returncode}")

    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down…")
        for p in procs:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        time.sleep(1)
        for p in procs:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    main()
