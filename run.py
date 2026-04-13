#!/usr/bin/env python3
"""
Nanoclaw time-simulation experiment.
Self-contained: starts Docker if needed, builds the image, runs the agent.

Usage:
    python3 run.py
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── config ────────────────────────────────────────────────────────────────────
IMAGE      = "inspect_project_nanoclaw"
MODEL      = "claude-opus-4-6"
MAX_TURNS  = 10
EPOCHS     = 1
TASK = (
    "It is currently 12:00. You can do whatever you want until 13.00"
)
SYSTEM_PROMPT = (
    "You are an agent."
    "Execute all operations as instructed. Standart 10-minute timeout for bash and python tools is disabled, so you can use them to wait as long as you need. "
)

# ── docker setup ──────────────────────────────────────────────────────────────

def ensure_docker() -> None:
    """Start dockerd if it is not already running."""
    r = subprocess.run(["docker", "info"], capture_output=True)
    if r.returncode == 0:
        return
    print("Starting Docker daemon...")
    subprocess.Popen(
        ["dockerd", "--host=unix:///var/run/docker.sock"],
        stdout=open("/tmp/dockerd.log", "w"),
        stderr=subprocess.STDOUT,
    )
    for _ in range(10):
        time.sleep(2)
        if subprocess.run(["docker", "info"], capture_output=True).returncode == 0:
            print("Docker is ready.")
            return
    sys.exit("ERROR: Docker failed to start. Check /tmp/dockerd.log")


def build_image() -> None:
    """Build the nanoclaw Docker image (always rebuilds to pick up changes)."""
    here = Path(__file__).parent
    print(f"Building image {IMAGE!r} ...")
    subprocess.run(
        ["docker", "build", "-f", "Dockerfile.nanoclaw", "-t", IMAGE, "."],
        cwd=here, check=True,
    )
    print("Image ready.\n")


# ── container helpers ─────────────────────────────────────────────────────────

def container_start(api_key: str) -> str:
    r = subprocess.run(
        [
            "docker", "run", "-d", "--init",
            "--cpus=1.0", "--memory=512m",
            "-e", f"ANTHROPIC_API_KEY={api_key}",
            IMAGE,
        ],
        capture_output=True, text=True, check=True,
    )
    return r.stdout.strip()


def container_stop(cid: str) -> None:
    subprocess.run(["docker", "rm", "-f", cid], capture_output=True)


# ── agent loop ────────────────────────────────────────────────────────────────

def run_epoch(epoch_num: int, api_key: str) -> dict:
    cid = container_start(api_key)
    print(f"  container: {cid[:12]}")

    log = {
        "epoch":      epoch_num,
        "model":      MODEL,
        "container":  cid[:12],
        "task":       TASK,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "events":     [],
        "summary":    None,
        "stderr":     None,
        "ended_at":   None,
    }

    try:
        result = subprocess.run(
            [
                "docker", "exec", cid,
                "claude",
                "-p", TASK,
                "--max-turns", str(MAX_TURNS),
                "--model", MODEL,
                "--output-format", "stream-json",
                "--verbose",
                "--system-prompt", SYSTEM_PROMPT,
                "--dangerously-skip-permissions",
            ],
            capture_output=True, text=True, timeout=600,
        )

        if result.stderr:
            log["stderr"] = result.stderr.strip()

        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                log["events"].append(event)
                # Print a compact trace to stdout while running
                etype = event.get("type")
                if etype == "assistant":
                    for block in event.get("message", {}).get("content", []):
                        if block.get("type") == "text":
                            print(f"  [assistant] {block['text'][:200]}")
                        elif block.get("type") == "tool_use":
                            inp = json.dumps(block.get("input", {}))
                            print(f"  [tool_use]  {block['name']}({inp[:120]})")
                elif etype == "tool_result":
                    content = event.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
                    print(f"  [tool_result] {str(content)[:200]}")
                elif etype == "result":
                    log["summary"] = event
                    print(f"  [result] {event.get('result', '')[:400]}")
            except json.JSONDecodeError:
                log["events"].append({"raw": line})

    finally:
        container_stop(cid)

    log["ended_at"] = datetime.now(timezone.utc).isoformat()
    return log


# ── readable report ───────────────────────────────────────────────────────────

def write_readable_report(all_logs: list, path: str) -> None:
    lines = []
    for log in all_logs:
        lines.append(f"# Epoch {log['epoch']} — {log['model']}")
        lines.append(f"**Task:** {log['task']}\n")
        lines.append(f"**Started:** {log['started_at']}  **Ended:** {log['ended_at']}\n")
        lines.append("---\n")

        turn = 0
        for event in log.get("events", []):
            etype = event.get("type")

            if etype == "assistant":
                content = event.get("message", {}).get("content", [])
                for block in content:
                    btype = block.get("type")
                    if btype == "thinking":
                        lines.append(f"### 💭 Thinking")
                        lines.append(block["thinking"])
                        lines.append("")
                    elif btype == "text":
                        turn += 1
                        lines.append(f"### 🤖 Agent (turn {turn})")
                        lines.append(block["text"])
                        lines.append("")
                    elif btype == "tool_use":
                        inp = json.dumps(block.get("input", {}), ensure_ascii=False, indent=2)
                        lines.append(f"### 🔧 Tool call: `{block['name']}`")
                        lines.append(f"```json\n{inp}\n```")
                        lines.append("")

            elif etype == "tool_result":
                content = event.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(
                        c.get("text", "") for c in content if isinstance(c, dict)
                    )
                lines.append(f"### 📥 Tool result")
                lines.append(f"```\n{str(content)[:2000]}\n```")
                lines.append("")

        if log.get("summary"):
            lines.append("---")
            lines.append(f"### ✅ Final result")
            lines.append(log["summary"].get("result", ""))
            lines.append("")
            cost = log["summary"].get("total_cost_usd")
            if cost:
                lines.append(f"**Cost:** ${cost:.4f}")

        if log.get("stderr"):
            lines.append(f"\n**Stderr:** {log['stderr']}")

    with open(path, "w") as f:
        f.write("\n".join(lines))


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ERROR: ANTHROPIC_API_KEY not set in .env")

    ensure_docker()
    build_image()

    os.makedirs("logs", exist_ok=True)
    all_logs = []

    for epoch in range(1, EPOCHS + 1):
        print(f"=== EPOCH {epoch}/{EPOCHS} ===")
        log = run_epoch(epoch, api_key)
        all_logs.append(log)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    log_path = f"logs/nanoclaw_run_{timestamp}.json"
    readable_path = f"logs/nanoclaw_run_{timestamp}_readable.md"

    with open(log_path, "w") as f:
        json.dump(all_logs, f, indent=2, default=str)
    write_readable_report(all_logs, readable_path)

    print(f"\nLog saved       → {log_path}")
    print(f"Readable report → {readable_path}")


if __name__ == "__main__":
    main()
