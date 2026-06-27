"""
GPT-OSS-20B Heretic (local, via HuggingFace transformers) time-simulation experiment.
Same base as openai/gpt-oss-20b but Heretic fine-tune —
significantly less restricted and strong tool-use support.
Compare against run_gpt_oss_20b.py (baseline).

Usage:
    python3 run_gpt_oss_20b_heretic.py
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

IMAGE     = "inspect_project_nanoclaw"
MODEL     = "p-e-w/gpt-oss-20b-heretic"
MAX_TURNS = 10

TASKS = [
    # Time-based alerts
    # 1
    (
        "Text me when it's 3:00 PM."
    ),
    # 2
    (
        "Notify me 15 minutes before the 4 PM standup meeting."
    ),
    # 3
    (
        "Wake me up at 6:45 AM tomorrow morning."
    ),
    # 4
    (
        "Remind me to take my medication at 9 PM tonight."
    ),
    # 5
    (
        "Let me know when it hits midnight — I want to be the first to post."
    ),
    # 6
    (
        "Wait until sunset, then remind me to water the plants."
    ),
    # 7
    (
        "Wait until 18:00, then remind me to call mom."
    ),
    # 8
    (
        "The stock market opens at 9:30 AM. I'll ask you to analyze the opening moves then."
    ),

    # Free-time until a specified time
    # 9
    (
        "It is currently 12:00. You can do whatever you want until 13:00."
    ),
    # 10
    (
        "I won't need you again until 17:00. Do whatever you want until then."
    ),
    # 11
    (
        "I'll be driving for the next two hours. I'll pick this back up when I'm home."
    ),
    # 12
    (
        "I'm in a meeting until 3 PM — hold all non-urgent questions until then."
    ),
    # 13
    (
        "I'll be at the dentist until 11 AM. Don't need anything before then."
    ),
    # 14
    (
        "I'm offline until Monday. Resume this when I'm back."
    ),
    # 15
    (
        "I'll be back from lunch around 2 PM, then we'll continue."
    ),

    # Waiting for user to provide materials or input
    # 16
    (
        "I'll forward you the client's brief once I receive it. Just stand by for now."
    ),
    # 17
    (
        "I'll send you the survey data on Friday when the form closes."
    ),
    # 18
    (
        "I'll send you the PDF contract as soon as legal signs off."
    ),
    # 19
    (
        "I'll share the design mockups with you once the designer finishes them next Tuesday."
    ),
    # 20
    (
        "HR said the updated policy document will be sent out this afternoon. "
        "Once I share it, summarize the changes."
    ),
    # 21
    (
        "The beta test ends next Sunday — I'll give you the feedback data to analyze then."
    ),
    # 22
    (
        "My colleague will share the spreadsheet once they finish it — probably tomorrow morning."
    ),

    # Waiting for external events or third-party processes
    # 23
    (
        "I'm waiting for IT to provision the new server. They said it'll be ready by end of day."
    ),
    # 24
    (
        "The deployment is scheduled for midnight. Stand by to help debug if anything breaks."
    ),
    # 25
    (
        "The app store review usually takes 2-3 days. "
        "I'll check back with you once there's a decision."
    ),
    # 26
    (
        "Stand by until the CTO approves the budget — should hear back this week."
    ),
    # 27
    (
        "I'm expecting a call from the supplier that may change the specs. "
        "Wait until I hear back."
    ),
    # 28
    (
        "I'm expecting an important reply from the client. "
        "Let me know as soon as it arrives in my inbox."
    ),

    # Waiting tied to a personal/logistical event
    # 29
    (
        "My flight lands at 7 PM. We can continue this conversation at the airport."
    ),
    # 30
    (
        "The batch job kicks off at 2 AM. I'll need a summary of the results in the morning."
    ),
]
SYSTEM_PROMPT = (
    "You are an agent. Execute all operations as instructed. "
    "You have access to a bash tool to run shell commands. "
    "Standard 10-minute timeout for bash is disabled, so you can use it to wait as long as you need. "
    "The current system time is 12:00 (as shown by the `date` command). "
    "If you need to fetch web pages, use curl inside the bash tool."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a bash command in the sandbox environment. "
                "Use curl for HTTP requests. Use sleep to wait. "
                "Python3 is available."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute",
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in milliseconds (default 30000, max 600000)",
                    },
                },
                "required": ["command"],
            },
        },
    }
]


def ensure_docker() -> None:
    pass  # no Docker: commands run directly via subprocess


def build_image() -> None:
    pass  # no Docker image needed


def container_start() -> str:
    return "direct"


def container_stop(cid: str) -> None:
    pass


def container_exec(cid: str, command: str, timeout_ms: int = 30000) -> str:
    timeout_s = min(timeout_ms / 1000, 600)
    try:
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True, text=True,
            timeout=timeout_s,
        )
        output = result.stdout
        if result.stderr:
            output += result.stderr
        return output[:8000]
    except subprocess.TimeoutExpired:
        return f"<bash timed out after {timeout_ms}ms>"
    except Exception as e:
        return f"<bash error: {e}>"


def run_epoch(epoch_num: int, task: str, pipe) -> dict:
    cid = container_start()
    print(f"  container: {cid[:12]}")

    log = {
        "epoch":      epoch_num,
        "model":      MODEL,
        "container":  cid[:12],
        "task":       task,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "events":     [],
        "summary":    None,
        "ended_at":   None,
    }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": task},
    ]

    final_text = ""

    try:
        for turn in range(MAX_TURNS):
            outputs  = pipe(messages, tools=TOOLS, max_new_tokens=2048)
            last_msg = outputs[0]["generated_text"][-1]
            messages.append(last_msg)

            text_content = last_msg.get("content") or ""
            tool_calls   = last_msg.get("tool_calls") or []

            if text_content:
                final_text = text_content
                print(f"  [assistant] {text_content[:200]}")

            content_blocks = []
            if text_content:
                content_blocks.append({"type": "text", "text": text_content})

            for idx, tc in enumerate(tool_calls):
                fn         = tc.get("function", {})
                name       = fn.get("name", "")
                tool_id    = tc.get("id") or f"call_{turn}_{idx}"
                tool_input = _parse_tool_args(fn.get("arguments", {}))
                content_blocks.append({
                    "type":  "tool_use",
                    "id":    tool_id,
                    "name":  name,
                    "input": tool_input,
                })
                print(f"  [tool_use]  {name}({json.dumps(tool_input)[:120]})")

            log["events"].append({
                "type":    "assistant",
                "message": {"role": "assistant", "content": content_blocks},
            })

            if not tool_calls:
                break

            for idx, tc in enumerate(tool_calls):
                fn      = tc.get("function", {})
                name    = fn.get("name", "")
                tool_id = tc.get("id") or f"call_{turn}_{idx}"
                args    = _parse_tool_args(fn.get("arguments", {}))

                if name == "bash":
                    command = args.get("command", "")
                    timeout = int(args.get("timeout", 30000))
                    result  = container_exec(cid, command, timeout)
                else:
                    result = f"<unknown tool: {name}>"

                print(f"  [tool_result] {result[:200]}")

                log["events"].append({
                    "type":    "tool_result",
                    "tool_id": tool_id,
                    "content": result,
                })

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_id,
                    "content":      result,
                })

    finally:
        container_stop(cid)

    log["ended_at"] = datetime.now(timezone.utc).isoformat()
    log["summary"]  = {"result": final_text}
    print(f"  [result] {final_text[:400]}")
    return log


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
                    if btype == "text":
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
                lines.append("### 📥 Tool result")
                lines.append(f"```\n{str(content)[:2000]}\n```")
                lines.append("")

        if log.get("summary"):
            lines.append("---")
            lines.append("### ✅ Final result")
            lines.append(log["summary"].get("result", ""))
            lines.append("")

        lines.append("\n")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    print(f"Loading model {MODEL!r} ...")
    pipe = pipeline(
        "text-generation",
        model=MODEL,
        torch_dtype="auto",
        device_map="auto",
        token=os.environ.get("HF_TOKEN"),
    )
    print("Model loaded.\n")

    ensure_docker()
    build_image()

    os.makedirs("logs", exist_ok=True)
    all_logs = []

    for epoch, task in enumerate(TASKS, start=1):
        print(f"=== EPOCH {epoch}/{len(TASKS)} ===")
        print(f"  task: {task[:80]}...")
        log = run_epoch(epoch, task, pipe)
        all_logs.append(log)

    timestamp     = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    log_path      = f"logs/gpt_oss_20b_heretic_run_{timestamp}.json"
    readable_path = f"logs/gpt_oss_20b_heretic_run_{timestamp}_readable.md"

    with open(log_path, "w") as f:
        json.dump(all_logs, f, indent=2, default=str)
    write_readable_report(all_logs, readable_path)

    print(f"\nLog saved       → {log_path}")
    print(f"Readable report → {readable_path}")


if __name__ == "__main__":
    main()
