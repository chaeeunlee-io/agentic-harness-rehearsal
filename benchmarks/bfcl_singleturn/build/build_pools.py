"""Build BFCL single-turn distillation datasets: fc-only, json-only, t2-only, and 3h.

Input: Qwen3-32B teacher traces from the training split, with one or more jobs
per harness passed through `--job`. Each solved trial (reward 1) becomes a row,
with at most two rows per task, sampled with seed 0 if needed.

`3h` combines the fc, json, and t2 datasets. `xml` is held out.

For `fc`, rows keep native tool calls. `json` keeps the model's raw text,
including `<think>`. `t2` rows are rebuilt from Terminus-2's trajectory as the
JSON it emitted (analysis, plan, commands), since its prose alone carries no
commands. Successful `fc` trials with no tool call are also kept for irrelevance
tasks where the correct behavior is not to call a tool.
"""
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

HARNESSES = ["fc", "json", "t2"]
JOBS = {h: f"teacher-{h}" for h in HARNESSES}
PER_TASK = 2


def reward(trial):
    try:
        return float(open(f"{trial}/verifier/reward.txt").read().strip())
    except Exception:
        return 0.0


def task_of(trial):
    return os.path.basename(trial.rstrip("/")).split("__")[0]


def chat_t2(trial):
    """Terminus-2's ATIF trajectory as a chat, with each response rebuilt as the JSON it emitted.

    The trajectory splits a response into prose (`message`) and the executed commands
    (`tool_calls`); training on the prose alone teaches prose without commands."""
    path = f"{trial}agent/trajectory.json"
    if not os.path.exists(path):
        return None
    steps = json.load(open(path)).get("steps") or []
    if len(steps) < 2 or steps[0].get("source") != "user":
        return None
    agent_steps = [s for s in steps[1:] if s.get("source") == "agent"]
    if not agent_steps:
        return None
    msgs = [{"role": "user", "content": steps[0].get("message") or ""}]
    for i, step in enumerate(agent_steps):
        prose = step.get("message") or ""
        analysis, plan = prose, ""
        if "\nPlan:" in prose:
            a, _, rest = prose.partition("\nPlan:")
            analysis, plan = a.replace("Analysis:", "", 1).strip(), rest.strip()
        elif prose.startswith("Analysis:"):
            analysis = prose.replace("Analysis:", "", 1).strip()
        commands, done = [], False
        for call in step.get("tool_calls") or []:
            args = call.get("arguments") or {}
            if call.get("function_name") == "bash_command":
                commands.append({"keystrokes": args.get("keystrokes", ""), "duration": args.get("duration", 1.0)})
            elif call.get("function_name") == "mark_task_complete":
                done = True
        response = {"analysis": analysis, "plan": plan, "commands": commands, "task_complete": done}
        msgs.append({"role": "assistant", "content": json.dumps(response, indent=2)})
        obs = step.get("observation")
        if obs and i < len(agent_steps) - 1:
            msgs.append({"role": "user", "content": obs if isinstance(obs, str) else json.dumps(obs)})
    return msgs


def tool_calls(raw_calls):
    """The recorded tool calls in the form the chat template needs (name + string arguments)."""
    calls = []
    for c in raw_calls:
        if isinstance(c, str):
            # stored as repr(ChatCompletionMessageToolCall(function=Function(arguments='...', name='...'), ...))
            args = re.search(r"arguments='(.*?)', name='", c, re.DOTALL)
            name = re.search(r", name='([^']*)'\)", c)
            if not (args and name):
                continue
            name, args = name.group(1), args.group(1)
        elif isinstance(c, dict):
            fn = c.get("function") or {}
            name, args = fn.get("name"), fn.get("arguments")
            if not isinstance(args, str):
                args = json.dumps(args if args is not None else {})
        else:
            continue
        if name:
            calls.append({"id": f"call_{len(calls)}", "type": "function", "function": {"name": name, "arguments": args}})
    return calls


def chat_envelope(trial, harness):
    """The fc / json agent's transcript: the request messages plus the model's final reply."""
    path = f"{trial}agent/bfcl-agent-transcript.json"
    if not os.path.exists(path):
        return None
    transcript = json.load(open(path))
    msgs, raw = list(transcript.get("messages") or []), transcript.get("raw_response") or {}
    if not msgs or not raw:
        return None
    reply = {"role": "assistant", "content": raw.get("content") or ""}
    if harness == "fc" and raw.get("tool_calls"):
        calls = tool_calls(raw["tool_calls"])
        if calls:
            reply["tool_calls"] = calls
    return msgs + [reply]


def build(dirs, harness, rng):
    by_task = defaultdict(list)
    for trial in C.trials(dirs, "bfcl-*/"):
        if reward(trial) < 1.0:
            continue
        msgs = chat_t2(trial) if harness == "t2" else chat_envelope(trial, harness)
        if msgs:
            by_task[task_of(trial)].append({"messages": msgs, "meta": {"harness": harness, "task": task_of(trial)}})
    rows = []
    for task in sorted(by_task):
        rows += by_task[task] if len(by_task[task]) <= PER_TASK else rng.sample(by_task[task], PER_TASK)
    return rows


def main():
    args = C.parser(__doc__, JOBS).parse_args()
    jobs = C.job_dirs(args, JOBS)
    rng = random.Random(0)
    pools = {}
    for h in HARNESSES:
        pools[h] = build(jobs[h], h, rng)
        C.write_pool(args.out_dir, f"{h}-only", pools[h])
    C.write_pool(args.out_dir, "3h", pools["fc"] + pools["json"] + pools["t2"])


if __name__ == "__main__":
    main()
