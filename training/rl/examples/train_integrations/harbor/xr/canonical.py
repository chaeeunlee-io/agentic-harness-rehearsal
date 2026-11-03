"""Canonical actions + feasibility (types + budget only, per pinned method).

Canonical action = tuple (VERB, detail):
  ("GET", url) | ("POST", url) | ("SUBMIT", "") | ("SHELL", command)
Task actions (GET/POST/SUBMIT) are universally feasible; SHELL is a harness
action, feasible only where the affordance map grants it. Budgets are max
ENVIRONMENT actions (SUBMIT excluded).

Parsers cover our two trainable-ladder sources:
  * parity transcripts   (role/content messages, bare protocol strings)
  * terminus-2 ATIF      (steps with tool_calls; shell commands canonicalized
    to GET/POST when they are recognizable FHIR calls, else SHELL)
"""

from __future__ import annotations

import re
from typing import Any

# Affordance map + budgets for the trainable ladder. Extend alongside
# HARNESS_REGISTRY (harbor adapters/medagentbench/harness_dispatch.py).
HARNESS_CAPS: dict[str, dict] = {
    "parity":        {"actions": {"GET", "POST", "SUBMIT"}, "budget": 8},
    "t2":            {"actions": {"GET", "POST", "SUBMIT", "SHELL"}, "budget": 50},
    "t2-lowctx":     {"actions": {"GET", "POST", "SUBMIT", "SHELL"}, "budget": 50},
    "t2-shortthink": {"actions": {"GET", "POST", "SUBMIT", "SHELL"}, "budget": 50},
}

_HTTP_IN_SHELL = re.compile(
    r"""(?:medagentbench_cli\.py\s+(get|post)|curl\s+[^|;]*?)\s+["']?(https?://[^\s"']+)""",
    re.I,
)
_FINISH_IN_SHELL = re.compile(r"medagentbench_cli\.py\s+finish", re.I)


def _canon_url(url: str) -> str:
    base, _, q = url.partition("?")
    params = sorted(p for p in q.split("&") if p and not p.startswith("_format="))
    return base + ("?" + "&".join(params) if params else "")


def canonicalize_parity(transcript: list[dict]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for m in transcript:
        if m.get("role") != "agent":
            continue
        c = (m.get("content") or "").strip()
        if c.startswith("GET"):
            out.append(("GET", _canon_url(c[3:].strip().split()[0])))
        elif c.startswith("POST"):
            out.append(("POST", _canon_url(c.splitlines()[0][4:].strip())))
        elif c.startswith("FINISH"):
            out.append(("SUBMIT", ""))
    return out


def canonicalize_atif(steps: list[dict]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for s in steps:
        for tc in s.get("tool_calls") or []:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            args = fn.get("arguments") or tc.get("arguments") or {}
            if isinstance(args, str):
                cmd = args
            else:
                cmd = str(args.get("keystrokes") or args.get("command") or "")
            if not cmd.strip():
                continue
            if _FINISH_IN_SHELL.search(cmd):
                out.append(("SUBMIT", ""))
                continue
            m = _HTTP_IN_SHELL.search(cmd)
            if m:
                verb = (m.group(1) or ("POST" if " -d " in cmd or " --data" in cmd else "GET")).upper()
                out.append((verb, _canon_url(m.group(2))))
            else:
                out.append(("SHELL", cmd.strip()[:200]))
    return out


def cost(actions: list[tuple[str, str]]) -> int:
    return sum(1 for a in actions if a[0] != "SUBMIT")


def feasible_under(actions: list[Any], harness: str) -> bool:
    caps = HARNESS_CAPS.get(harness)
    if caps is None:
        return False  # unknown target: refuse, never guess
    verbs = {tuple(a)[0] for a in actions}
    if not verbs <= caps["actions"]:
        return False
    return cost([tuple(a) for a in actions]) <= caps["budget"]
