"""Harness protocols for BFCL single-turn: fc, json, and xml.

Each task gives the model a query and function definitions. The model replies
once, and the runner parses its calls and writes them as a JSON array to
`/app/result.json`. There is no agent loop, tool-result feedback, or persistent
scaffold state, so the harnesses differ only in output format.

All three protocols normalize to the same answer shape:
`[{function_name: {param: value}}]`, matching the object used by the grader.

Each harness has its own system prompt and action-format instructions rather
than sharing one prompt template with only the format swapped.
"""

from __future__ import annotations

import json
import re
from typing import Any

# ---------------------------------------------------------------------------
# Parsing the adapter's generated instruction.md
# ---------------------------------------------------------------------------

_FUNCS_PATTERN = re.compile(
    r"you can invoke\.\s*(\[.*?\])\s*\n\n## Output", re.DOTALL
)
_QUERY_PATTERN = re.compile(r"#\s*Task\s*\n+(.*?)\n+##\s*Available Functions", re.DOTALL)


def parse_instruction(instruction: str) -> tuple[str, list[dict[str, Any]]]:
    """Split the adapter's instruction.md into (query, function_defs).

    Returns an EMPTY function list for the "No functions available." tasks --
    4 live_irrelevance tasks ship that way, and their correct answer is []. Do
    not treat that as a parse failure (verified against the generated corpus:
    3637/3641 carry a JSON function list, the other 4 are exactly these).
    """
    query_match = _QUERY_PATTERN.search(instruction)
    query = query_match.group(1).strip() if query_match else instruction.strip()

    funcs_match = _FUNCS_PATTERN.search(instruction)
    if not funcs_match:
        return query, []
    try:
        funcs = json.loads(funcs_match.group(1))
    except json.JSONDecodeError:
        return query, []
    return query, funcs if isinstance(funcs, list) else []


# ---------------------------------------------------------------------------
# bfcl-fc: native OpenAI tool_calls
# ---------------------------------------------------------------------------

# BFCL's schema dialect -> JSON Schema. BFCL uses "dict" where JSON Schema says
# "object", "float" where it says "number", and carries Java/JS type names in
# the simple_java / simple_javascript categories. Survey of the full generated
# corpus (3641 tasks) found exactly these type strings:
#   string 21854, dict 9464, integer 4757, boolean 3115, float 1690,
#   array 959, any 199, String 115, tuple 66, Array 13, long 7, HashMap 7,
#   ArrayList 6, Boolean 4, double 1
_TYPE_MAP = {
    "dict": "object",
    "float": "number",
    "double": "number",
    "long": "integer",
    "tuple": "array",
    "any": None,  # drop the constraint entirely rather than guess
    "": None,  # 1 occurrence (bfcl-simple-javascript-23): empty type string
    # Java/JS spellings
    "String": "string",
    "Boolean": "boolean",
    "Array": "array",
    "ArrayList": "array",
    "HashMap": "object",
    "char": "string",  # 1 occurrence (bfcl-simple-java-72)
}

# Everything JSON Schema actually accepts. Anything not in this set after
# mapping gets DROPPED rather than passed through: an unknown "type" makes the
# tool schema invalid, and a strict server-side tool parser (vLLM with
# --tool-call-parser hermes, which is how we serve) can reject the whole
# request -- turning a schema quirk in one task into an unexplained trial
# failure. Dropping the constraint degrades gracefully instead: the model
# still sees the parameter and its description, just without a type.
_VALID_JSON_SCHEMA_TYPES = {
    "object",
    "array",
    "string",
    "integer",
    "number",
    "boolean",
    "null",
}

_ILLEGAL_TOOL_NAME_CHARS = re.compile(r"[^A-Za-z0-9_-]")


def sanitize_tool_name(name: str) -> str:
    """OpenAI tool names must match ^[A-Za-z0-9_-]{1,64}$.

    Verified against the full corpus: the ONLY illegal character present is
    "." (2178 occurrences across 1853 distinct names, affecting 1354/3637
    tasks), and the longest name is exactly 64 chars, so no truncation is ever
    needed. Crucially the adapter's own evaluate.py normalizes function names
    with the identical `name.replace(".", "_")` rule, so a call emitted under
    the sanitized name still matches ground truth -- and a corpus check found
    ZERO within-task collisions after normalization, so the mapping is
    unambiguous per task. If either of those facts changes upstream, the fc
    rung silently starts scoring 0 on dotted-name tasks: re-verify before
    trusting fc numbers after a BFCL version bump.
    """
    return _ILLEGAL_TOOL_NAME_CHARS.sub("_", name)


def _convert_schema(node: Any) -> Any:
    """Recursively rewrite BFCL parameter schemas into valid JSON Schema."""
    if isinstance(node, list):
        return [_convert_schema(v) for v in node]
    if not isinstance(node, dict):
        return node

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "type" and isinstance(value, str):
            mapped = _TYPE_MAP.get(value, value)
            if mapped in _VALID_JSON_SCHEMA_TYPES:
                out["type"] = mapped
            # else: unmapped/unknown -> drop the constraint (see comment above)
            continue
        out[key] = _convert_schema(value)
    return out


def to_openai_tools(funcs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Return (tool_schemas, sanitized_name -> original_name)."""
    tools: list[dict[str, Any]] = []
    name_map: dict[str, str] = {}
    for fn in funcs:
        original = fn.get("name", "")
        safe = sanitize_tool_name(original)
        name_map[safe] = original
        params = _convert_schema(fn.get("parameters", {}) or {})
        if not isinstance(params, dict) or "type" not in params:
            params = {"type": "object", "properties": {}} if not isinstance(params, dict) else {**params, "type": "object"}
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": safe,
                    "description": fn.get("description", "") or "",
                    "parameters": params,
                },
            }
        )
    return tools, name_map


FC_SYSTEM_PROMPT = """
You are a function-calling assistant operating in BFCL-FC mode. You are given a
user request and a set of callable tools, exposed to you through the native
tool-calling interface.

Decide which tool call or calls -- if any -- satisfy the request, and emit them
as native tool calls. Emit ALL required calls in a single response: some
requests need several calls in parallel, and some need none at all.

Rules:
- Only call a tool when the request genuinely requires it AND every required
  parameter is available. If no tool applies, or the request lacks a required
  parameter, make NO tool call and reply with a brief plain-text explanation.
- Never invent parameter values that the request does not supply.
- Do not describe the call in prose; make the call.
""".strip()


def build_fc_messages(query: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": FC_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]


def parse_fc_response(message: dict[str, Any], name_map: dict[str, str]) -> list[dict[str, Any]]:
    """Native tool_calls -> [{original_name: {args}}]."""
    calls = message.get("tool_calls") or []
    result: list[dict[str, Any]] = []
    for call in calls:
        fn = call.get("function") or {}
        safe = fn.get("name", "")
        original = name_map.get(safe, safe)
        raw_args = fn.get("arguments") or "{}"
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                continue
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            continue
        result.append({original: args})
    return result


# ---------------------------------------------------------------------------
# bfcl-json
# ---------------------------------------------------------------------------

JSON_SYSTEM_PROMPT = """
You are a function-selection assistant operating under the BFCL-JSON protocol.
You will be shown a user request and a catalogue of available functions in JSON.
You do not have a live tool interface: you SELECT calls and report them as text.

Reply with exactly one JSON object and nothing else -- no prose around it, no
markdown fences. It has two fields:

{"analysis": "<your reasoning>", "calls": [{"name": "<function name>", "arguments": {<parameters>}}]}

Set "calls" to a list with one entry per function call the request requires --
several entries when the request needs several calls, and an EMPTY list [] when
no available function applies or a required parameter is missing from the
request. Never invent parameter values that the request does not supply.

Copy function names EXACTLY as they appear in the catalogue, including any dots.

Available functions:
{function_catalogue}
""".strip()


def build_json_messages(query: str, funcs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    catalogue = json.dumps(funcs, indent=2) if funcs else "(no functions are available for this request)"
    system = JSON_SYSTEM_PROMPT.replace("{function_catalogue}", catalogue)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]


def _extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def _calls_to_answer(calls: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(calls, list):
        return out
    for call in calls:
        if not isinstance(call, dict):
            continue
        name = call.get("name")
        args = call.get("arguments")
        if isinstance(name, str) and isinstance(args, dict):
            out.append({name: args})
    return out


def parse_json_response(text: str) -> list[dict[str, Any]]:
    obj = _extract_json_object(text)
    if obj is None:
        return []
    return _calls_to_answer(obj.get("calls"))


# ---------------------------------------------------------------------------
# bfcl-xml (HELD OUT -- eval-only, never used for teacher traces or training)
# ---------------------------------------------------------------------------

XML_SYSTEM_PROMPT = """
You are a function-selection assistant operating under the BFCL-XML protocol.
You will be shown a user request and a catalogue of available functions. You
report your chosen calls using XML tags, not a live tool interface.

Reply using exactly this structure and nothing outside the tags:

<reasoning>why these calls, or why none apply</reasoning>
<calls>
<call name="function_name">{"parameter": "value"}</call>
</calls>

Include one <call> element per required function call. If no available function
applies to the request, or the request is missing a parameter the function
requires, emit an empty <calls></calls> block. Never invent parameter values
that the request does not supply.

Each <call> element's body must be a JSON object of that call's arguments, and
the name attribute must copy the catalogue name EXACTLY, including any dots.

Available functions:
{function_catalogue}
""".strip()

_XML_CALL_PATTERN = re.compile(
    r'<call\s+name="(?P<name>[^"]+)"\s*>(?P<args>.*?)</call>', re.DOTALL
)


def build_xml_messages(query: str, funcs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    catalogue = json.dumps(funcs, indent=2) if funcs else "(no functions are available for this request)"
    system = XML_SYSTEM_PROMPT.replace("{function_catalogue}", catalogue)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]


def parse_xml_response(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for match in _XML_CALL_PATTERN.finditer(text):
        args = _extract_json_object(match.group("args"))
        if args is None:
            stripped = match.group("args").strip()
            if stripped in ("", "{}"):
                args = {}
            else:
                continue
        out.append({match.group("name"): args})
    return out


PROTOCOLS = {
    "fc": {"native_tools": True},
    "json": {
        "native_tools": False,
        "build_messages": build_json_messages,
        "parse_response": parse_json_response,
    },
    "xml": {
        "native_tools": False,
        "build_messages": build_xml_messages,
        "parse_response": parse_xml_response,
    },
}
