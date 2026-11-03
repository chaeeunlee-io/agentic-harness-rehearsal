"""Protocols for the AppWorld harnesses.

All harnesses execute the same action: a Python program against the persistent
AppWorld REPL. They differ only in how that program is represented and requested.
`fc` therefore exposes a single `execute_python(code)` tool rather than one tool
per AppWorld API.

Each harness has its own system prompt and defines how to:
- add request-specific fields such as `tools`
- extract Python from the model response
- record assistant turns and execution results in the harness's native format

Recorded traces preserve the model's literal output, including fences, tags, or
tool calls, rather than normalizing it. This preserves the format used for
distillation.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

# --------------------------------------------------------------------------- #
# shared task framing (the APPWORLD rules that are true regardless of envelope)
# --------------------------------------------------------------------------- #
# NOTE: kept SHORT and factored so each harness prompt can embed it while its
# ENVELOPE INSTRUCTIONS stay unique. The envelope instructions are what must be
# distinct; the world rules are shared truth about AppWorld and identical by
# necessity (they describe the environment, not the interface).
_WORLD_RULES = """
AppWorld is a sandboxed world of everyday apps (Spotify, Amazon, Gmail, Venmo,
Todoist, ...) exposed through a Python API. Your code runs in a PERSISTENT
IPython session: variables, imports and logins from earlier turns are still
available on later turns. Call APIs as `apis.<app>.<api>(**kwargs)`.

=== OPERATE INTERACTIVELY, ONE SMALL STEP PER TURN (most important rule) ===
Do NOT write a whole multi-step program in one turn. AppWorld APIs must be
DISCOVERED and INSPECTED, not guessed -- guessing API names and response shapes
is the #1 cause of failure. Each turn, do the smallest next step, PRINT what you
get back, and read it on the next turn before depending on its structure:
  1. Look up an API before its first use:
       print(apis.api_docs.show_api_doc(app_name='spotify', api_name='search_artists'))
     (list an app's real api names with
       print(apis.api_docs.show_api_descriptions(app_name='spotify')))
  2. Call ONE api and PRINT the raw result -- inspect its exact shape (is it a
     list? a dict with 'items'? what keys?) BEFORE writing code that indexes it.
  3. Only after you have SEEN the real structure, write the code that uses it.
Never assume a response shape from memory of other APIs. Respect constraints in
the doc (e.g. page_limit max 20). Results are PAGINATED (page_index/page_limit)
-- loop pages until you have every item.

=== LOGGING IN + ACCESS TOKENS ===
username is the supervisor's OWN email; password is that app's entry in the
supervisor's saved passwords. There is NO get_username/get_password/
get_credentials. login() returns an access_token you MUST capture and pass to
EVERY authenticated call of that app (required access_token=... argument):

    profile = apis.supervisor.show_profile()             # has 'email', ...
    pwds = apis.supervisor.show_account_passwords()       # [{'account_name','password'}, ...]
    app_pw = next(p['password'] for p in pwds if p['account_name'] == 'spotify')
    token = apis.spotify.login(username=profile['email'], password=app_pw)['access_token']

=== WORKED EXAMPLE of the interactive loop (task: like every song by 'Adele') ===
  turn 1 (discover):  print(apis.api_docs.show_api_doc(app_name='spotify', api_name='search_songs'))
  turn 2 (inspect):   r = apis.spotify.search_songs(query='Adele', access_token=token, page_index=0)
                      print(type(r), r)          # SEE the real shape before using it
  turn 3 (act on what you saw):
                      # now that you know r is a list of dicts each with 'id','title','artist':
                      print([s for s in r if s['artist']=='Adele'][:3])
  turn 4 (paginate + do the work): loop pages, call apis.spotify.like_song(song_id=..., access_token=token)
  turn 5 (VERIFY): re-query and confirm the songs are now liked, THEN complete_task.

=== FINISHING ===
Work THOROUGHLY (tasks often need 15-30 API calls) and VERIFY the effect
(re-query and check) before finishing. Only then:
`apis.supervisor.complete_task(answer=<answer_or_None>, status="success")`
(pass `answer` only if the task asks a question); use `status="fail"` ONLY if
certain the task is impossible. Do NOT call complete_task right after an action
without verifying it worked. Output of your code (prints / return value /
traceback) is shown to you next turn -- never fabricate it.
""".strip()


# --------------------------------------------------------------------------- #
# code  (parity / native rung -- fenced ```python)
# --------------------------------------------------------------------------- #
CODE_SYSTEM_PROMPT = f"""
You are an autonomous coding agent in AppWorld. You act by WRITING PYTHON CODE.

{_WORLD_RULES}

On every turn, reply with your reasoning followed by EXACTLY ONE fenced Python
code block, and nothing after it:

```python
<your code>
```
""".strip()

_CODE_BLOCK = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)


def _extract_code_fenced(message: dict[str, Any]) -> str | None:
    text = message.get("content") or ""
    matches = _CODE_BLOCK.findall(text)
    return matches[-1].strip() if matches else None  # last block wins


# --------------------------------------------------------------------------- #
# fc  (function-calling -- single execute_python tool, native tool_calls)
# --------------------------------------------------------------------------- #
FC_SYSTEM_PROMPT = f"""
You are a function-calling agent in AppWorld. You act by CALLING the single
tool `execute_python`, whose one argument `code` is the Python to run this
turn. Do not write code in prose or in markdown fences -- put it in the tool
call's `code` argument and make the call natively.

{_WORLD_RULES}

Emit exactly one `execute_python` call per turn (put your reasoning in the
message content, then make the call).
""".strip()

EXECUTE_PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": (
            "Execute one blob of Python in the persistent AppWorld REPL session "
            "and return its printed output / return value / traceback. State "
            "persists across calls."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python source to execute this turn.",
                }
            },
            "required": ["code"],
        },
    },
}


def _extract_code_fc(message: dict[str, Any]) -> str | None:
    calls = message.get("tool_calls") or []
    for call in calls:
        fn = call.get("function") or {}
        if fn.get("name") != "execute_python":
            continue
        raw = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            return None
        code = args.get("code") if isinstance(args, dict) else None
        return code.strip() if isinstance(code, str) else None
    return None


# --------------------------------------------------------------------------- #
# xml  (HELD OUT -- eval-only, never used for teacher traces or training)  <code>...</code>
# --------------------------------------------------------------------------- #
XML_SYSTEM_PROMPT = f"""
You are a coding agent in AppWorld operating under the XML protocol. You act by
emitting Python wrapped in XML tags -- no markdown fences, no tool interface.

{_WORLD_RULES}

Reply using exactly this structure and nothing outside the tags:

<reasoning>your reasoning for this turn</reasoning>
<code>
your Python for this turn
</code>
""".strip()

_XML_CODE = re.compile(r"<code>\s*\n?(.*?)</code>", re.DOTALL)


def _extract_code_xml(message: dict[str, Any]) -> str | None:
    text = message.get("content") or ""
    matches = _XML_CODE.findall(text)
    return matches[-1].strip() if matches else None


# --------------------------------------------------------------------------- #
# history recording -- store the LITERAL emitted form (SFT training signal)
# --------------------------------------------------------------------------- #
def _record_assistant_text(history: list[dict[str, Any]], message: dict[str, Any]) -> None:
    history.append({"role": "assistant", "content": message.get("content") or ""})


def _record_observation_user(
    history: list[dict[str, Any]], message: dict[str, Any], obs: str
) -> None:
    history.append({"role": "user", "content": obs})


def _record_assistant_fc(history: list[dict[str, Any]], message: dict[str, Any]) -> None:
    """Keep the REAL tool_calls -- that literal structure is the fc training
    signal. Preserve content (reasoning / <think>) alongside the call."""
    row: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
    calls = message.get("tool_calls") or []
    if calls:
        row["tool_calls"] = calls
    history.append(row)


def _record_observation_fc(
    history: list[dict[str, Any]], message: dict[str, Any], obs: str
) -> None:
    calls = message.get("tool_calls") or []
    if calls:
        history.append(
            {"role": "tool", "tool_call_id": calls[0].get("id"), "content": obs}
        )
    else:  # model failed to emit a tool call -> plain feedback turn
        history.append({"role": "user", "content": obs})


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
class Protocol:
    def __init__(
        self,
        *,
        system_prompt: str,
        extract_code: Callable[[dict[str, Any]], "str | None"],
        record_assistant: Callable[[list, dict], None],
        record_observation: Callable[[list, dict, str], None],
        request_extra: dict[str, Any] | None = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.extract_code = extract_code
        self.record_assistant = record_assistant
        self.record_observation = record_observation
        self._request_extra = request_extra or {}

    def request_extra(self) -> dict[str, Any]:
        return dict(self._request_extra)


PROTOCOLS: dict[str, Protocol] = {
    "code": Protocol(
        system_prompt=CODE_SYSTEM_PROMPT,
        extract_code=_extract_code_fenced,
        record_assistant=_record_assistant_text,
        record_observation=_record_observation_user,
    ),
    "fc": Protocol(
        system_prompt=FC_SYSTEM_PROMPT,
        extract_code=_extract_code_fc,
        record_assistant=_record_assistant_fc,
        record_observation=_record_observation_fc,
        request_extra={"tools": [EXECUTE_PYTHON_TOOL], "tool_choice": "auto"},
    ),
    "xml": Protocol(  # HELD OUT -- eval-only
        system_prompt=XML_SYSTEM_PROMPT,
        extract_code=_extract_code_xml,
        record_assistant=_record_assistant_text,
        record_observation=_record_observation_user,
    ),
}


def get_protocol(harness: str) -> Protocol:
    if harness not in PROTOCOLS:
        raise ValueError(
            f"unknown appworld harness {harness!r}; known: {sorted(PROTOCOLS)}"
        )
    return PROTOCOLS[harness]
