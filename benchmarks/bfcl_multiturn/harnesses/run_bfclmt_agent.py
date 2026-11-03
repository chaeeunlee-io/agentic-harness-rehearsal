"""BFCL multi-turn runner for the fc, json, code (`pycall`), and xml harnesses.

For each user turn, the runner repeatedly calls the model, executes extracted
BFCL API calls, and returns their results until the model emits no further calls
or reaches the 20-step limit.

Writes calls to `/logs/agent/mt_calls.json` for verification and the full
transcript to `/logs/agent/mt_transcript.json` for distillation. Model access is
configured through `OPENAI_API_BASE` and `OPENAI_API_KEY`.
"""
import argparse, json, os, re, sys, traceback

sys.path.insert(0, "/agent_lib")       # bfcl_env, uploaded by the agent at install
from bfcl_env.multi_turn_utils import execute_multi_turn_func_call  # noqa: E402
from litellm import completion  # noqa: E402

STEP_CAP = 20

def to_openai_tools(functions):
    tools, name_map = [], {}
    for f in functions:
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", f["name"])
        name_map[name] = f["name"]
        params = f.get("parameters") or {"type": "object", "properties": {}}
        if params.get("type") == "dict": params = {**params, "type": "object"}
        tools.append({"type": "function", "function": {
            "name": name, "description": f.get("description",""), "parameters": params}})
    return tools, name_map

def call_str(name, args: dict):
    return f"{name}({', '.join(f'{k}={v!r}' for k, v in args.items())})"

def extract_fc(msg, name_map):
    out = []
    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        try: args = json.loads(fn.get("arguments") or "{}")
        except Exception: continue
        name = name_map.get(fn.get("name"), fn.get("name"))
        if name: out.append(call_str(name, args))
    return out

CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
def extract_code(msg, _):
    m = CODE_RE.findall(msg.get("content") or "")
    if not m: return []
    return [ln.strip() for ln in m[-1].splitlines() if ln.strip() and not ln.strip().startswith("#")]

JSON_RE = re.compile(r"\{[^{}]*\"calls\"[^{}]*\[[^\]]*\][^{}]*\}", re.DOTALL)
def extract_json(msg, _):
    c = msg.get("content") or ""
    for m in reversed(JSON_RE.findall(c)):
        try:
            o = json.loads(m)
            if isinstance(o.get("calls"), list):
                return [x for x in o["calls"] if isinstance(x, str)]
        except Exception: continue
    return []

XML_CALL_RE = re.compile(r'<call\s+name="(?P<name>[^"]+)"\s*>(?P<args>.*?)</call>', re.DOTALL)

def extract_xml(msg, name_map):
    """<calls><call name="fn">{"a": 1}</call></calls> -> ['fn(a=1)'] (same call-string
    form every other envelope produces, so grading is identical across harnesses)."""
    c = msg.get("content") or ""
    out = []
    for m in XML_CALL_RE.finditer(c):
        name = name_map.get(m.group("name"), m.group("name"))
        body = (m.group("args") or "").strip()
        try:
            args = json.loads(body) if body else {}
            if not isinstance(args, dict): continue
        except Exception:
            continue
        out.append(f"{name}(" + ", ".join(f"{k}={v!r}" for k, v in args.items()) + ")")
    return out

PROMPTS = {
 "fc": "You control app APIs by CALLING the provided tools natively, one or more calls per "
       "turn. After each call you will see its result; read results before further calls. "
       "When the user's current request is fully done, reply with NO tool calls.",
 "code": "You control app APIs by writing python-style calls, one per line, inside one fenced "
       "```python block, e.g.:\n```python\ncd(folder='documents')\nmkdir(dir_name='temp')\n```\n"
       "After each block you will see each call's result; read results before further calls. "
       "When the user's current request is fully done, reply WITHOUT any code block.",
 "xml": "You operate app APIs through XML-tagged call blocks. Reply using exactly this "
       "structure and nothing outside the tags:\n<calls>\n<call name=\"function_name\">{\"parameter\": "
       "\"value\"}</call>\n</calls>\nEach <call> body is a JSON object of that call's arguments. "
       "After each reply you will see each call's result; read results before issuing further "
       "calls. When the user's current request is fully done, reply in plain prose with an "
       "empty <calls></calls> block omitted entirely.",
 "json": 'You control app APIs by replying with a JSON object {"calls": ["func(a=1)", ...]} '
       "containing python-style call strings. After each reply you will see each call's "
       "result; read results before further calls. When the user's current request is fully "
       'done, reply with plain text and NO {"calls": ...} object.',
}
EXTRACT = {"fc": extract_fc, "code": extract_code, "json": extract_json, "xml": extract_xml}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--harness", required=True, choices=["fc","json","code","xml"])
    ap.add_argument("--instruction-file", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()

    raw = open(args.instruction_file).read()
    payload = json.loads(raw.split("```json",1)[1].split("```",1)[0])
    entry = payload  # initial_config/involved_classes ride in the instruction payload
    tools, name_map = to_openai_tools(payload["functions"])

    docs_text = "\n".join(f"- {f['name']}: {f.get('description','')[:200]}" for f in payload["functions"])
    sysmsg = PROMPTS[args.harness]
    if args.harness != "fc":
        sysmsg += "\n\nAvailable functions:\n" + docs_text + \
          "\n\nFull signatures:\n" + json.dumps(payload["functions"])[:60000]

    messages = [{"role": "system", "content": sysmsg}]
    all_calls = []
    injected = payload.get("injected_functions") or {}
    os.makedirs("/logs/agent", exist_ok=True)
    for ti, turn_text in enumerate(payload["turns"]):
        if str(ti) in injected:  # miss_func: held-out functions become available now
            new_funcs = injected[str(ti)]
            new_tools, new_map = to_openai_tools(new_funcs)
            tools.extend(new_tools); name_map.update(new_map)
            note = "New function(s) now available: " + ", ".join(f["name"] for f in new_funcs) + "."
            if args.harness != "fc":
                note += "\n\nSignatures:\n" + json.dumps(new_funcs)
            turn_text = (note + "\n\n" + turn_text).strip()
        messages.append({"role": "user", "content": turn_text})
        turn_calls = []
        for _step in range(STEP_CAP):
            kw = {"temperature": args.temperature, "max_tokens": args.max_tokens}
            if args.harness == "fc": kw["tools"] = tools
            resp = completion(model=args.model, messages=messages, **kw)
            msg = resp.choices[0].message.model_dump()
            calls = EXTRACT[args.harness](msg, name_map)
            am = {"role": "assistant", "content": msg.get("content") or ""}
            if args.harness == "fc" and msg.get("tool_calls"):
                am["tool_calls"] = msg["tool_calls"]
            messages.append(am)
            if not calls: break
            turn_calls.append(calls)
            results, _ = execute_multi_turn_func_call(
                calls, entry.get("initial_config") or {}, entry.get("involved_classes") or [],
                "student", payload["id"], long_context=("long_context" in payload.get("category","")))
            if args.harness == "fc" and am.get("tool_calls"):
                for tc, c, r in zip(am["tool_calls"], calls, results):
                    messages.append({"role": "tool", "tool_call_id": tc.get("id") or "x",
                                     "content": str(r)[:2000]})
            else:
                fb = "\n".join(f"{c} -> {r}"[:2000] for c, r in zip(calls, results))
                messages.append({"role": "user", "content": "Results:\n" + fb})
        all_calls.append(turn_calls)
        json.dump(all_calls, open("/logs/agent/mt_calls.json","w"))
    json.dump({"harness": args.harness, "model": args.model, "messages": messages},
              open("/logs/agent/mt_transcript.json","w"))
    print("DONE turns:", len(all_calls))

if __name__ == "__main__":
    try: main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
