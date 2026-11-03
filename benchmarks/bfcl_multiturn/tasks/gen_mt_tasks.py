"""Generate harbor task dirs for BFCL multi-turn (v1: multi_turn_base).
Task dir: instruction.md (JSON payload: turns, per-class func docs minus excluded, meta),
tests/{evaluate.py, entry.json, ground_truth.json, bfcl_env/ (BFCL's API simulators + checker)}, task.toml.
Turn sequencing is enforced by the RUNNER (model never sees future turns)."""
import json, os, shutil, random

B = "${PROJECT_ROOT}/<user>_bfcl_runs/mt"
OUT = f"{B}/tasks"
CAT = "multi_turn_base"
DOC_FILE = {  # class -> func-doc jsonl
    "GorillaFileSystem": "gorilla_file_system.json", "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json", "MessageAPI": "message_api.json", "MathAPI": "math_api.json",
    "VehicleControlAPI": "vehicle_control.json", "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
}
docs_cache = {}
def docs_for(cls):
    if cls not in docs_cache:
        docs_cache[cls] = [json.loads(l) for l in open(f"{B}/multi_turn_func_doc/{DOC_FILE[cls]}")]
    return docs_cache[cls]

entries = [json.loads(l) for l in open(f"{B}/BFCL_v4_{CAT}.json")]
answers = {json.loads(l)["id"]: json.loads(l)["ground_truth"] for l in open(f"{B}/possible_answer/BFCL_v4_{CAT}.json")}
TEMPLATE_EVAL = open(f"{B}/eval/evaluate_template.py").read()

made = 0
for e in entries:
    tid = e["id"]  # e.g. multi_turn_base_0
    d = f"{OUT}/bfclmt-{tid.replace('multi_turn_','').replace('_','-')}"
    os.makedirs(f"{d}/tests", exist_ok=True)
    excluded = set(e.get("excluded_function") or [])
    funcs = []
    for cls in e["involved_classes"]:
        for f in docs_for(cls):
            if f["name"] not in excluded:
                funcs.append(f)
    payload = {"id": tid, "category": CAT, "involved_classes": e["involved_classes"],
               "initial_config": e.get("initial_config") or {},
               "turns": [t[0]["content"] for t in e["question"]], "functions": funcs}
    open(f"{d}/instruction.md","w").write(
        "BFCL multi-turn task. The runner supplies turns sequentially.\n\n```json\n"
        + json.dumps(payload) + "\n```\n")
    json.dump(e, open(f"{d}/tests/entry.json","w"))
    json.dump(answers[tid], open(f"{d}/tests/ground_truth.json","w"))
    open(f"{d}/tests/evaluate.py","w").write(TEMPLATE_EVAL)
    open(f"{d}/tests/test.sh","w").write(
        "#!/bin/bash\n# evaluate.py writes /logs/verifier/reward.txt itself (float); do not overwrite.\n"
        "cd /tests\npython3 /tests/evaluate.py\nexit 0\n")
    os.chmod(f"{d}/tests/test.sh", 0o755)
    if not os.path.exists(f"{d}/tests/bfcl_env"):
        shutil.copytree(f"{B}/bfcl_env", f"{d}/tests/bfcl_env")
    os.makedirs(f"{d}/environment", exist_ok=True)
    open(f"{d}/environment/Dockerfile","w").write(
        "FROM python:3.10-slim\nWORKDIR /app\n")
    open(f"{d}/task.toml","w").write(
        'schema_version = "1.0"\n\n[task]\nname = "bfclmt/' + os.path.basename(d) + '"\n'
        'keywords = ["bfcl-multi-turn"]\n\n[metadata]\nsource = "bfcl-mt"\nsource_id = "' + tid + '"\n\n'
        '[agent]\ntimeout_sec = 1200.0\n\n[verifier]\ntimeout_sec = 300.0\n\n'
        '[environment]\nbuild_timeout_sec = 3600.0\ncpus = 1\nmemory_mb = 4096\nstorage_mb = 10240\n')
    made += 1
print(f"generated {made} task dirs in {OUT}")
# split
random.seed(20260907)
ids = sorted(os.listdir(OUT))
random.shuffle(ids)
split = {"train": sorted(ids[:150]), "test": sorted(ids[150:])}
json.dump(split, open(f"{B}/mt_split.json","w"), indent=0)
print(f"split: train {len(split['train'])} / test {len(split['test'])}")
