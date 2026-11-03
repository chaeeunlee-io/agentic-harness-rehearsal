"""Generate harbor task dirs for BFCL multi-turn (v1: multi_turn_base).
Task dir: instruction.md (JSON payload: turns, per-class func docs minus excluded, meta),
tests/{evaluate.py, entry.json, ground_truth.json, bfcl_env/ (BFCL's API simulators + checker)}, task.toml.
Turn sequencing is enforced by the RUNNER (model never sees future turns)."""
import json, os, shutil, random

B = "${PROJECT_ROOT}/<user>_bfcl_runs/mt"
OUT = f"{B}/tasks"
CATS = ["multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context"]
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

TEMPLATE_EVAL = open(f"{B}/eval/evaluate_template.py").read()

made = 0
new_dirs = {}
for CAT in CATS:
  entries = [json.loads(l) for l in open(f"{B}/BFCL_v4_{CAT}.json")]
  answers = {json.loads(l)["id"]: json.loads(l)["ground_truth"] for l in open(f"{B}/possible_answer/BFCL_v4_{CAT}.json")}
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
      missed = e.get("missed_function") or {}
      missed_names = {n for v in missed.values() for n in v}
      avail = [f for f in funcs if f["name"] not in missed_names]
      injected = {ti: [f for f in funcs if f["name"] in set(names)] for ti, names in missed.items()}
      payload = {"id": tid, "category": CAT, "involved_classes": e["involved_classes"],
                 "initial_config": e.get("initial_config") or {},
                 "turns": [(t[0]["content"] if t else "") for t in e["question"]], "functions": avail}
      if injected:
          payload["injected_functions"] = injected
      new_dirs.setdefault(CAT, []).append(os.path.basename(d))
      e = dict(e); e["category"] = CAT
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
# per-category splits (150/50, same seed protocol); base split stays in mt_split.json
import copy
random.seed(20260907)
ext_split = {}
for CAT, dirs in new_dirs.items():
    ds = sorted(dirs); random.shuffle(ds)
    ext_split[CAT] = {"train": sorted(ds[:150]), "test": sorted(ds[150:])}
    print(f"{CAT}: train {len(ext_split[CAT]['train'])} / test {len(ext_split[CAT]['test'])}")
base = json.load(open(f"{B}/mt_split.json"))
ext_split["multi_turn_base"] = base
json.dump(ext_split, open(f"{B}/mt_split_ext.json","w"), indent=0)
print("wrote mt_split_ext.json")
