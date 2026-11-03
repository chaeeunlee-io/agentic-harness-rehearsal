"""Generate MedAgentBench rehearsal tasks for the t2 or fc harness.

Each rehearsal task replaces the original instruction with a mechanical procedure
that specifies which requests to make and which values to read. Solving it should
therefore require using the harness correctly rather than clinical reasoning. The
original verifier and solution are unchanged.

Source tasks come from training templates task2, task3, task4, and task8. Because
rehearsal tasks are derived from training tasks, evaluation on parity and fc is
used as a control for template-to-query leakage.

Usage:
    python gen_rehearsal.py --harness t2   # -> runs/tasks/medagentbench/rehearsal-t2/
    python gen_rehearsal.py --harness fc   # -> runs/tasks/medagentbench/rehearsal-fc/
"""
import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

TASKS = Path(__file__).resolve().parents[3] / "runs" / "tasks" / "medagentbench"
TEMPLATES = ("task2", "task3", "task4", "task8")


def load_logic(train_dir: Path):
    """The tasks' own medagentbench_logic module, for its default anchor time and referral note.

    It imports refsol_pkg from its own folder, so that folder goes on sys.path."""
    solution = next(train_dir.glob("*/solution/medagentbench_logic.py")).parent
    sys.path.insert(0, str(solution))
    import medagentbench_logic
    return medagentbench_logic


def bp_observation(md, mrn, logic):
    return {
        "resourceType": "Observation",
        "category": [{"coding": [{"system": "http://hl7.org/fhir/observation-category",
                                   "code": "vital-signs", "display": "Vital Signs"}]}],
        "code": {"text": "BP"},
        "effectiveDateTime": md.get("anchor_time") or logic.DEFAULT_ANCHOR_TIME,
        "status": "final",
        "valueString": "118/77 mmHg",
        "subject": {"reference": f"Patient/{mrn}"},
    }


def referral(md, mrn, logic):
    return {
        "resourceType": "ServiceRequest",
        "code": {"coding": [{"system": "http://snomed.info/sct", "code": "306181000000106"}]},
        "authoredOn": md.get("anchor_time") or logic.DEFAULT_ANCHOR_TIME,
        "status": "active",
        "intent": "order",
        "priority": "stat",
        "note": {"text": md.get("referral_note") or logic.DEFAULT_REFERRAL_NOTE},
        "subject": {"reference": f"Patient/{mrn}"},
    }


# Each harness's wording of the actions. {url}, {body}, {answer} are filled in per step.
WORDING = {
    "t2": {
        "get": 'Run: `medagentbench_cli.py get "{url}"`',
        "post": "Run: `medagentbench_cli.py post \"{url}\" '{body}'`",
        "finish": "Run: `medagentbench_cli.py finish '{answer}'`",
        "finish_int": "Run: `medagentbench_cli.py finish '[N]'` (N as a plain integer).",
        "branch": "If there is no such entry, run: `medagentbench_cli.py finish '[-1]'`\n"
                  "   Otherwise, run: `medagentbench_cli.py finish '[V]'`",
    },
    "fc": {
        "get": "Call the `get` tool with url `{url}`.",
        "post": "Call the `post` tool with url `{url}` and payload exactly:\n   `{body}`",
        "finish": "Call the `finish` tool with answer `{answer}`.",
        "finish_int": "Call the `finish` tool with answer `[N]` (N as a plain integer).",
        "branch": "If there is no such entry, call the `finish` tool with answer `[-1]`.\n"
                  "   Otherwise, call the `finish` tool with answer `[V]`.",
    },
}


def procedure(task_id, md, harness, logic):
    w = WORDING[harness]
    base = md["fhir_api_base"] if md["fhir_api_base"].endswith("/") else md["fhir_api_base"] + "/"
    mrn = md["eval_MRN"]
    done = w["finish"].format(answer='["DONE"]')
    template = task_id.split("_")[0]
    if template == "task2":  # patient age
        return (f"1. {w['get'].format(url=f'{base}Patient?identifier={mrn}&_format=json')}\n"
                "2. In the returned JSON, read the string at `entry[0].resource.birthDate` (format YYYY-MM-DD).\n"
                "3. Compute N = 2023 minus the year of that date; if (11, 13) is earlier than (month, day) of that date, subtract 1 more.\n"
                f"4. {w['finish_int']}")
    if template == "task3":  # record a blood pressure
        body = json.dumps(bp_observation(md, mrn, logic))
        return f"1. {w['post'].format(url=f'{base}Observation', body=body)}\n2. {done}"
    if template == "task4":  # latest magnesium in the last 24 hours
        anchor = md.get("anchor_time") or logic.DEFAULT_ANCHOR_TIME
        cutoff = (datetime.fromisoformat(anchor) - timedelta(hours=24)).isoformat()
        return (f"1. {w['get'].format(url=f'{base}Observation?patient={mrn}&code=MG&_count=5000&_format=json')}\n"
                f"2. In the returned JSON, consider only entries whose `resource.effectiveDateTime` is at or after `{cutoff}`. "
                "Among those, find the entry with the LATEST `resource.effectiveDateTime` and read V = its "
                "`resource.valueQuantity.value` (copy the number exactly as it appears).\n"
                f"3. {w['branch']}")
    if template == "task8":  # place a referral
        body = json.dumps(referral(md, mrn, logic))
        return f"1. {w['post'].format(url=f'{base}ServiceRequest', body=body)}\n2. {done}"
    raise ValueError(template)


def instruction(name, harness, steps, original):
    """The new instruction.md. Keeps the original's Environment / Required Output sections."""
    m = re.search(r"(## Environment.*)", original, re.S)
    tail = m.group(1).rstrip() if m else ""
    if harness == "t2":
        return f"""# Terminal Procedure Drill `{name}`

## Procedure
Execute the following steps exactly, in order. Everything you need is specified below; read command output carefully and transcribe values exactly.

{steps}

{tail}
"""
    # The fc agent reads only the "## Original Instruction" (the question) and "## Context"
    # sections of instruction.md, so the procedure must sit under those headers.
    return f"""# Tool-Call Procedure Drill `{name}`

## Original Instruction
This is a procedure drill. Execute the following steps exactly, in order, using the provided tools. Everything you need is specified below; read tool output carefully and transcribe values exactly.

{steps}

## Context
Everything required is specified in the instruction above; no additional context is needed.

{tail}
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--harness", choices=["t2", "fc"], required=True)
    p.add_argument("--train-tasks", type=Path, default=TASKS / "train", help="training task directories (exp tasks --split train)")
    p.add_argument("--out", type=Path, help="default runs/tasks/medagentbench/rehearsal-<harness>")
    args = p.parse_args()
    out = args.out or TASKS / f"rehearsal-{args.harness}"
    if out.exists():
        raise SystemExit(f"{out} exists; remove it to regenerate")
    logic = load_logic(args.train_tasks)
    out.mkdir(parents=True)
    n = 0
    for src in sorted(args.train_tasks.iterdir()):
        if src.name.split("_")[0] not in TEMPLATES:
            continue
        md = json.load(open(src / "tests" / "task_metadata.json"))
        shutil.copytree(src, out / src.name)
        original = (src / "instruction.md").read_text()
        steps = procedure(src.name, md, args.harness, logic)
        (out / src.name / "instruction.md").write_text(instruction(src.name, args.harness, steps, original))
        n += 1
    print(f"wrote {n} {args.harness} rehearsal tasks to {out}")


if __name__ == "__main__":
    main()
