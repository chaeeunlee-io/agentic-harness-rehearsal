"""BFCL multi-turn verifier (in-pod). Reads the runner's recorded per-turn call strings
(/logs/agent/mt_calls.json: [turn][step][call_str]), grades with the vendored
multi_turn_checker (state comparison vs ground truth), writes /logs/verifier/reward.txt."""
import json, os, sys, traceback

TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TESTS)  # makes `bfcl_env` (BFCL's API simulators + checker) importable

def main():
    reward = 0.0
    try:
        entry = json.load(open(f"{TESTS}/entry.json"))
        gt = json.load(open(f"{TESTS}/ground_truth.json"))
        calls_path = "/logs/agent/mt_calls.json"
        if os.path.exists(calls_path):
            model_calls = json.load(open(calls_path))
            # pad/truncate to ground-truth turn count (checker indexes by turn)
            while len(model_calls) < len(gt):
                model_calls.append([])
            model_calls = model_calls[:len(gt)]
            from bfcl_env.multi_turn_checker import multi_turn_checker
            r = multi_turn_checker(model_calls, gt, entry, entry.get("category","multi_turn_base"), "student")
            reward = 1.0 if r.get("valid") else 0.0
            json.dump(r, open("/logs/verifier/checker_detail.json","w"), default=str)
    except Exception:
        traceback.print_exc()
    os.makedirs("/logs/verifier", exist_ok=True)
    open("/logs/verifier/reward.txt","w").write(str(reward))
    print("reward:", reward)

if __name__ == "__main__":
    main()
