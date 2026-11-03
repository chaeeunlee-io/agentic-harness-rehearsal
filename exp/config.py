"""Load configs/ and answer questions about benchmarks, harnesses, pools and models."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"


def _load(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def models() -> dict:
    return _load(CONFIGS / "models.yaml")


def model(key: str) -> dict:
    m = models()
    if key not in m:
        raise SystemExit(f"unknown model {key!r}; known: {', '.join(m)}")
    return m[key]


def model_source(key: str) -> str:
    """Where to load a model's weights from: <SHARED_MODEL_STORE>/<hf id> when that directory
    exists (a local copy, e.g. a cluster's shared model store), else the Hugging Face id."""
    hf = model(key)["hf"]
    store = os.environ.get("SHARED_MODEL_STORE")
    if store and (Path(store) / hf).is_dir():
        return str(Path(store) / hf)
    return hf


def benchmark_names() -> list[str]:
    return sorted(p.stem for p in (CONFIGS / "benchmarks").glob("*.yaml"))


def benchmark(name: str) -> "Benchmark":
    path = CONFIGS / "benchmarks" / f"{name}.yaml"
    if not path.exists():
        raise SystemExit(f"unknown benchmark {name!r}; known: {', '.join(benchmark_names())}")
    return Benchmark(_load(path))


def experiments() -> dict:
    return _load(CONFIGS / "experiments.yaml")


@dataclass
class Benchmark:
    cfg: dict

    @property
    def name(self) -> str:
        return self.cfg["name"]

    @property
    def cluster(self) -> int:
        return int(self.cfg["cluster"])

    # ---- harnesses -------------------------------------------------------
    def harnesses(self, spec="trained") -> list[str]:
        h = self.cfg["harnesses"]
        if isinstance(spec, list):
            return spec
        if "," in spec:  # a comma-separated list, e.g. "parity,t2,xml"
            return [h for part in spec.split(",") for h in self.harnesses(part)]
        if spec == "trained":
            return list(h["trained"])
        if spec == "heldout":
            return list(h["heldout"])
        if spec == "all":
            return list(h["trained"]) + list(h["heldout"])
        if spec in h["trained"] or spec in h["heldout"]:
            return [spec]
        raise SystemExit(f"{self.name}: unknown harness spec {spec!r}; harnesses: {h['trained'] + h['heldout']}")


    # ---- splits ----------------------------------------------------------
    def split(self, name: str = "test") -> dict:
        s = self.cfg["splits"]
        if name not in s:
            raise SystemExit(f"{self.name}: unknown split {name!r}; splits: {', '.join(s)}")
        return s[name]

    def split_tasks(self, name: str = "test") -> list[str] | None:
        """Task names of a split, or None when the split is a whole task directory."""
        sp = self.split(name)
        if sp.get("split_file"):
            data = json.load(open(ROOT / sp["split_file"]))
            return list(data[sp["split_key"]])
        if sp.get("task_list"):
            p = ROOT / sp["task_list"]
            if p.suffix == ".json":
                return list(json.load(open(p)))
            return [t for t in p.read_text().split() if t]
        return None

    def tier(self, split: str) -> str | None:
        return self.split(split).get("tier")

    # ---- pools -----------------------------------------------------------
    def pools(self) -> dict:
        return self.cfg["pools"]

    def pool(self, name: str) -> dict:
        p = self.pools()
        if name not in p:
            raise SystemExit(f"{self.name}: unknown pool {name!r}; pools: {', '.join(p)}")
        return p[name]

    # ---- eval ------------------------------------------------------------
    @property
    def eval(self) -> dict:
        return self.cfg["eval"]

    def layer(self, harness: str) -> str:
        layer = self.eval["layer"]
        if isinstance(layer, dict):
            return layer.get(harness, layer.get("default"))
        return layer

    def agent_spec(self, harness: str) -> dict:
        """Merged agent definition for one harness (benchmark default + per-harness override)."""
        base = dict(self.eval.get("agent") or {})
        over = dict((self.eval.get("per_harness") or {}).get(harness) or {})
        merged = {**base, **{k: v for k, v in over.items() if k not in ("kwargs", "env", "replace_kwargs")}}
        if over.get("replace_kwargs"):
            merged["kwargs"] = dict(over.get("kwargs") or {})
        else:
            merged["kwargs"] = {**(base.get("kwargs") or {}), **(over.get("kwargs") or {})}
        merged["env"] = dict(over["env"]) if "env" in over else dict(base.get("env") or {})
        return merged

    @property
    def rl(self) -> dict | None:
        return self.cfg.get("rl")


def rel(path) -> str:
    """Path relative to the repo root, for printing."""
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return str(path)
