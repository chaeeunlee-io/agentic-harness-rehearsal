"""Harness-knob system: controlled perturbations of task representation.

Design: docs/eidf/harness-knobs-medagentbench-design.md. Two interception points:
offline task-dir variant generation (adapters/*/make_arm_variants.py) and pure
request/response kernels applied at an agent's tool layer (fhir.py for MedAgentBench).

Contract (inherited from pi-task-bind HARNESS_TRANSFORM_CONTRACT.md): transforms are
pure and deterministic; every knob's default value is the identity, so a stock arm
reproduces unperturbed behaviour exactly.
"""

from harbor.harness.spec import HarnessArm

__all__ = ["HarnessArm"]
