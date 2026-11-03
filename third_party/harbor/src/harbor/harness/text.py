"""Pure terminal-text transforms — the terminus-2 runtime kernel.

Serves every terminus-2-driven benchmark (Terminal-Bench, SWE-bench, CodeContests…):
the agent's observations all pass through one seam
(`Terminus2._limit_output_length`), and this kernel runs there, BEFORE the agent's
native 10 KB middle-cut limiter — so harness cuts act on raw output and the native
limiter remains the final guard.

Contract: pure, deterministic, and identity at stock (`long_output_mode: native`
or `max_output_chars: 0` ⇒ input returned unchanged, so unperturbed runs are
byte-identical to pre-harness terminus-2).
"""

from __future__ import annotations

from harbor.harness.spec import HarnessArm


def apply_output(text: str, arm: HarnessArm) -> str:
    """Shape one terminal observation according to the arm. Stock arm: identity."""
    if arm.long_output_mode == "native" or arm.max_output_chars <= 0:
        return text
    if len(text) <= arm.max_output_chars:
        return text

    kept = arm.max_output_chars
    marker = (
        f"\n…[harness: output truncated — kept {arm.long_output_mode} "
        f"{kept:,} of {len(text):,} chars]"
    )
    if arm.long_output_mode == "head":
        return text[:kept] + marker
    # tail: the informative end survives; the beginning is lost.
    return marker.lstrip("\n") + "\n" + text[-kept:]
