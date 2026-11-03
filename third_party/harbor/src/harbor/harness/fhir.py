"""Pure FHIR request/response transforms — the MedAgentBench runtime kernel.

Contract: every function is pure and deterministic (no I/O, no clock, no RNG), and
returns its input unchanged when the relevant knob is at its stock default. The
agent's tool layer is the only caller. Transform order for responses mirrors the
canonical order from the pi/verl kernel: content edits first, truncation last, so
truncation acts on final content.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from harbor.harness.spec import HarnessArm

# Fields that are FHIR envelope/bloat, never the clinical payload. Deliberately a
# blacklist: an over-eager whitelist could delete a field some task needs
# (e.g. Patient.identifier carries the MRN for category-1 lookups).
_TRIM_FIELDS = ("meta", "text", "extension", "contained")

_SORT_PARAM_RE = re.compile(r"(?:(?<=[?&]))_sort=[^&]*&?")
_CODE_PARAM_RE = re.compile(r"(?:[?&])code=([^&]*)")


@dataclass(frozen=True)
class RequestPlan:
    """What the tool layer should do with one outgoing GET.

    Either forward `url` to the server, or skip the server entirely and hand the
    agent `synthetic_response` (used when the knobbed "server" would answer
    differently from the real one, e.g. an empty bundle for a bare code).
    """

    url: str
    synthetic_response: str | None = None


def apply_request(url: str, arm: HarnessArm) -> RequestPlan:
    """Transform an outgoing GET URL according to the arm. Stock arm: identity."""
    out = url

    if arm.sort_support == "ignored":
        out = _strip_sort_param(out)

    if arm.bundle_count > 0 and "_count=" not in out:
        out += ("&" if "?" in out else "?") + f"_count={arm.bundle_count}"

    if arm.code_system_form == "qualified":
        code_match = _CODE_PARAM_RE.search(out)
        if code_match is not None:
            code_value = code_match.group(1)
            if "|" in code_value or "%7C" in code_value or "%7c" in code_value:
                # Agent used the qualified form the knobbed server "wants".
                # The real server only accepts bare codes, so translate back:
                # system|code -> code (on the raw, still-encoded value).
                bare = re.split(r"\||%7C|%7c", code_value)[-1]
                out = out.replace("code=" + code_value, "code=" + bare, 1)
            else:
                # Bare code under a qualified-only server: syntactically fine,
                # semantically no results — indistinguishable from "no labs exist".
                return RequestPlan(url=out, synthetic_response=empty_bundle_json())

    return RequestPlan(url=out)


def apply_response(text: str, arm: HarnessArm) -> str:
    """Transform a GET response body according to the arm. Stock arm: identity."""
    out = text

    needs_json_edit = arm.response_verbosity == "trimmed" or arm.link_stripped
    if needs_json_edit:
        parsed = _try_parse_json(out)
        if parsed is not None:
            if arm.response_verbosity == "trimmed":
                parsed = _drop_fields(parsed)
            if arm.link_stripped:
                parsed = _strip_next_links(parsed)
            out = json.dumps(parsed)

    if arm.max_fhir_chars > 0 and len(out) > arm.max_fhir_chars:
        out = out[: arm.max_fhir_chars] + " …[response truncated]"

    return out


def empty_bundle_json() -> str:
    """A syntactically valid, semantically empty FHIR searchset bundle."""
    return json.dumps(
        {"resourceType": "Bundle", "type": "searchset", "total": 0, "entry": []}
    )


def _strip_sort_param(url: str) -> str:
    out = _SORT_PARAM_RE.sub("", url)
    return out.rstrip("&?") if out != url else url


def _try_parse_json(text: str):
    try:
        return json.loads(text)
    except ValueError:  # includes JSONDecodeError; non-JSON responses pass through
        return None


def _drop_fields(node):
    """Recursively remove envelope/bloat fields from a parsed FHIR structure."""
    if isinstance(node, dict):
        return {
            key: _drop_fields(value)
            for key, value in node.items()
            if key not in _TRIM_FIELDS
        }
    if isinstance(node, list):
        return [_drop_fields(item) for item in node]
    return node


def _strip_next_links(parsed):
    """Remove Bundle.link entries with relation 'next' (page 1 poses as everything)."""
    if isinstance(parsed, dict) and isinstance(parsed.get("link"), list):
        parsed = dict(parsed)
        parsed["link"] = [
            entry
            for entry in parsed["link"]
            if not (isinstance(entry, dict) and entry.get("relation") == "next")
        ]
    return parsed
