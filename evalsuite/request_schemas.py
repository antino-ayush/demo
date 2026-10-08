"""Typed request schemas, one per module whose real service contract has
been confirmed against live traffic (a real run for open_gpt, a captured
curl for gpts -- see configs/modules/<module>.yaml's own comments for
what's actually been verified).

generator.py's ResponseGenerator.build_payload constructs the wire-format
JSON dict actually sent to each module's endpoint, either from a module's
YAML request_template or from the default Gemma-style contract. The
dataclasses here are that same contract's typed twin: every field named and
typed, with any field that only ever takes a small set of real values
pinned to an Enum instead of a bare str. Two modules can look almost
identical in their YAML (a question/session_id/turn_id/generation_id shape)
while actually meaning different things by a same-named field -- see
VarunaDocumentScope below -- so this module exists to make those
differences readable in code, not just by diffing two YAML files.

REQUEST_SCHEMAS registers each module's schema; build_payload uses it to
validate the payload it just built actually round-trips through the
matching schema before sending, catching config drift (e.g. a typo'd field
name in a module's request_template, or a value outside the enum) at
request time with a clear error instead of a confusing 422 from the real
service. A module with no entry here (every module except open_gpt and
gpts, as of writing) simply isn't validated -- its contract hasn't
been confirmed against a real request yet, so there's nothing true to
validate against.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class OpenGptModule(str, Enum):
    """open_gpt always sends this field empty -- unlike gpts, this
    service doesn't scope requests to a document set, so there's nothing to
    select. Kept as an enum (not a bare "") so it reads as deliberately
    empty, and so it's visibly a different field from gpts's
    same-named `module` (see VarunaDocumentScope)."""

    NONE = ""


@dataclass
class OpenGptRequest:
    """open_gpt's real request contract: the default shape
    ResponseGenerator.build_payload falls back to when a module config sets
    no request_template (see generator.py's module docstring) -- exactly
    what configs/modules/open_gpt.yaml relies on by not setting one."""

    question: str
    session_id: str
    offline_gpt_id: str
    turn_id: str
    generation_id: str
    module: OpenGptModule = OpenGptModule.NONE
    regenerate: bool = False
    turn_index: int = 0

    def to_payload(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "module": self.module.value,
            "session_id": self.session_id,
            "offline_gpt_id": self.offline_gpt_id,
            "regenerate": self.regenerate,
            "turn_id": self.turn_id,
            "turn_index": self.turn_index,
            "generation_id": self.generation_id,
        }

    @classmethod
    def from_payload(cls, data: dict[str, Any]) -> "OpenGptRequest":
        return cls(
            question=data["question"],
            session_id=data["session_id"],
            offline_gpt_id=data["offline_gpt_id"],
            turn_id=data["turn_id"],
            generation_id=data["generation_id"],
            module=OpenGptModule(data.get("module", "")),
            regenerate=data.get("regenerate", False),
            turn_index=data.get("turn_index", 0),
        )


class VarunaProfile(str, Enum):
    """Values observed on the wire against /api/chat/chatbot/query, one per
    real dashboard sharing that endpoint (see GPT_VARIANTS below). Add more
    members here once another profile is confirmed for real; don't guess at
    what else might exist."""

    ADMINISTRATION = "ADMINISTRATION"
    JANES_DOC = "JANES DOC"


class VarunaDocumentScope(str, Enum):
    """gpts's `module` field is NOT the same concept as this
    framework's own `module` (e.g. "gpts", "open_gpt" -- EvalRow.module
    / ModuleConfig.module) or open_gpt's always-empty `module` field: on the
    wire it selects which document set the RAG lookup searches, e.g.
    "ALL_DOCUMENTS". Renamed to `document_scope` here specifically so that
    collision doesn't read as "the same field" in code --
    VarunaGptRequest.to_payload()/from_payload() map it back to the real
    wire key ("module") when building/parsing the actual HTTP body. Only one
    value has been observed for real (the same live curl) -- add more once
    confirmed."""

    ALL_DOCUMENTS = "ALL_DOCUMENTS"


@dataclass
class VarunaGptRequest:
    """gpts's real request contract, confirmed from a live curl
    against /api/chat/chatbot/query -- what
    configs/modules/gpts.yaml's request_template builds. No
    offline_gpt_id or turn_index (both open_gpt-only); has profile and
    document_scope (both gpts-only, no open_gpt equivalent)."""

    profile: VarunaProfile
    question: str
    document_scope: VarunaDocumentScope
    session_id: str
    turn_id: str
    generation_id: str
    regenerate: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "profile": self.profile.value,
            "question": self.question,
            "module": self.document_scope.value,  # wire key is "module" -- see VarunaDocumentScope
            "session_id": self.session_id,
            "regenerate": self.regenerate,
            "turn_id": self.turn_id,
            "generation_id": self.generation_id,
        }

    @classmethod
    def from_payload(cls, data: dict[str, Any]) -> "VarunaGptRequest":
        return cls(
            profile=VarunaProfile(data["profile"]),
            question=data["question"],
            document_scope=VarunaDocumentScope(data["module"]),
            session_id=data["session_id"],
            turn_id=data["turn_id"],
            generation_id=data["generation_id"],
            regenerate=data.get("regenerate", False),
        )


#: module name -> its typed request schema, for modules whose real contract
#: has been confirmed (see this file's module docstring). build_payload
#: validates the payload it built against this before sending, when an
#: entry exists.
REQUEST_SCHEMAS: dict[str, type] = {
    "open_gpt": OpenGptRequest,
    "gpts": VarunaGptRequest,
}


#: module name -> {variant name -> request_template field overrides}, for
#: modules whose shared service contract (same endpoint/token/criteria/
#: response parsing) actually serves more than one real dashboard, differing
#: only in a couple of request field values. A module absent here has no
#: registered variants -- scripts/run_<module>.py's --gpt-variant flag has no
#: effect for it. Add a new dashboard's entry only once its curl is captured
#: for real -- don't guess at values (same rule as VarunaProfile above).
#:
#: The FIRST entry for a module is its default variant: scripts/_gen_module_scripts.py's
#: TEMPLATE relies on dict insertion order (Python 3.7+) to pick --gpt-variant's
#: default, so this entry must reproduce that module's own YAML request_template
#: verbatim (a no-op override) -- keeping un-suffixed output filenames and the
#: ResultsStore regression-baseline key unchanged for callers who never pass
#: --gpt-variant.
GPT_VARIANTS: dict[str, dict[str, dict[str, str]]] = {
    "gpts": {
        "varuna": {
            "profile": VarunaProfile.ADMINISTRATION.value,
            "module": VarunaDocumentScope.ALL_DOCUMENTS.value,
        },
        "janes_gpt": {
            "profile": VarunaProfile.JANES_DOC.value,
            "module": VarunaDocumentScope.ALL_DOCUMENTS.value,
        },
    },
}
