"""
schemas.py — All Pydantic contracts for Agent6.


These models serve four purposes simultaneously:
  1. Runtime validator at construction time.
  2. JSON Schema sent to the LLM via response_format.
  3. Round-trip serialiser for persistence (model_dump / model_validate).
  4. Documentation of the inter-role contract.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class MemoryItem(BaseModel):
    """One record in the persistent memory store."""

    id: str
    kind: Literal["fact", "preference", "tool_outcome", "scratchpad"]
    keywords: list[str]
    descriptor: str          # one short human-readable line
    value: dict[str, Any]    # structured payload
    artifact_id: str | None  # handle into the artifact store, e.g. "art:09ff..."
    source: str
    run_id: str
    goal_id: str | None
    confidence: float = 1.0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MemoryClassification(BaseModel):
    """The LLM-fillable subset of MemoryItem returned by the remember() classifier.

    Excludes auto-set fields (id, artifact_id, source, run_id, goal_id, created_at)
    that the Memory.remember() method fills in programmatically after the LLM call.
    This is what the gateway response_format schema is built from.
    """

    kind: Literal["fact", "preference", "tool_outcome", "scratchpad"]
    keywords: list[str]
    descriptor: str
    value: dict[str, Any]
    confidence: float


class MemoryRanking(BaseModel):
    """Structured output for Memory.relevant() — ordered list of candidate indices.

    The LLM returns the indices of the most relevant memory candidates,
    most relevant first. Used with response_format to enforce JSON structure.
    """

    indices: list[int]


# ---------------------------------------------------------------------------
# Artifact store
# ---------------------------------------------------------------------------

class Artifact(BaseModel):
    """Metadata record for one artifact blob."""

    id: str            # "art:<sha256-prefix-12-chars>"
    content_type: str
    size_bytes: int
    source: str
    descriptor: str


# ---------------------------------------------------------------------------
# Goals & Observation (Perception output)
# ---------------------------------------------------------------------------

class Goal(BaseModel):
    """One bounded work item emitted by Perception."""

    id: str
    text: str                          # short imperative description
    done: bool = False
    attach_artifact_ids: list[str] = Field(default_factory=list)
    # ^ Set by Perception for the first open goal.  Max 3 entries.
    # Use a list with multiple ids only when the goal must combine raw data from
    # several parallel prerequisite goals that did not produce intermediate answer events.
    # For sequential read-then-summarise patterns, one id per read goal is correct;
    # the synthesis goal then works from answer events in history_tail.

    @property
    def attach_artifact_id(self) -> str | None:
        """Backwards-compatible accessor — returns the first id or None."""
        return self.attach_artifact_ids[0] if self.attach_artifact_ids else None


class Observation(BaseModel):
    """Full output of one Perception.observe() call."""

    goals: list[Goal]

    @property
    def all_done(self) -> bool:
        """True when every goal in the list is marked done."""
        return bool(self.goals) and all(g.done for g in self.goals)

    def next_unfinished(self) -> Goal | None:
        """Return the first goal that is not yet done, preserving order."""
        for g in self.goals:
            if not g.done:
                return g
        return None


# ---------------------------------------------------------------------------
# Decision output
# ---------------------------------------------------------------------------

class ToolCall(BaseModel):
    """A single MCP tool dispatch chosen by Decision."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class DecisionOutput(BaseModel):
    """Output of Decision.next_step().

    Exactly one of `answer` or `tool_call` must be populated.
    """

    answer: str | None = None
    tool_call: ToolCall | None = None

    @model_validator(mode="after")
    def exactly_one_populated(self) -> "DecisionOutput":
        has_answer = self.answer is not None
        has_tool = self.tool_call is not None
        if has_answer == has_tool:  # both set or neither set
            raise ValueError(
                "DecisionOutput must have exactly one of 'answer' or 'tool_call' set, "
                f"got answer={self.answer!r}, tool_call={self.tool_call!r}"
            )
        return self

    @property
    def is_answer(self) -> bool:
        return self.answer is not None


# ---------------------------------------------------------------------------
# Gateway ResponseFormat helper
# ---------------------------------------------------------------------------

class GatewayResponseFormat(BaseModel):
    """Mirrors llm_gatewayV3 ResponseFormat for building structured-output requests."""

    type: Literal["json_schema"] = "json_schema"
    schema_: dict[str, Any] = Field(alias="schema")
    name: str = "out"
    strict: bool = True

    model_config = {"populate_by_name": True}

    @classmethod
    def for_model(cls, model_cls: type[BaseModel], name: str = "out") -> dict[str, Any]:
        """Return a dict suitable for the gateway's `response_format` field."""
        return {
            "type": "json_schema",
            "schema": model_cls.model_json_schema(),
            "name": name,
            "strict": True,
        }
