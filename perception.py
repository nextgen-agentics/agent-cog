"""
perception.py — Perception role for Agent6.

Perception is the orchestrator.  It runs every loop iteration.
One LLM call per invocation: provider="gemini", structured output → Observation.

Four obligations it fulfills:
  1. If prior_goals is empty → decompose the query into bounded goals.
  2. For each prior goal → examine history → mark done=True if satisfied.
  3. For the first unfinished goal → set attach_artifact_id if raw bytes needed.
  4. Preserve goal order. No reorder, no mid-list insert, no drops.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
# Ensure local code/ is FIRST so our schemas.py wins over llm_gatewayV3/schemas.py
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
sys.path.append(str(BASE_DIR / "llm_gatewayV3"))

from schemas import Goal, MemoryItem, Observation, GatewayResponseFormat  # noqa: E402


def _llm_client():
    from client import LLM  # type: ignore  # resolved via sys.path above
    return LLM()


# ---------------------------------------------------------------------------
# System prompt (constant; the four obligations as instructions)
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are the PERCEPTION module of an autonomous AI agent. You run once per loop iteration.
Your sole output is an updated goal list (Observation). You do not execute tasks.

## Context
You receive:
- query: the original user request (never changes across iterations)
- prior_goals: the goal list you emitted last iteration (empty on first call)
- history_tail: recent agent actions and answers (chronological, newest last)
- memory_hits: relevant facts retrieved from long-term memory

## Your four duties — execute in this exact order

### 1. DECOMPOSE  (only when prior_goals is empty)
Break the query into 1–5 atomic, independently-completable goals.
Each goal must:
  - Start with an action verb (Fetch, Find, Extract, Convert, List, Answer, Remember, ...)
  - Be testable: you can check from history or memory alone whether it is satisfied
  - Get a short stable id: g1, g2, g3 ...
Do NOT create goals for things the query does not ask for.
Do NOT split a simple factual lookup into multiple sub-goals (e.g. "identify X" + "inform user of X").
  If the answer is likely in memory_hits or can be given in one step, use a SINGLE goal: "Answer <question>".
If the query asks to remember or store something, include a goal to record it (e.g. "Remember <fact>").

### 2. MARK DONE  (check every prior goal every iteration)
For each goal in prior_goals, scan history_tail for evidence of completion.
Set done=true when ANY of these is true:
  a) An event with kind="answer" has goal_id matching this goal's id  → DONE
  b) An event with kind="action" produced a result that fully satisfies this goal  → DONE
Rules:
  - Once done=true, keep it true forever. Never flip back to false.
  - Apply this check BEFORE deciding on attachments (step 3).
  - A goal whose purpose is searching/retrieving a list of sources is done when a search
    action event exists for it in history.
  - A goal whose purpose is reading/fetching the FULL content of a source is done when
    a full-content retrieval action event exists for it in history.
  - A goal about "reading/fetching top N sources": count the full-content retrieval events
    in history. This goal is done ONLY when N separate retrieval events appear.
    Short snippet results (from a search) do NOT count as full-content retrievals.
  - A goal about extracting, summarising, or answering from content is done when an
    answer event exists for it.

### 3. ATTACH ARTIFACT  (only for the FIRST goal still done=false)

**HARD RULE: NEVER populate attach_artifact_ids on a goal with done=true.**
**Only the first open goal may receive artifact attachments. Maximum 3 ids.**

Choose the right pattern for the open goal:

**PATTERN A — Single artifact**  (most common)
  Use when the goal processes ONE piece of raw content:
    • An extraction or summarisation goal reading one fetched page
    • A reading goal that is the current step in a sequential read-then-answer chain
  Action: set attach_artifact_ids to a list containing exactly ONE artifact id.

**PATTERN B — Multiple artifacts** (synthesis goals with parallel prerequisites)
  Use when the goal genuinely needs raw data from MULTIPLE completed prerequisite goals
  simultaneously and those inputs were NOT already summarised into answer events.
  Apply this pattern when:
    • Two or more parallel prerequisite goals each produced a raw artifact, AND
    • Neither prerequisite produced an intermediate ANSWER event (i.e. the raw data
      was not yet summarised into history_tail), AND
    • The current goal requires combining those raw inputs to reach its answer.
  Action: set attach_artifact_ids to a list of the relevant artifact ids (max 3).
  CAUTION: If each prerequisite goal already produced an ANSWER event, those summaries
  are already in history_tail — do NOT attach the raw artifacts again. The synthesis
  goal can work from history alone.

**Procedure — apply after choosing the pattern:**
  a) Check HISTORY for action events that have an artifact_id field.
  b) Each action event carries a goal_id — use this to identify which goal produced it.
  c) For each needed artifact, confirm the producing goal is a direct prerequisite of
     the current open goal (match by goal_id, not by keyword similarity).
  d) If the open goal requires a fresh tool call, or all needed information is already
     in history answers or memory_hits, set attach_artifact_ids to [] (empty list).

### 4. PRESERVE ORDER
Return goals in the same order as prior_goals.
You may append new goals at the end only if the query requires steps not yet listed.
Never reorder, never drop, never insert in the middle.
"""


# ---------------------------------------------------------------------------
# Observation response_format — derived from Pydantic model via GatewayResponseFormat
# ---------------------------------------------------------------------------

_RESPONSE_FORMAT = GatewayResponseFormat.for_model(Observation, name="observation")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def observe(
    query: str,
    hits: list[MemoryItem],
    history: list[dict],
    prior_goals: list[Goal],
    run_id: str,
) -> Observation:
    """Run one Perception iteration. Returns an updated Observation.

    Calls Gemini (provider="gemini") with structured output.
    """
    llm = _llm_client()

    # Build a structured, clearly-labelled user message
    # Separate sections with headers so the LLM can parse each part independently.
    memory_section = "\n".join(
        f"  [{h.kind}] {h.descriptor}" + (f" (artifact: {h.artifact_id})" if h.artifact_id else "")
        for h in hits
    ) or "  (none)"

    history_lines = []
    for ev in history[-12:]:
        if ev.get("kind") == "answer":
            history_lines.append(f"  iter={ev['iter']} kind=answer goal_id={ev['goal_id']} text={ev['text'][:120]!r}")
        elif ev.get("kind") == "action":
            art = f" artifact_id={ev['artifact_id']}" if ev.get("artifact_id") else ""
            history_lines.append(f"  iter={ev['iter']} kind=action goal_id={ev['goal_id']} tool={ev['tool']}{art} result={ev.get('result_descriptor','')[:80]!r}")
    history_section = "\n".join(history_lines) or "  (none — first iteration)"

    prior_section = "\n".join(
        f"  {g['id']}: {g['text']} [{'DONE' if g['done'] else 'OPEN'}]"
        for g in [g.model_dump() for g in prior_goals]
    ) or "  (empty — decompose the query now)"

    # Build a concise artifact→goal lookup for step 3 of the attachment decision.
    # This lets Perception match by goal_id without parsing history.
    goal_id_to_text = {g.id: g.text for g in prior_goals}
    artifact_lines = []
    seen_art = set()
    for ev in history:
        if ev.get("kind") == "action" and ev.get("artifact_id"):
            art_id = ev["artifact_id"]
            if art_id not in seen_art:
                g_id = ev.get("goal_id", "?")
                g_text = goal_id_to_text.get(g_id, "")
                artifact_lines.append(f"  {art_id}  produced_by={g_id}  ({g_text[:60]})")
                seen_art.add(art_id)
    artifacts_section = "\n".join(artifact_lines) or "  (none yet)"

    user_msg = (
        f"USER QUERY: {query}\n\n"
        f"PRIOR GOALS:\n{prior_section}\n\n"
        f"ARTIFACTS (artifact_id → which goal produced it):\n{artifacts_section}\n\n"
        f"HISTORY (newest last):\n{history_section}\n\n"
        f"MEMORY HITS:\n{memory_section}"
    )

    try:
        resp = llm.chat(
            prompt=user_msg,
            system=_SYSTEM,
            provider="gemini",
            max_tokens=1024,
            temperature=0.2,
            response_format=_RESPONSE_FORMAT,
        )
        parsed = resp.get("parsed") or json.loads(resp.get("text", "{}"))
        return Observation.model_validate(parsed)

    except Exception as e:
        # Fallback: if Gemini fails and there are prior goals, preserve them unchanged.
        if prior_goals:
            return Observation(goals=prior_goals)
        # No prior goals and Gemini failed — create a single catch-all goal.
        return Observation(goals=[Goal(id="g1", text=query, done=False, attach_artifact_id=None)])
