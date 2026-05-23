"""
decision.py — Decision role for Agent6.

Decision receives ONE goal and must return either:
  • A plain-text final answer (DecisionOutput.answer is set), OR
  • Exactly one MCP tool call (DecisionOutput.tool_call is set).

It never sees other goals. It never narrates. It never picks more than one tool.

One LLM call per invocation via auto_route="decision" (gateway router picks
the provider based on prompt size; must support structured output).
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
# Ensure local code/ is FIRST so our schemas.py wins over llm_gatewayV3/schemas.py
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
sys.path.append(str(BASE_DIR / "llm_gatewayV3"))

from schemas import DecisionOutput, Goal, MemoryItem, ToolCall  # noqa: E402


def _llm_client():
    from client import LLM  # type: ignore
    return LLM()


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are the DECISION module of an autonomous AI agent.
You receive exactly ONE goal and must choose the single best next action.

## Your output — populate exactly one field, leave the other null:
  answer    : a complete, self-contained plain-text reply that fully satisfies the goal.
  tool_call : a single tool invocation to gather or persist the information needed.

Never populate both fields. Never leave both null.

## Decision tree — reason through these steps in order

STEP 1 — Do I already have enough information?
  Check in this order:
  a) MEMORY HITS: If a memory hit directly answers the goal (e.g. a stored fact or
     preference that the user previously asked you to remember), answer from memory now.
     Do not call a tool when the answer already exists in memory.
  b) HISTORY: If history_tail contains an action or answer that satisfies the goal, answer
     from that — do not repeat the same tool call.
  c) ATTACHED ARTIFACT: If artifact content is provided, read it carefully.
     Distinguish between two artifact types:
       • Search/snippet artifact: contains only short URL excerpts (title + brief summary).
         This does NOT contain the full text of each source. If the goal requires reading
         or extracting detail from the full source, you must retrieve the full content.
       • Full-content artifact: contains the complete text of a page or document.
         If such an artifact is attached and covers the goal, answer from it directly.
  If all checks pass → produce the answer now. Do not call a tool.
  If no  → go to step 2.

STEP 2 — Which single tool will get me closest to done?
  Pick ONE tool from available_tools. Fill in all required arguments precisely.
  Prefer specialised tools over general-purpose search when a dedicated tool exists
  (e.g. a time tool for current time, a conversion tool for currency exchange rates).
  Do not chain tools in one reply. One call per iteration.

  For a goal that requires reading N full sources:
    Check history_tail for previous full-content retrievals for those URLs.
    Retrieve the next source that has NOT yet been fetched in this run.

## Rules
- MEMORY FIRST: A goal whose answer is already in memory_hits must be answered from memory.
  Never call an external tool when the required information is already stored.
- ARTIFACT IDs ARE NOT FILES: When history_tail shows "artifact=art:N" after a tool call,
  that is an internal content-store handle, NOT a file path. You cannot access it with any
  file-read or directory tool. If that artifact's content is needed for the current goal,
  it will arrive pre-loaded in the ATTACHED ARTIFACT CONTENT section below — you never
  need to (and must never try to) retrieve artifact content yourself.
- PERSIST WITH TOOLS: If the goal requires creating or storing something — a reminder,
  note, record, or any information that must survive beyond this conversation — use a
  file-write tool to create it. Do not describe the action in text; actually perform it.
  Make reasonable filename and content choices; do not ask the user for more detail.
- NO QUESTIONS: Never ask the user for additional information or clarification.
  Make sensible assumptions and act. An autonomous agent cannot wait for user input.
- ARTIFACT CONTENT: One or more artifacts may be attached. Read ALL of them before deciding.
  If they collectively provide enough information to answer the goal, produce the answer now.
  Do not re-fetch a source whose full content is already in an attached artifact.
- FULL CONTENT vs. SNIPPETS: Short snippet/excerpt results do not satisfy a goal that
  requires reading, extracting detail from, or summarising the full text of a source.
  Retrieve the full content of each unread source one at a time.
- NO DUPLICATE CALLS: Do not call a tool for a URL or query already in history_tail.
- TEMPORAL AWARENESS: If the goal involves current time, dates, deadlines, "this weekend",
  "today", or any relative time expression, use the current date/time context from memory
  or call a time tool before making recommendations that depend on it.
- SELF-CONTAINED ANSWERS: The user sees ONLY your answer text. Do not reference artifacts,
  tool names, goal IDs, or internal agent state in the answer.
- VALID ARGUMENTS: All tool arguments must be valid JSON values (strings, numbers, booleans).
  Never use placeholder values.
- EXACTLY ONE OUTPUT: Populate either answer or tool_call — never both, never neither.
"""




# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def next_step(
    goal: Goal,
    hits: list[MemoryItem],
    attached: list[tuple[str, bytes]],
    history: list[dict],
    mcp_tools: list[dict],
) -> DecisionOutput:
    """Select the next action for one bounded goal.

    Parameters
    ----------
    goal:       The single goal to work on.
    hits:       Relevant memory items from Memory.read().
    attached:   List of (artifact_id, raw_bytes) fetched by the loop.
    history:    Full run history accumulated so far.
    mcp_tools:  MCP tool definitions — passed as native tools= to the gateway.
    """
    llm = _llm_client()

    # Build artifact attachments for the user message
    artifact_sections = []
    for art_id, raw_bytes in attached:
        try:
            text_content = raw_bytes.decode("utf-8")
            # Truncate very large artifacts to avoid HUGE-tier rejection
            if len(text_content) > 60_000:
                text_content = text_content[:30_000] + "\n...[truncated]...\n" + text_content[-10_000:]
            artifact_sections.append(
                f"--- ARTIFACT {art_id} ---\n{text_content}\n--- END ARTIFACT ---"
            )
        except UnicodeDecodeError:
            b64 = base64.b64encode(raw_bytes[:4096]).decode("ascii")
            artifact_sections.append(
                f"--- ARTIFACT {art_id} (binary/base64 preview) ---\n{b64}\n--- END ARTIFACT ---"
            )

    # Build the user message: goal → context (memory + history) → artifact.
    # Tools are passed natively via tools= below — not embedded as text.

    memory_lines = "\n".join(
        f"  [{h.kind}] {h.descriptor}" + (f" → artifact={h.artifact_id}" if h.artifact_id else "")
        for h in hits
    ) or "  (none)"

    # Extract URLs that have already been fetched this run, so Decision knows
    # exactly what remains without parsing artifact IDs.
    fetched_urls = [
        ev.get("arguments", {}).get("url", "")
        for ev in history
        if ev.get("kind") == "action" and ev.get("arguments", {}).get("url")
        and ev.get("tool", "") not in ("web_search",)
    ]
    fetched_section = (
        "\n".join(f"  {u}" for u in fetched_urls if u)
        or "  (none yet)"
    )

    history_lines = []
    for ev in history[-8:]:
        if ev.get("kind") == "answer":
            history_lines.append(f"  iter={ev['iter']} ANSWER goal={ev['goal_id']}: {ev['text'][:100]!r}")
        elif ev.get("kind") == "action":
            # Note: artifact=art:N is an internal content handle, NOT a file path.
            art = f" → stored as internal artifact (NOT a file path)" if ev.get("artifact_id") else ""
            history_lines.append(f"  iter={ev['iter']} TOOL {ev['tool']}({json.dumps(ev.get('arguments',{}))[:80]}){art}")
    history_section = "\n".join(history_lines) or "  (none — first iteration)"

    user_msg = (
        f"GOAL: {goal.text}\n"
        f"goal_id: {goal.id}\n\n"
        f"MEMORY (relevant facts from past runs):\n{memory_lines}\n\n"
        f"ALREADY FETCHED URLs this run (do not re-fetch these):\n{fetched_section}\n\n"
        f"HISTORY (this run, newest last):\n{history_section}"
    )

    if artifact_sections:
        user_msg += "\n\nATTACHED ARTIFACT CONTENT (read this — do not re-fetch):\n" + "\n\n".join(artifact_sections)

    try:
        resp = llm.chat(
            prompt=user_msg,
            system=_SYSTEM,
            auto_route="decision",
            max_tokens=1024,
            temperature=0.2,
            tools=mcp_tools,          # ← native tool definitions to the gateway
            tool_choice="auto",       # ← let the LLM decide: call a tool or answer
        )

        # Gateway returns either:
        #   tool_use:  {"tool_calls": [{"name": ..., "input": {...}}], "stop_reason": "tool_use"}
        #   end_turn:  {"text": "...", "stop_reason": "end_turn"}
        tool_calls = resp.get("tool_calls") or []
        answer_text = resp.get("text", "").strip()

        if tool_calls:
            # Take only the first tool call (Decision must stay atomic)
            tc = tool_calls[0]
            return DecisionOutput(
                answer=None,
                tool_call=ToolCall(
                    name=tc["name"],
                    arguments=tc.get("input") or tc.get("arguments", {}),
                ),
            )

        if answer_text:
            return DecisionOutput(answer=answer_text, tool_call=None)

        # Neither field populated — fallback
        return DecisionOutput(
            answer="I could not determine the next step.",
            tool_call=None,
        )

    except Exception as e:
        # Hard fallback: return the raw LLM text as an answer
        return DecisionOutput(
            answer=f"[Decision error: {e}]",
            tool_call=None,
        )
