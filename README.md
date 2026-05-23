# Agent Cog: A Memory-First Autonomous Agent Harness

This repository contains an implementation of an agent harness built from scratch in Python, utilizing an LLM gateway and MCP `stdio`. The agent is designed to execute multi-step goals with dynamic memory, artifact management, and robust reasoning.

## Setup & Execution

### Prerequisites
1. Python 3.11+
2. The `llm_gatewayV3` running in a separate terminal.

### Starting the Gateway
To start the gateway, open a terminal and run:
```bash
cd llm_gatewayV3
python main.py
```

### Running the Agent
Open a second terminal and start the interactive agent loop:
```bash
python agent6.py
```

Once the `agent6>` prompt appears, you can paste any of the example queries below and press Enter. The agent will autonomously decompose the goals, execute tool calls, manage artifacts, and synthesize a final response.

---

## Example Queries (Terminal Output)

Please find below the end-to-end terminal outputs for four distinct queries demonstrating the agent's capabilities in fetching, synthesis, persistence, and memory retrieval.

### Query A. Shannon Wikipedia (artifact attach test)
**Query:** `Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his birth date, death date, and three key contributions to information theory.`

<details>
<summary>Click to expand terminal output</summary>

```text
[PASTE TERMINAL OUTPUT FOR QUERY 1 HERE]
```
</details>


### Query B: Tokyo activities with weather constraint (multi-goal plus memory carryover)
**Query:** `Find 3 family-friendly things to do in Tokyo this weekend. Check Saturday's weather forecast there and tell me which one is most appropriate.`

<details>
<summary>Click to expand terminal output</summary>

```text
[PASTE TERMINAL OUTPUT FOR QUERY 2 HERE]
```
</details>


### Query C: Mom's birthday (durable memory across two runs)
**Query:** 

`Run 1: My mom's birthday is 15 May 2026. Remember that and give me a calendar reminder for two weeks before and on the day.` 

`Run 2: When is mom's birthday?`

<details>
<summary>Click to expand terminal output</summary>

```text
[PASTE TERMINAL OUTPUT FOR QUERY 3 HERE]
```
</details>


### Query D. Asyncio research (multi-source synthesis)
**Query:** `Search for 'Python asyncio best practices', read the top 3 results, and give me a short numbered list of the advice they agree on.`

<details>
<summary>Click to expand terminal output</summary>

```text
[PASTE TERMINAL OUTPUT FOR QUERY 4 HERE]
```
</details>

---

## Demo Video

[![▶️ Watch the End-to-End Demo on YouTube](https://img.youtube.com/vi/VaB4PNo3DLc/0.jpg)](https://www.youtube.com/watch?v=VaB4PNo3DLc)


---

## Architecture & Prompts

The core logic of the agent is divided into `perception.py` (goal tracking and artifact nomination) and `decision.py` (tool selection and answering). 

### Perception Prompt
*The system prompt used by the Perception module to evaluate goal state and nominate artifacts.*

<details>
<summary>Click to expand Perception Prompt</summary>

```text
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
```
</details>

### Decision Prompt
*The system prompt used by the Decision module to choose tools, persist data, or answer.*

<details>
<summary>Click to expand Decision Prompt</summary>

```text
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
```
</details>

### Validation JSON of PoP
*Proof of Performance (PoP) Validation JSON.*

Perception:
```json
{
  "explicit_reasoning": true,
  "structured_output": true,
  "tool_separation": true,
  "conversation_loop": true,
  "instructional_framing": true,
  "internal_self_checks": true,
  "reasoning_type_awareness": false,
  "fallbacks": false,
  "overall_clarity": "Excellent structure and operational precision. The prompt strongly supports deterministic multi-step reasoning, state tracking, and controlled agent planning. It clearly separates decomposition, completion tracking, and artifact attachment logic while minimizing drift through strict ordering and rules. It could improve further by explicitly labeling reasoning types (e.g. retrieval vs synthesis vs verification) and by defining fallback behavior for ambiguous history, missing artifacts, or uncertain completion states."
}
```

Decision:
```json
{
  "explicit_reasoning": true,
  "structured_output": true,
  "tool_separation": true,
  "conversation_loop": true,
  "instructional_framing": true,
  "internal_self_checks": true,
  "reasoning_type_awareness": true,
  "fallbacks": false,
  "overall_clarity": "Extremely strong agent-control prompt with clear operational semantics, deterministic branching, and strong safeguards against redundant actions. The decision tree tightly constrains behavior, clearly separates answering from tool execution, and provides robust handling of memory, history, artifacts, and retrieval depth. The prompt also demonstrates strong reasoning-type awareness by distinguishing retrieval, synthesis, persistence, and temporal reasoning. The main missing piece is explicit fallback/error behavior for unavailable tools, malformed artifacts, contradictory history, or uncertainty handling."
}
```