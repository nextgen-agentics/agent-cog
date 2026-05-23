"""
agent6.py — LLM Agent Harness & Orchestration Loop (Session 6)

Wires Memory → Perception → Decision → Action over a goal-list loop.
No frameworks.  Basic Python + llm_gatewayV3 + mcp_server.py (MCP stdio).

Usage:
    python agent6.py               # interactive input loop
    python agent6.py "your query"  # single run from argv
"""
from __future__ import annotations

import asyncio
import json
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters  # type: ignore
from mcp.client.stdio import stdio_client  # type: ignore

# ── local roles ──────────────────────────────────────────────────────────────
# IMPORTANT: local code/ must come BEFORE llm_gatewayV3/ so our schemas.py
# is found instead of llm_gatewayV3/schemas.py.
_CODE_DIR = str(Path(__file__).parent)
_GATEWAY_DIR = str(Path(__file__).parent / "llm_gatewayV3")
if _CODE_DIR in sys.path:
    sys.path.remove(_CODE_DIR)
sys.path.insert(0, _CODE_DIR)
# Append gateway at the END so it doesn't shadow local names
if _GATEWAY_DIR not in sys.path:
    sys.path.append(_GATEWAY_DIR)

load_dotenv(Path(__file__).parent / ".env")

import action as action_mod
import decision as decision_mod
import perception as perception_mod
from memory import ArtifactStore, Memory
from schemas import Goal, Observation

# ── configuration ─────────────────────────────────────────────────────────────
GATEWAY_URL = "http://localhost:8101"
MCP_SERVER = str(Path(__file__).parent / "mcp_server.py")
MAX_ITERATIONS = 20

# Signals that indicate a query carries a durable fact/preference worth storing.
# If ANY of these appear, the query is a candidate for memory.remember().
_REMEMBER_SIGNALS = (
    "my ", "i ", "i'", "remember", "save", "note", "prefer",
    "birthday", "anniversary", "remind", "favourite", "favorite",
    "always", "never", "i am", "i'm", "my name",
)
# If ANY of these appear (and no remember signal), the query is a transient lookup.
_SKIP_SIGNALS = (
    "fetch ", "search ", "find ", "look up", "what is", "what are",
    "who is", "when is", "where is", "how do", "tell me", "show me",
    "give me", "list ",
)
# Tool names whose outcomes are durable and worth persisting across runs.
_DURABLE_TOOLS: frozenset[str] = frozenset({
    "create_file", "update_file", "edit_file",
    "currency_convert", "get_time",
})

# Shared singletons (process-scoped)
memory = Memory()
artifacts = ArtifactStore()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_gateway() -> None:
    """Verify the gateway is reachable. Print a warning if not."""
    try:
        r = httpx.get(f"{GATEWAY_URL}/v1/status", timeout=5)
        r.raise_for_status()
    except Exception as e:
        print(f"\n⚠  Gateway not reachable at {GATEWAY_URL}: {e}")
        print("   Start it with: cd llm_gatewayV3 && python main.py\n")


async def load_tools(session: ClientSession) -> list:
    """Fetch tool list from the MCP server."""
    result = await session.list_tools()
    return result.tools


def mcp_tools_for_decision(mcp_tools: list) -> list[dict]:
    """Convert MCP Tool objects into gateway-compatible tool dicts."""
    out = []
    for t in mcp_tools:
        schema = {}
        if hasattr(t, "inputSchema") and t.inputSchema:
            if hasattr(t.inputSchema, "model_dump"):
                schema = t.inputSchema.model_dump()
            elif isinstance(t.inputSchema, dict):
                schema = t.inputSchema
        out.append({
            "name": t.name,
            "description": getattr(t, "description", "") or "",
            "input_schema": schema,
        })
    return out


@asynccontextmanager
async def mcp_session():
    """Async context manager that spawns mcp_server.py and yields a ClientSession."""
    params = StdioServerParameters(
        command=sys.executable,
        args=[MCP_SERVER],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


# ─────────────────────────────────────────────────────────────────
# Rich terminal logger  — original structure + colour
# ─────────────────────────────────────────────────────────────────

from rich.console import Console
from rich.markup import escape as _esc

_con = Console(highlight=False)


# ── helpers ──────────────────────────────────────────────────────

def _p(markup: str) -> None:
    """Print one rich markup line."""
    _con.print(markup, end="\n")


# ── log functions (exact same structural layout as before) ────────

def _log_run_header(run_id: str, query: str) -> None:
    _p(f"\n[bold cyan]{'═' * 60}[/]")
    _p(f"[bold cyan] Agent6  run_id={run_id}[/]")
    _p(f"[bold white] Query: {_esc(query)}[/]")
    _p(f"[bold cyan]{'═' * 60}[/]")


def _log_iter_header(it: int) -> None:
    dashes = "─" * (50 - len(str(it)))
    _p(f"\n[bold cyan]─── iter {it} {dashes}[/]")


def _log_memory_read(hits: list) -> None:
    n = len(hits)
    _p(f"[bold green]\\[memory.read][/]   [green]{n}[/] hit{'s' if n != 1 else ''}")
    for h in hits:
        art = f"  [dim magenta]artifact={_esc(h.artifact_id)}[/]" if h.artifact_id else ""
        _p(f"[dim]{'':16}{_esc(h.descriptor[:90])}[/]{art}")


def _log_memory_write(kind: str, descriptor: str) -> None:
    _p(f"[green]\\[memory.{kind}][/]  [dim green]{_esc(descriptor[:80])}[/]")


def _log_perception(obs) -> None:
    for g in obs.goals:
        txt = _esc(g.text)
        if g.done:
            _p(f"[bold yellow]\\[perception][/]    [dim]\\[done] {txt}[/]")
        else:
            _p(f"[bold yellow]\\[perception][/]    [bold yellow]\\[open][/] [yellow]{txt}[/]")
            # Only show attach annotation for open goals; escape [ ] so Rich
            # doesn't interpret artifact IDs like "art:1" as markup tags.
            if g.attach_artifact_ids:
                ids_str = _esc(", ".join(g.attach_artifact_ids))
                _p(f"{'':18}[magenta]attach=\\[{ids_str}\\][/]")


def _log_attach(art_id: str, size: int) -> None:
    _p(f"[bold magenta]\\[attach][/]        [magenta]{_esc(art_id)}[/] [dim]({size:,} bytes)[/]")


def _log_decision_tool(tool_name: str, arguments: dict) -> None:
    args_str = _esc(json.dumps(arguments))
    _p(f"[bold blue]\\[decision][/]      [dim]TOOL_CALL:[/] [bold blue]{tool_name}[/][blue]({args_str})[/]")


def _log_decision_answer(text: str) -> None:
    preview = _esc(text[:150].replace("\n", " "))
    suffix  = "[dim]…[/]" if len(text) > 150 else ""
    _p(f"[bold blue]\\[decision][/]      [dim]ANSWER:[/] [bright_white]{preview}[/]{suffix}")


def _log_action_artifact(art_id: str, size: int, descriptor: str) -> None:
    _p(f"[bold cyan]\\[action][/]        [cyan]→[/] [dim]\\[artifact[/] [magenta]{_esc(art_id)}[/][dim],[/] [green]{size:,} bytes[/][dim]][/] [dim]preview: {_esc(descriptor[:60])}[/]")


def _log_action_result(descriptor: str) -> None:
    _p(f"[bold cyan]\\[action][/]        [cyan]→[/] [dim cyan]{_esc(descriptor[:120])}[/]")


def _log_done() -> None:
    _p(f"\n[bold green]\\[done] all goals satisfied[/]")


def _log_warn(msg: str) -> None:
    _p(f"\n[bold red]⚠  {_esc(msg)}[/]")


def _log_final(text: str) -> None:
    _p(f"\n[bold green]{'─' * 60}[/]")
    _p(f"[bold bright_white]FINAL: {_esc(text)}[/]")
    _p(f"[bold green]{'─' * 60}[/]\n")



# ─────────────────────────────────────────────────────────────────────────────
# Final answer synthesis
# ─────────────────────────────────────────────────────────────────────────────

def _synthesise_final_answer(
    query: str,
    history: list[dict],
    memory_hits: list | None = None,
) -> str:
    """One LLM call (provider=gemini) to synthesise all goal answers into a
    single coherent final answer for the user."""
    from client import LLM  # type: ignore

    # Collect all answer events
    answer_events = [e for e in history if e.get("kind") == "answer"]
    action_events = [e for e in history if e.get("kind") == "action"]

    context_parts = []
    for ev in answer_events:
        context_parts.append(f"[Goal {ev['goal_id']} answer]\n{ev['text']}")
    for ev in action_events:
        context_parts.append(
            f"[Tool: {ev['tool']}]\n{ev.get('result_descriptor', '')}"
        )

    # Safety net: if the loop exited with no history (e.g. all goals were
    # already satisfied by memory before Decision ran), use memory hits.
    if not context_parts and memory_hits:
        for h in memory_hits:
            context_parts.append(f"[Memory] {h.descriptor}")

    if not context_parts:
        return "The agent completed the run but produced no final answer."

    context = "\n\n".join(context_parts)

    prompt = (
        "You are writing the final response to a user. "
        "Below is the original question and all information the agent collected.\n\n"
        f"QUESTION: {query}\n\n"
        f"COLLECTED INFORMATION:\n{context}\n\n"
        "Write a direct, complete answer to the question above.\n"
        "Requirements:\n"
        "- Answer only what was asked. Do not add unsolicited information.\n"
        "- Quote specific values (numbers, dates, names, URLs) exactly as they appear in the collected information.\n"
        "- If calendar dates or reminders were requested, state them clearly and precisely (e.g. '1 May 2026').\n"
        "- Do not mention tools, goals, iterations, artifacts, memory, or any agent internals.\n"
        "- If multiple sub-questions were asked, answer each one in order with a clear label.\n"
        "- Keep it concise: 1–5 sentences unless the question demands a structured list or more detail."
    )

    try:
        llm = LLM()
        resp = llm.chat(prompt=prompt, provider="gemini", max_tokens=1024, temperature=0.3)
        return resp.get("text", "").strip()
    except Exception as e:
        # Fallback: join all answer texts
        if answer_events:
            return "\n\n".join(e["text"] for e in answer_events)
        return f"[synthesis failed: {e}]"


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestration loop
# ─────────────────────────────────────────────────────────────────────────────

async def run(query: str) -> str:
    """Execute the full agent loop for a user query. Returns the final answer."""
    ensure_gateway()

    run_id = uuid.uuid4().hex[:8]
    history: list[dict] = []
    prior_goals: list[Goal] = []

    _log_run_header(run_id, query)

    # Durable memory: only remember the query if it carries a fact, preference,
    # or an explicit instruction to store something.  Pure lookups are transient.
    _query_lower = query.lower()
    _has_remember_signal = any(sig in _query_lower for sig in _REMEMBER_SIGNALS)
    _has_skip_signal = any(sig in _query_lower for sig in _SKIP_SIGNALS)
    # Remember when there's a personal/storage signal AND no pure-lookup signal,
    # OR when both are present but the remember signal dominates (explicit "remember"/"save").
    _explicit_store = any(sig in _query_lower for sig in ("remember", "save", "note", "remind"))
    _should_remember = _has_remember_signal and (_explicit_store or not _has_skip_signal)
    if _should_remember:
        _log_memory_write("remembering query", query[:80])
        memory.remember(query, source="user_query", run_id=run_id)

    async with mcp_session() as session:
        mcp_tools = await load_tools(session)
        tools = mcp_tools_for_decision(mcp_tools)

        for it in range(1, MAX_ITERATIONS + 1):
            _log_iter_header(it)

            # ── Memory read ──────────────────────────────────────────────
            hits = memory.read(query, history)
            _log_memory_read(hits)

            # ── Perception ───────────────────────────────────────────────
            obs = perception_mod.observe(query, hits, history, prior_goals, run_id)
            # Safety: deterministically mark done based on history —
            # this ensures Perception's LLM failure to mark done doesn't
            # cause infinite looping.
            answered_goal_ids = {e["goal_id"] for e in history if e.get("kind") == "answer"}
            # Also check: if prior_goals were all single-goal and Decision answered, mark done
            if len(obs.goals) == 1 and history:
                last = history[-1]
                if last.get("kind") == "answer":
                    answered_goal_ids.add(obs.goals[0].id)
            for g in obs.goals:
                if g.id in answered_goal_ids:
                    g.done = True
            prior_goals = obs.goals
            _log_perception(obs)

            if obs.all_done:
                _log_done()
                break

            goal = obs.next_unfinished()
            if goal is None:
                break

            # ── Artifact attachment ──────────────────────────────────────
            # Collect all artifact ids the Perception module nominated for this goal.
            # Scale per-artifact truncation inversely with count so total context
            # stays bounded: 1→60K, 2→30K each, 3→20K each.
            attached: list[tuple[str, bytes]] = []
            art_ids = [aid for aid in goal.attach_artifact_ids if artifacts.exists(aid)]
            if art_ids:
                per_limit = max(20_000, 60_000 // len(art_ids))
                for aid in art_ids:
                    raw = artifacts.get_bytes(aid)
                    _log_attach(aid, len(raw))
                    # Trim individual artifacts to per_limit bytes (UTF-8 safe)
                    if len(raw) > per_limit:
                        raw = raw[:per_limit] + b"\n...[truncated]..."
                    attached.append((aid, raw))

            # ── Decision ─────────────────────────────────────────────────
            out = decision_mod.next_step(goal, hits, attached, history, tools)

            if out.is_answer:
                _log_decision_answer(out.answer)
                history.append({
                    "iter": it,
                    "kind": "answer",
                    "goal_id": goal.id,
                    "text": out.answer,
                })
                continue

            # ── Action (MCP dispatch) ─────────────────────────────────────
            _log_decision_tool(out.tool_call.name, out.tool_call.arguments)

            result_text, art_id = await action_mod.execute(session, out.tool_call, artifacts)

            art_size = None
            if art_id and artifacts.exists(art_id):
                art_size = artifacts.get_meta(art_id).size_bytes

            if art_id and art_size:
                _log_action_artifact(art_id, art_size, result_text)
            else:
                _log_action_result(result_text)

            # Only persist tool outcomes that are durable and reusable across runs.
            # Ephemeral results (web pages, search snippets, directory listings) live
            # only in the in-process history + ArtifactStore for the current run.
            if out.tool_call.name in _DURABLE_TOOLS:
                _log_memory_write("tool_outcome", f"{out.tool_call.name}() → {result_text[:60]}")
                memory.record_outcome(
                    tool_call=out.tool_call,
                    result_text=result_text,
                    artifact_id=art_id,
                    run_id=run_id,
                    goal_id=goal.id,
                )
            history.append({
                "iter": it,
                "kind": "action",
                "goal_id": goal.id,
                "tool": out.tool_call.name,
                "arguments": out.tool_call.arguments,
                "result_descriptor": result_text[:300],
                "artifact_id": art_id,
            })

        else:
            _log_warn(f"reached MAX_ITERATIONS ({MAX_ITERATIONS}) without completing all goals")

    # ── Final answer synthesis ────────────────────────────────────────────────
    # Pass the last known memory hits so synthesis can fall back to them when
    # the loop exited immediately (history empty, all goals done from memory).
    final = _synthesise_final_answer(query, history, memory_hits=hits)
    _log_final(final)
    return final


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

async def _interactive_loop() -> None:
    """Read queries from stdin until the user quits."""
    print("Agent6 — type a query and press Enter.  Type 'quit' or Ctrl-C to exit.\n")
    while True:
        try:
            query = input("agent6> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break
        await run(query)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single query from command line: python agent6.py "my query"
        asyncio.run(run(" ".join(sys.argv[1:])))
    else:
        # Interactive loop
        asyncio.run(_interactive_loop())
