"""
action.py — Action role for Agent6.

Pure async dispatch.  No LLM calls.  No logic beyond:
  1. Call the MCP tool via the live ClientSession.
  2. If the result is large (> ARTIFACT_THRESHOLD bytes), push to ArtifactStore.
  3. Return (descriptor, artifact_id_or_None).
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from mcp import ClientSession  # type: ignore

from schemas import ToolCall

if TYPE_CHECKING:
    from memory import ArtifactStore

# Results larger than this go to the artifact store instead of inline memory.
ARTIFACT_THRESHOLD_BYTES = 4096


async def execute(
    session: ClientSession,
    tool_call: ToolCall,
    artifacts: "ArtifactStore",
) -> tuple[str, str | None]:
    """Dispatch one MCP tool call.

    Returns
    -------
    descriptor  : A short human-readable summary of the result (≤ 300 chars).
    artifact_id : Handle to the artifact store if the result was large, else None.
    """
    result = await session.call_tool(tool_call.name, tool_call.arguments)

    # MCP returns a list of content blocks; normalise to a single string.
    raw = _extract_text(result)
    raw_bytes = raw.encode("utf-8")

    if len(raw_bytes) > ARTIFACT_THRESHOLD_BYTES:
        # Push the full payload to the artifact store; keep only a short descriptor.
        descriptor = f"{tool_call.name}() → {len(raw_bytes)} bytes"
        art_id = artifacts.put(
            raw_bytes,
            content_type="text/plain",
            source=f"mcp:{tool_call.name}",
            descriptor=descriptor,
        )
        return descriptor, art_id

    # Small result: return inline.
    descriptor = raw[:300]
    return descriptor, None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(mcp_result) -> str:
    """Flatten MCP CallToolResult content blocks into a single string."""
    if hasattr(mcp_result, "content"):
        parts = []
        for block in mcp_result.content:
            if hasattr(block, "text"):
                parts.append(block.text)
            elif isinstance(block, dict):
                parts.append(block.get("text", json.dumps(block)))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    # Fallback for raw dicts or strings
    if isinstance(mcp_result, dict):
        return json.dumps(mcp_result)
    return str(mcp_result)
