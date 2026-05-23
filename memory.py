"""
memory.py — Memory role for Agent6.

Two sub-systems live here:
  • ArtifactStore  — content-addressable blob store (state/artifacts/)
  • Memory         — typed persistent fact/preference/outcome store (state/memory.json)

LLM cost profile
  read()          → none  (keyword overlap, pure Python)
  filter()        → none  (structured list filter)
  relevant()      → one   (auto_route="memory"; only when keyword recall is weak)
  remember()      → one   (provider="gemini"; classifies free-form text)
  record_outcome()→ none  (kind forced to tool_outcome)
"""
from __future__ import annotations

import hashlib
import json
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# Path setup: local code/ MUST come before llm_gatewayV3/ so our schemas.py
# is found instead of llm_gatewayV3/schemas.py.
BASE_DIR = Path(__file__).parent
_GATEWAY_DIR = str(BASE_DIR / "llm_gatewayV3")
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if _GATEWAY_DIR not in sys.path:
    sys.path.append(_GATEWAY_DIR)

from schemas import Artifact, GatewayResponseFormat, MemoryClassification, MemoryItem, MemoryRanking  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

STATE_DIR = BASE_DIR / "state"
MEMORY_PATH = STATE_DIR / "memory.json"
ARTIFACTS_DIR = STATE_DIR / "artifacts"

STATE_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Gateway client (lazy import so schemas.py can be tested without gateway)
# ---------------------------------------------------------------------------

def _llm_client():
    """Return a fresh LLM gateway client."""
    from client import LLM  # type: ignore  # resolved via sys.path above
    return LLM()


# ===========================================================================
# ArtifactStore
# ===========================================================================

class ArtifactStore:
    """Content-addressable store for raw bytes produced by MCP tools.

    Handle format: "art:<N>"  (e.g. art:1, art:2, ...) — short, LLM-friendly.
    Filesystem:    art<N>.bin / art<N>.json (colon stripped for cross-platform safety).
    Dedup:         state/artifacts/index.json  {sha256_digest: handle}
    Counter:       state/artifacts/counter.json  (plain integer, incremented per blob)
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.counter_path = ARTIFACTS_DIR / "counter.json"
        self.index_path = ARTIFACTS_DIR / "index.json"

    def _get_index(self) -> dict[str, str]:
        if self.index_path.exists():
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        return {}

    def put(
        self,
        blob: bytes,
        *,
        content_type: str,
        source: str,
        descriptor: str,
    ) -> str:
        """Store blob and return its handle. Identical blobs deduplicate."""
        digest = hashlib.sha256(blob).hexdigest()
        
        with self._lock:
            index = self._get_index()
            if digest in index:
                return index[digest]
            
            # Increment counter
            count = 1
            if self.counter_path.exists():
                count = int(self.counter_path.read_text()) + 1
            self.counter_path.write_text(str(count))
            
            handle = f"art:{count}"
            index[digest] = handle
            self.index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
            
            stem = handle.replace(":", "")  # art:1 → art1 (filesystem-safe)
            bin_path = ARTIFACTS_DIR / f"{stem}.bin"
            meta_path = ARTIFACTS_DIR / f"{stem}.json"
            
            bin_path.write_bytes(blob)
            meta = Artifact(
                id=handle,
                content_type=content_type,
                size_bytes=len(blob),
                source=source,
                descriptor=descriptor,
            )
            meta_path.write_text(meta.model_dump_json(indent=2), encoding="utf-8")
            return handle

    @staticmethod
    def _stem(artifact_id: str) -> str:
        """art:1 → art1  (strip colon for filesystem paths)."""
        return artifact_id.replace(":", "")

    def get_bytes(self, artifact_id: str) -> bytes:
        """Return raw bytes for the given handle."""
        path = ARTIFACTS_DIR / f"{self._stem(artifact_id)}.bin"
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_id}")
        return path.read_bytes()

    def get_meta(self, artifact_id: str) -> Artifact:
        """Return metadata for the given handle."""
        path = ARTIFACTS_DIR / f"{self._stem(artifact_id)}.json"
        if not path.exists():
            raise FileNotFoundError(f"Artifact metadata not found: {artifact_id}")
        return Artifact.model_validate_json(path.read_text(encoding="utf-8"))

    def exists(self, artifact_id: str) -> bool:
        return (ARTIFACTS_DIR / f"{self._stem(artifact_id)}.bin").exists()


# ===========================================================================
# Memory
# ===========================================================================

_KEYWORD_STOP = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might must shall can of in on at to "
    "for with by from and or but not this that these those i you he "
    "she it we they what which who whom when where how".split()
)


def _tokenize(text: str) -> set[str]:
    tokens = set()
    for word in text.lower().split():
        w = "".join(c for c in word if c.isalnum())
        if w and w not in _KEYWORD_STOP and len(w) > 1:
            tokens.add(w)
    return tokens


class Memory:
    """Typed persistent memory service.

    All items live in state/memory.json.  The file is loaded on first access
    and written back after every mutation.  A threading.Lock serialises writes.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: list[MemoryItem] = []
        self._loaded = False

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._loaded:
            return
        if MEMORY_PATH.exists():
            try:
                raw = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
                self._items = [MemoryItem.model_validate(r) for r in raw]
            except Exception:
                self._items = []
        self._loaded = True

    def _save(self) -> None:
        data = [item.model_dump(mode="json") for item in self._items]
        MEMORY_PATH.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    # ------------------------------------------------------------------
    # Read methods (no LLM cost)
    # ------------------------------------------------------------------

    def read(
        self,
        query: str,
        history: list[dict],
        kinds: list[str] | None = None,
        top_k: int = 8,
    ) -> list[MemoryItem]:
        """Keyword-overlap search. Pure Python, no LLM call.

        Scores each item by the size of the intersection between the query
        tokens and the item's keywords + descriptor tokens.  Returns top-k.
        """
        with self._lock:
            self._load()

        q_tokens = _tokenize(query)
        # also include last few history entries in the query token set
        for event in history[-5:]:
            q_tokens |= _tokenize(str(event.get("tool", "")))
            q_tokens |= _tokenize(str(event.get("result_descriptor", "")))

        scored: list[tuple[int, MemoryItem]] = []
        for item in self._items:
            if kinds and item.kind not in kinds:
                continue
            item_tokens = set(item.keywords) | _tokenize(item.descriptor)
            score = len(q_tokens & item_tokens)
            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda x: (-x[0], x[1].created_at.isoformat()))
        return [item for _, item in scored[:top_k]]

    def filter(
        self,
        kinds: list[str] | None = None,
        goal_id: str | None = None,
        recent: int | None = None,
    ) -> list[MemoryItem]:
        """Structured filter. Pure Python, no LLM call."""
        with self._lock:
            self._load()
        results = list(self._items)
        if kinds:
            results = [i for i in results if i.kind in kinds]
        if goal_id:
            results = [i for i in results if i.goal_id == goal_id]
        if recent:
            results = sorted(results, key=lambda x: x.created_at, reverse=True)[:recent]
        return results

    def relevant(
        self,
        query: str,
        kinds: list[str] | None = None,
        top_k: int = 5,
    ) -> list[MemoryItem]:
        """LLM-scored relevance over a kind-filtered candidate pool.

        Called only when keyword recall returns fewer than 2 hits.
        Uses auto_route="memory" — the gateway router picks the provider.
        """
        candidates = self.filter(kinds=kinds)
        if not candidates:
            return []

        llm = _llm_client()
        n = len(candidates)
        candidate_text = "\n".join(
            f"[{i}] ({c.kind}) {c.descriptor}"
            for i, c in enumerate(candidates)
        )
        prompt = (
            f"You are ranking memory records by relevance to a query.\n\n"
            f"QUERY: {query}\n\n"
            f"CANDIDATES (index: kind, descriptor):\n{candidate_text}\n\n"
            f"Return the indices of the {top_k} most relevant candidates, "
            f"most relevant first, as JSON: {{\"indices\": [i, j, ...]}}\n"
            f"Only include indices 0–{n-1}. Return nothing else."
        )
        response_format = GatewayResponseFormat.for_model(MemoryRanking, name="memory_ranking")
        try:
            resp = llm.chat(
                prompt=prompt,
                auto_route="memory",
                max_tokens=64,
                temperature=0,
                response_format=response_format,
            )
            raw = resp.get("parsed") or json.loads(resp.get("text", "{}"))
            ranking = MemoryRanking.model_validate(raw)
            return [candidates[i] for i in ranking.indices if 0 <= i < len(candidates)][:top_k]
        except Exception:
            return candidates[:top_k]

    # ------------------------------------------------------------------
    # Write methods
    # ------------------------------------------------------------------

    def remember(
        self,
        raw_text: str,
        source: str,
        run_id: str,
        goal_id: str | None = None,
    ) -> MemoryItem:
        """Classify free-form text into a typed MemoryItem using Gemini.

        One gateway call: provider="gemini", response_format=MemoryItem schema.
        """
        llm = _llm_client()

        system = (
            "You are a memory classifier for an AI agent's long-term knowledge store.\n"
            "Given raw text, extract a structured memory record with these fields:\n\n"
            "kind — choose exactly one:\n"
            "  fact        : an observed truth about the world or the user (durable, run-independent)\n"
            "  preference  : a stated or inferred user preference, constraint, or reminder/calendar item\n"
            "  tool_outcome: the result of a specific tool call (include tool name + key result)\n"
            "  scratchpad  : temporary working note scoped to this run only\n\n"
            "keywords — 3 to 8 lowercase content words; omit stop words and punctuation.\n"
            "descriptor — one concise sentence (≤ 15 words) summarising what this record is.\n"
            "value — a flat or nested JSON object capturing the structured content.\n"
            "  For facts: {\"subject\": ..., \"predicate\": ..., \"object\": ...}\n"
            "  For preferences/reminders: {\"subject\": ..., \"detail\": ..., \"date\": <ISO date if applicable>}\n"
            "  For tool_outcome: {\"tool\": ..., \"args\": ..., \"result_summary\": ...}\n"
            "  For scratchpad: {\"note\": ...}\n"
            "confidence — float 0.0–1.0: how certain you are this classification is correct."
        )

        user_msg = f"Classify the following text into a memory record:\n\n<text>\n{raw_text}\n</text>"

        # Build response_format from the MemoryClassification Pydantic model
        response_format = GatewayResponseFormat.for_model(MemoryClassification, name="memory_item")

        try:
            resp = llm.chat(
                prompt=user_msg,
                system=system,
                provider="gemini",
                max_tokens=512,
                temperature=0.3,
                response_format=response_format,
            )
            raw = resp.get("parsed") or json.loads(resp.get("text", "{}"))
            classified = MemoryClassification.model_validate(raw)
            parsed = classified.model_dump()
        except Exception as e:
            # Fallback: store as scratchpad without classification
            parsed = {
                "kind": "scratchpad",
                "keywords": list(_tokenize(raw_text))[:8],
                "descriptor": raw_text[:120],
                "value": {"raw": raw_text},
                "confidence": 0.5,
            }

        item = MemoryItem(
            id=uuid.uuid4().hex[:12],
            kind=parsed.get("kind", "scratchpad"),
            keywords=parsed.get("keywords", []),
            descriptor=parsed.get("descriptor", raw_text[:120]),
            value=parsed.get("value", {"raw": raw_text}),
            artifact_id=None,
            source=source,
            run_id=run_id,
            goal_id=goal_id,
            confidence=float(parsed.get("confidence", 1.0)),
            created_at=datetime.utcnow(),
        )

        with self._lock:
            self._load()
            self._items.append(item)
            self._save()

        return item

    def record_outcome(
        self,
        tool_call: Any,  # schemas.ToolCall
        result_text: str,
        artifact_id: str | None,
        run_id: str,
        goal_id: str | None,
    ) -> MemoryItem:
        """Record an MCP dispatch result. No LLM call — kind is forced to tool_outcome."""
        # Keywords: tool name tokens + argument value tokens
        kw_tokens = _tokenize(tool_call.name)
        for v in tool_call.arguments.values():
            kw_tokens |= _tokenize(str(v))
        keywords = sorted(kw_tokens)[:12]

        descriptor = f"{tool_call.name}({', '.join(f'{k}={v!r}' for k, v in tool_call.arguments.items())}) → {result_text[:80]}"

        item = MemoryItem(
            id=uuid.uuid4().hex[:12],
            kind="tool_outcome",
            keywords=keywords,
            descriptor=descriptor,
            value={
                "tool": tool_call.name,
                "arguments": tool_call.arguments,
                "result_preview": result_text[:500],
                "artifact_id": artifact_id,
            },
            artifact_id=artifact_id,
            source="action",
            run_id=run_id,
            goal_id=goal_id,
            confidence=1.0,
            created_at=datetime.utcnow(),
        )

        with self._lock:
            self._load()
            self._items.append(item)
            self._save()

        return item
