#!/usr/bin/env python3
"""
Ingest markdown files from data/vector-sources/private/learningyard/ into Qdrant
collection ly-frontend-kb. Reads frontmatter for payload; chunks by section (≤512 tokens);
embeds and upserts with deterministic IDs for dedup.

Usage (from qdrant-mcp directory; requires deps: pip install -e . or uv run):
  uv run python scripts/ingest_learningyard.py
  python scripts/ingest_learningyard.py --sources-dir ../data/vector-sources/private/learningyard --collection ly-frontend-kb

Remote Qdrant (use QDRANT_URL; do not set QDRANT_LOCAL_PATH):
  $env:QDRANT_URL="http://localhost:6333"; $env:EMBEDDING_PROVIDER="fastembed"; python scripts/ingest_learningyard.py --sources-dir ../data/vector-sources/private/learningyard --collection ly-frontend-kb

Env: QDRANT_URL (remote server), QDRANT_LOCAL_PATH (only when QDRANT_URL is unset), COLLECTION_NAME (default: ly-frontend-kb), EMBEDDING_MODEL (optional). Set QDRANT_CHECK_COMPATIBILITY=false when using Qdrant server 1.12.x with client 1.17.x to skip version check warning.
"""
from __future__ import annotations

import argparse
import asyncio
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

# Suppress RequestsDependencyWarning (urllib3/chardet version mismatch with requests)
warnings.filterwarnings("ignore", module="requests")

from qdrant_client import models

from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.qdrant import Entry, QdrantConnector
from mcp_server_qdrant.settings import EmbeddingProviderSettings, METADATA_PATH, QdrantSettings


# Map frontmatter tool_source to Qdrant payload "source" per docs/50-rag-qdrant-vector-index.md
TOOL_SOURCE_TO_SOURCE = {
    "mcp-fetch": "fetch",
    "mcp_fetch": "fetch",
    "fetch": "fetch",
    "firecrawl": "firecrawl",
    "brave-search": "brave_search",
    "brave_search": "brave_search",
}


def _payload_source(tool_source: str | None) -> str:
    if not tool_source:
        return "fetch"
    v = (tool_source or "").strip().lower().replace("-", "_")
    return TOOL_SOURCE_TO_SOURCE.get(v, v if v in ("fetch", "firecrawl", "brave_search") else "fetch")


# Payload indexes for ly-frontend-kb (align with docs/50-rag-qdrant-vector-index.md § Metadata)
LY_FRONTEND_KB_INDEXES = {
    f"{METADATA_PATH}.source_url": models.PayloadSchemaType.KEYWORD,
    f"{METADATA_PATH}.source_file": models.PayloadSchemaType.KEYWORD,
    f"{METADATA_PATH}.source": models.PayloadSchemaType.KEYWORD,
    f"{METADATA_PATH}.category": models.PayloadSchemaType.KEYWORD,
    f"{METADATA_PATH}.subcategory": models.PayloadSchemaType.KEYWORD,
    f"{METADATA_PATH}.title": models.PayloadSchemaType.TEXT,
    f"{METADATA_PATH}.tool_source": models.PayloadSchemaType.KEYWORD,
    f"{METADATA_PATH}.project_id": models.PayloadSchemaType.KEYWORD,
    f"{METADATA_PATH}.visibility": models.PayloadSchemaType.KEYWORD,
    f"{METADATA_PATH}.fetched_at": models.PayloadSchemaType.KEYWORD,
    f"{METADATA_PATH}.ingested_at": models.PayloadSchemaType.KEYWORD,
    f"{METADATA_PATH}.chunk_index": models.PayloadSchemaType.INTEGER,
    f"{METADATA_PATH}.quality_score": models.PayloadSchemaType.FLOAT,
    f"{METADATA_PATH}.fact_checked": models.PayloadSchemaType.BOOL,
    f"{METADATA_PATH}.tech_stack": models.PayloadSchemaType.KEYWORD,
}


def parse_frontmatter(content: str) -> tuple[dict[str, str | int], str]:
    """Parse YAML-like frontmatter between first --- and second ---. Returns (metadata, body)."""
    if not content.strip().startswith("---"):
        return {}, content
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content
    meta: dict[str, str | int] = {}
    for line in parts[1].strip().split("\n"):
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if value.isdigit():
            meta[key] = int(value)
        else:
            meta[key] = value
    return meta, parts[2].lstrip("\n")


def chunk_text(text: str, max_chars: int = 2000, overlap_chars: int = 200) -> list[str]:
    """Split text into chunks by section (##) then by size with overlap."""
    sections = re.split(r"\n(?=##?\s)", text)
    chunks: list[str] = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        if len(sec) <= max_chars:
            chunks.append(sec)
            continue
        start = 0
        while start < len(sec):
            end = start + max_chars
            if end < len(sec):
                # Try to break at sentence or newline
                break_at = sec.rfind("\n", start, end + 1)
                if break_at > start:
                    end = break_at + 1
            chunks.append(sec[start:end])
            start = end - overlap_chars
            if start >= len(sec):
                break
    return chunks


def collect_entries(sources_dir: Path) -> list[Entry]:
    """Read all .md files, parse frontmatter, chunk, return list of Entry."""
    entries: list[Entry] = []
    for path in sorted(sources_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8", errors="replace")
        meta, body = parse_frontmatter(raw)
        chunks = chunk_text(body)
        for i, chunk in enumerate(chunks):
            payload = dict(meta)
            payload["chunk_index"] = i
            payload["source_url"] = meta.get("source_url", str(path))
            payload["source_file"] = path.name
            payload["source"] = _payload_source(meta.get("tool_source"))
            payload["project_id"] = meta.get("project_id", "learningyard")
            payload["visibility"] = meta.get("visibility", "private")
            # tech_stack from frontmatter (may be string or list-like string)
            if "tech_stack" in meta and meta["tech_stack"]:
                ts = meta["tech_stack"]
                payload["tech_stack"] = ts if isinstance(ts, str) else str(ts)
            # Optional per docs/50-rag-qdrant-vector-index.md
            if "quality_score" in meta:
                try:
                    payload["quality_score"] = float(meta["quality_score"])
                except (TypeError, ValueError):
                    pass
            if "fact_checked" in meta:
                v = meta["fact_checked"]
                payload["fact_checked"] = v in (True, "true", "1", "yes")
            entries.append(Entry(content=chunk, metadata=payload))
    return entries


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest LearningYard vector sources into Qdrant")
    parser.add_argument(
        "--sources-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "data" / "vector-sources" / "private" / "learningyard",
        help="Directory containing .md files with frontmatter",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Qdrant collection name (default: from env COLLECTION_NAME or ly-frontend-kb)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embed + upsert",
    )
    args = parser.parse_args()

    sources_dir = args.sources_dir.resolve()
    if not sources_dir.is_dir():
        raise SystemExit(f"Sources dir not found: {sources_dir}")

    entries = collect_entries(sources_dir)
    if not entries:
        print("No entries found.")
        return

    ingested_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for e in entries:
        e.metadata["ingested_at"] = ingested_at

    print(f"Collected {len(entries)} chunks from {sources_dir}")

    qdrant_settings = QdrantSettings()
    collection = args.collection or qdrant_settings.collection_name or "ly-frontend-kb"
    # Prefer QDRANT_URL if set, otherwise fallback to local path
    qdrant_url = qdrant_settings.location
    qdrant_local_path = None
    if not qdrant_url:
        qdrant_local_path = qdrant_settings.local_path or str(Path(__file__).resolve().parent.parent / "data" / "qdrant")

    embedding_settings = EmbeddingProviderSettings()
    provider = create_embedding_provider(embedding_settings)

    connector = QdrantConnector(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_settings.api_key,
        collection_name=collection,
        embedding_provider=provider,
        qdrant_local_path=qdrant_local_path,
        field_indexes=LY_FRONTEND_KB_INDEXES,
        check_compatibility=qdrant_settings.check_compatibility,
    )

    batch_size = args.batch_size
    for i in range(0, len(entries), batch_size):
        batch = entries[i : i + batch_size]
        await connector.store_many(batch, collection_name=collection, deterministic_ids=True)
        print(f"Upserted batch {i // batch_size + 1} ({len(batch)} points)")

    print(f"Done. Total points: {len(entries)} in collection {collection}")


if __name__ == "__main__":
    asyncio.run(main())
