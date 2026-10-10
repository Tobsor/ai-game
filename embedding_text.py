"""Shared presentation of raw document text for embedding and retrieval."""
import json
from enum import Enum
from typing import Any, Mapping

PREFIX_FORMAT_VERSION = "descriptive-metadata-v1"


def format_embedding_text(text: str, metadata: Mapping[str, Any] | None = None) -> str:
    """Prefix raw text once at ingestion; callers must not pass already formatted documents."""
    metadata = metadata or {}
    fields = [
        ("Name", metadata.get("name")),
        ("Type", metadata.get("entity_type") or metadata.get("type")),
        ("Faction", metadata.get("faction")),
        ("Category", metadata.get("category")),
        ("Subcategory", metadata.get("subcategory")),
    ]
    tags = metadata.get("tags")
    if isinstance(tags, str):
        try:
            decoded = json.loads(tags)
            if isinstance(decoded, list):
                tags = decoded
        except ValueError:
            pass
    if isinstance(tags, (list, tuple)):
        tags = ", ".join(str(tag).strip() for tag in tags if str(tag).strip())
    fields.append(("Tags", tags))

    lines = []
    for label, value in fields:
        if isinstance(value, Enum):
            value = value.value
        if value is not None and str(value).strip():
            lines.append(f"{label}: {str(value).strip()}")
    return "\n".join(lines) + "\n\n" + text if lines else text
