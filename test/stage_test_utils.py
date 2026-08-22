import json
from pathlib import Path
from typing import Any


def create_test_embedding(db: Any, document: str) -> list[float]:
    embedding = db.get_embedding(document)
    if len(embedding) == 0:
        return []
    first_row = embedding[0]
    return [float(value) for value in first_row]


def create_test_embeddings(db: Any, entries: list[dict[str, Any]]) -> dict[str, Any]:
    normalized_entries = [entry for entry in entries if str(entry.get("document", "")).strip() != ""]
    if len(normalized_entries) == 0:
        return {"ids": [], "count": 0}

    ids = [str(entry["id"]) for entry in normalized_entries]
    documents = [str(entry["document"]).strip() for entry in normalized_entries]
    metadatas = [dict(entry.get("metadata") or {}) for entry in normalized_entries]
    embeddings = [create_test_embedding(db, document) for document in documents]

    db.db.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    return {"ids": ids, "count": len(ids)}


def load_vector_state_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Vector state config must be a JSON object: {path}")
    return payload


def extend_vector_state_from_config(db: Any, config: dict[str, Any]) -> dict[str, Any]:
    entries = config.get("embeddings", [])
    if not isinstance(entries, list):
        raise ValueError("Vector state config field 'embeddings' must be a list.")

    result = create_test_embeddings(db, entries)
    return {
        "config_name": str(config.get("name", "")),
        "base_snapshot": str(config.get("base_snapshot", "")),
        "inserted_ids": result["ids"],
        "inserted_count": result["count"],
    }


def save_vector_snapshot(db: Any, snapshot_path: str | Path, base_config_path: str | Path) -> dict[str, Any]:
    snapshot_file = Path(snapshot_path)
    snapshot_file.parent.mkdir(parents=True, exist_ok=True)

    collection_payload = db.db.get(include=["documents", "metadatas"])
    snapshot = {
        "base_config": str(base_config_path),
        "ids": collection_payload.get("ids", []),
        "documents": collection_payload.get("documents", []),
        "metadatas": collection_payload.get("metadatas", []),
    }
    with open(snapshot_file, "w", encoding="utf-8") as file_handle:
        json.dump(snapshot, file_handle, ensure_ascii=True, indent=2)

    return {
        "snapshot_path": str(snapshot_file),
        "document_count": len(snapshot.get("ids", [])),
    }
