"""Synchronize authored records without deleting conversation-generated state."""
from __future__ import annotations

import json
import math

from data_ingestion import digest, validate_rows, warn_skipped
from embedding_text import PREFIX_FORMAT_VERSION, format_embedding_text

OWNER = "word_sources_v1"
CONFIG_KEY = "word_sources_embedding_config"


def chunks(value: str):
    words = value.split()
    for start in range(0, len(words), 50):
        yield " ".join(words[start:start + 100])


def to_chroma_entries(rows, fingerprint):
    result = {}
    occurrences = {}
    for row in rows:
        if row["category"] in {"pl_list", "ali_chat"}:
            continue
        record_id = row["record_id"]
        occurrence = occurrences.get(record_id, 0)
        occurrences[record_id] = occurrence + 1
        # Keep repeated rows, even if they carry the same source record ID.
        instance_id = record_id if occurrence == 0 else f"{record_id}-entry-{occurrence + 1}"
        for index, text in enumerate(chunks(row["value"])):
            metadata = {key: row[key] for key in ("name", "faction", "category", "entity_id", "entity_type", "source_path") if row[key]}
            if row.get("subcategory"):
                metadata["subcategory"] = row["subcategory"]
            if row.get("tags"):
                metadata["tags"] = row["tags"]
            metadata.update(type=row["entity_type"] if row["entity_type"] in {"character", "faction"} else "lore",
                            source_record_id=row["record_id"], importer=OWNER)
            text = format_embedding_text(text, metadata)
            metadata.update(content_fingerprint=digest(text), embedding_fingerprint=fingerprint)
            result[f"source-{instance_id}-{index}"] = (text, metadata)
    return result


def embedding_fingerprint(settings):
    config = settings.embedding_model
    return digest(json.dumps({"provider": config.provider, "model": config.model,
                              "base_url": config.base_url, "hf_provider": config.hf_provider,
                              "distance_space": settings.chroma.distance_space,
                              "prefix_format": PREFIX_FORMAT_VERSION}, sort_keys=True))


def put(collection, identifier, document, metadata, embedding):
    kwargs = dict(ids=[identifier], embeddings=[embedding])
    if document is not None:
        kwargs["documents"] = [document]
    if metadata:
        kwargs["metadatas"] = [metadata]
    collection.upsert(**kwargs)


def synchronize(rows, settings=None, *, db_reset=False, client=None, provider=None):
    """Synchronize authored records, or reset the collection using only current input.

    Prepare and validate vectors before writes. Failures propagate without rollback.
    Stop the application before syncing; a reset discards runtime records too.
    """
    # 1. Keep the existing row policy: warn and skip invalid/empty source rows.
    rows = validate_rows(rows)

    from ai import create_embedding_provider, get_ai_settings
    import chromadb

    # 2. Resolve the incoming configuration independently of the stored collection.
    settings = settings or get_ai_settings()
    if settings.chroma.distance_space not in {"cosine", "l2", "ip"}:
        raise ValueError(f"Unsupported distance metric: {settings.chroma.distance_space}")
    fingerprint = embedding_fingerprint(settings)
    if client is None:
        client = chromadb.PersistentClient(path=settings.chroma.path)

    collection = None
    collection_meta = {}
    existing = {}
    collection_exists = False
    if db_reset:
        # Only discover the name. Old documents, vectors, and configuration are irrelevant.
        collection_exists = any((item if isinstance(item, str) else item.name) == settings.chroma.collection
                                for item in client.list_collections())
    else:
        # 3. Normal synchronization needs existing records to compare and reuse vectors.
        collection = client.get_or_create_collection(
            settings.chroma.collection, metadata={"hnsw:space": settings.chroma.distance_space})
        collection_meta = dict(collection.metadata or {})
        current = collection.get(include=["documents", "metadatas"])
        existing = {identifier: (current["documents"][i], current["metadatas"][i] or {})
                    for i, identifier in enumerate(current["ids"])}

        # 4. Stored incompatibilities are fatal unless the caller explicitly requests a reset.
        known = {metadata.get("embedding_fingerprint") for _, metadata in existing.values()
                 if metadata.get("importer") == OWNER}
        known.discard(None)
        if collection_meta.get(CONFIG_KEY):
            known.add(collection_meta[CONFIG_KEY])
        metric = collection_meta.get("word_sources_distance_space", collection_meta.get("hnsw:space", "l2"))
        if (known and known != {fingerprint}) or metric != settings.chroma.distance_space:
            raise ValueError("Embedding configuration changed; use sync_data.py --db-reset")

    # 5. A reset derives the entire new collection from input, never from stored runtime state.
    desired = to_chroma_entries(rows, fingerprint)
    stale = {identifier for identifier, (_, metadata) in existing.items()
             if metadata.get("importer") == OWNER and identifier not in desired}
    updates = desired if db_reset else {
        identifier: value for identifier, value in desired.items() if existing.get(identifier) != value}

    # An explicit reset must still clear the collection when there are no embeddable rows.
    if not db_reset and not updates and not stale:
        return {"upserted": 0, "deleted": 0, "unchanged": len(desired), "skipped": 0, "reset": False}

    # 6. Generate every vector afresh for reset; otherwise reuse metadata-only changes.
    provider = provider or create_embedding_provider(settings.embedding_model)
    prepared = {}
    skipped = set()
    for identifier, (document, metadata) in updates.items():
        if not document or not document.strip():
            warn_skipped(identifier, "missing or empty source text")
            skipped.add(identifier)
            continue

        old = existing.get(identifier)
        if not db_reset and old and old[0] == document and old[1].get("embedding_fingerprint") == fingerprint:
            vector = collection.get(ids=[identifier], include=["embeddings"])["embeddings"][0]
            prepared[identifier] = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        else:
            vectors = provider.embed(document)
            if len(vectors) != 1 or not vectors[0]:
                raise ValueError(f"Embedding provider returned invalid vectors for {identifier}")
            prepared[identifier] = vectors[0]

    # 7. Incoming vectors must be valid and consistent even when resetting the database.
    for identifier, vector in prepared.items():
        if not vector or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                             or not math.isfinite(value) for value in vector):
            raise ValueError(f"Invalid embedding values for {identifier}")
    dimensions = {len(vector) for vector in prepared.values()}
    if len(dimensions) > 1:
        raise ValueError("Embedding provider returned inconsistent dimensions")

    if existing and prepared and not db_reset:
        prior_vector = collection.get(ids=[next(iter(existing))], include=["embeddings"])["embeddings"][0]
        if len(prior_vector) not in dimensions:
            raise ValueError("Embedding dimensions changed; use sync_data.py --db-reset")

    # 8. Only now clear the selected collection. Reset never inherits its old settings.
    target_meta = {**collection_meta, CONFIG_KEY: fingerprint,
                   "hnsw:space": settings.chroma.distance_space,
                   "word_sources_distance_space": settings.chroma.distance_space}
    if db_reset:
        if collection_exists:
            client.delete_collection(settings.chroma.collection)
        collection = client.create_collection(settings.chroma.collection, metadata=target_meta)

    # 9. Write the prepared input. Storage/provider errors remain fatal, with no rollback.
    for identifier, (document, metadata) in updates.items():
        if identifier in skipped:
            continue
        if not db_reset and identifier in existing:
            removed_keys = existing[identifier][1].keys() - metadata.keys()
            if removed_keys:
                # Chroma merges metadata; replace a record to clear removed keys.
                collection.delete(ids=[identifier])
        put(collection, identifier, document, metadata, prepared[identifier])

    # 10. Incremental synchronization removes stale authored chunks, then saves configuration.
    if not db_reset:
        if stale:
            collection.delete(ids=sorted(stale))
        collection.modify(metadata={key: value for key, value in target_meta.items() if key != "hnsw:space"})

    # Reset does not read old records, so its previous record count is intentionally unknown.
    return {"upserted": len(prepared), "deleted": None if db_reset else len(stale),
            "unchanged": len(desired.keys() - updates.keys()), "skipped": len(skipped), "reset": db_reset}
