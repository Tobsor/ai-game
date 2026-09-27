"""Controlled evidence for stage roleplay evaluations; no hidden live retrieval."""
from contextlib import contextmanager
from dataclasses import replace

from pydantic import TypeAdapter
from workflow.enrichment import ContextEnrichment
from workflow.models import ContextEvidence


class FixtureRetriever:
    def __init__(self, payload):
        self.payload = payload

    def retrieve(self, request, stage_name):
        if isinstance(self.payload, dict) and self.payload.get("error"):
            raise RuntimeError(str(self.payload["error"]))
        records = TypeAdapter(list[ContextEvidence]).validate_python(self.payload)
        return [replace(record, stage=stage_name) for record in records]


@contextmanager
def stage_enrichment(stage, inputs):
    """An explicit live_enrichment flag opts into the real backend for integration scenarios."""
    previous = stage.enrichment
    if not inputs.get("live_enrichment", False):
        stage.enrichment = ContextEnrichment(
            stage.character, retriever=FixtureRetriever(inputs.get("enrichment_evidence", [])),
        )
    try:
        yield
    finally:
        stage.enrichment = previous
