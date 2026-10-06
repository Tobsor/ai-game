"""Shared, single-pass context enrichment owned by the calling cognitive stage."""
import json
import re
from typing import Any, Callable, TypeVar
from dataclasses import asdict

from logger import get_logger
from workflow.models import ContextEvidence, EnrichmentDecision, RetrievedContext

logger = get_logger(__name__)
StageResult = TypeVar("StageResult")


class GapAnalysis:
    """Prompt instructions and validation, not a separate reasoning/model stage."""

    INSTRUCTIONS = """Context enrichment
Add two top-level strings to your stage JSON: context_request (one concise retrieval query, or "")
and gap_reason (briefly explain the material gap or why no fetch is needed).
Request only retrievable information that could change this stage's result, grounded in the user
input and NPC's perspective. Prefer no fetch when context suffices; low confidence alone is no gap.
Reuse supplied evidence; do not repeat answered or failed requests in retrieval history.
Keep upstream results fixed and preserve uncertainty and subjective beliefs without inventing facts.
Always return a usable stage result; after at most one retrieval and revision, finalize even if unresolved.
"""

    @classmethod
    def prompt(cls, prompt: str, context: RetrievedContext) -> str:
        history = [asdict(item) for item in context.history]
        return prompt + "\n\nRetrieval history:\n" + json.dumps(history, ensure_ascii=False)

    @staticmethod
    def key(request: str) -> str:
        # Conservative deterministic matching. Semantic reuse is also required in the prompt.
        return " ".join(re.findall(r"\w+", request.casefold()))


class ContextRetriever:
    def __init__(self, character):
        self.character = character

    def retrieve(self, request: str, stage_name: str) -> list[ContextEvidence]:
        # All shared information is accessible; only explicit character records are private.
        access_filter = {"$or": [
            {"type": {"$ne": "character"}},
            {"name": {"$eq": self.character.name}},
        ]}
        records = self.character.db.query_records(prompt=request, filter=access_filter)
        evidence = [ContextEvidence(
            id=record["id"], text=record["text"], metadata=record["metadata"],
            distance=record["distance"], stage=stage_name,
        ) for record in records if not (
            record["metadata"].get("type") == "character"
            and record["metadata"].get("name") != self.character.name
        )]
        logger.conversation_event(
            stage_name=stage_name, event="context_retrieval",
            payload={"request": request, "filter": access_filter},
            result=evidence,
        )
        return evidence


class ContextEnrichment:
    def __init__(self, character, retriever=None):
        self.character = character
        self.retriever = retriever or ContextRetriever(character)

    def run(self, *, stage_name: str, initial_context, context: RetrievedContext,
            build_prompt: Callable[[], str],
            parse_result: Callable[[dict[str, Any], Any, str], StageResult],
            tools: list[Callable] | None = None,
            retrieval_topics: bool = False,
            build_refinement_prompt: Callable[[], str] | None = None,
            refinement_system_prompt: str | None = None) -> StageResult:
        """parse_result(payload, response, prompt) validates cognition independently of gaps."""
        context.initial_context = initial_context
        prompt = build_prompt()
        instructions = (
            'Return retrieval_topics as a list of descriptive information needs, or [] when context suffices. '
            'Retrieve only missing NPC context needed for personal significance. Never redo finalized Perception or plan Strategy. '
            'Reuse evidence and request history; finalize after at most one retrieval and revision. '
            'Do not output context_request or gap_reason.'
        ) if retrieval_topics else GapAnalysis.INSTRUCTIONS
        response = self.character.agent.run_prompt(
            prompt=prompt, stage_name=stage_name, tools=tools,
            payload={"stage_prompt": prompt, "phase": "draft"},
            system_prompt=instructions,
        )
        payload = json.loads(response.content)
        if not isinstance(payload, dict):
            raise ValueError(f"{stage_name}: expected a JSON object")
        if retrieval_topics:
            topics = payload.pop("retrieval_topics", None)
            valid_topics = isinstance(topics, list) and all(isinstance(t, str) and t.strip() for t in topics)
            request = "\n".join(dict.fromkeys(t.strip() for t in topics)) if valid_topics else None
            reason = "Missing NPC context for personal significance"
        else:
            request = payload.pop("context_request", None)
            reason = payload.pop("gap_reason", None)
        # A broken draft is a normal stage failure; a broken gap must not discard a valid draft.
        draft = parse_result(payload, response, prompt)
        decision = EnrichmentDecision(stage=stage_name)
        previous = list(context.history)
        context.history.append(decision)
        try:
            if not isinstance(request, str) or not isinstance(reason, str) or not reason.strip():
                return draft
            decision.request, decision.reason = request.strip(), reason.strip()
            if not decision.request:
                return draft
            if any(item.stage == stage_name for item in previous):
                return draft
            key = GapAnalysis.key(decision.request)
            if not key:
                return draft
            if any(item.request and GapAnalysis.key(item.request) == key for item in previous):
                return draft
            try:
                evidence = self.retriever.retrieve(decision.request, stage_name)
            except Exception:
                evidence = []
            if not evidence:
                return draft
            known_ids = {item.id for item in context.evidence}
            new_evidence = []
            for item in evidence:
                if item.id not in known_ids:
                    new_evidence.append(item)
                    known_ids.add(item.id)
            if not new_evidence:
                return draft
            context.evidence.extend(new_evidence)
            # Retain source excerpts, not a perception/response-focused summary.
            snippets = [json.dumps(asdict(item), ensure_ascii=False) for item in new_evidence]
            context.combined_context = "\n".join(filter(None, [context.combined_context, *snippets]))
            # Appraisal reuses its original context and appends only new evidence, reducing
            # refinement input versus resending the full stage prompt and expanded history.
            revision_base = build_refinement_prompt() if build_refinement_prompt else build_prompt()
            revision_prompt = revision_base + "\n\nStage-local revision\n" + "\n".join([
                "Revise only your own draft using the added evidence. Preserve all finalized upstream results.",
                "Keep valid parts of the draft. Evidence may fail to resolve the gap; finalize anyway.",
                ("Return complete final JSON with retrieval_topics: []; no further retrieval is possible."
                 if retrieval_topics else "Return the complete final stage JSON. Set context_request to an empty string; no further retrieval is possible."),
                "Requested context: " + decision.request,
                ("New evidence: " + "\n".join(snippets)) if retrieval_topics else "",
                ("Preserve, refine, add or remove aspects as NPC evidence warrants; revise summaries and emotion, never finalized Perception.") if retrieval_topics else "",
                "If action tools apply, choose the complete final action set; draft actions have not executed.",
                "Original draft: " + json.dumps(payload, ensure_ascii=False),
                "Draft actions: " + json.dumps(getattr(draft, "immediate_actions", [])),
            ])
            try:
                revised = self.character.agent.run_prompt(
                    prompt=revision_prompt, stage_name=stage_name, tools=tools,
                    payload={"stage_prompt": revision_prompt, "phase": "revision"},
                    system_prompt=refinement_system_prompt or instructions,
                )
                revised_payload = json.loads(revised.content)
                if not isinstance(revised_payload, dict):
                    raise ValueError("revision must be a JSON object")
                if retrieval_topics:
                    revised_payload.pop("retrieval_topics", None)
                revised_payload.pop("context_request", None)
                revised_payload.pop("gap_reason", None)
                result = parse_result(revised_payload, revised, revision_prompt)
                return result
            except Exception:
                return draft
        finally:
            logger.conversation_event(
                stage_name=stage_name, event="context_enrichment", result=decision,
            )
