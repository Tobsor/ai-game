import json
from dataclasses import asdict
from workflow.enrichment import GapAnalysis
from workflow.perception_contract import format_perception_core

from logger import get_logger
from workflow.models import AppraisalResult, EmotionResult, InitialContext, PerceptionResult, RetrievedContext
from workflow.stages.base import EnrichedStage
from workflow.stages.prompting import format_prompt

logger = get_logger(__name__)


class AppraisalStage(EnrichedStage):
    GROUNDING_GUIDANCE = (
        "Preserve the player's stance as represented by finalized Perception. Distinguish that stance from the NPC's beliefs and subjective reaction; do not turn disagreement into agreement.",
        "Explain how each activated aspect affects the NPC's concerns. Defensive self-affirmation is not external validation; when present, explain its basis without attributing agreement to the player.",
        "Keep appraisal summary, aspects, and emotion mutually consistent. Mixed reactions require supported reasons; derive emotions from those reasons.",
    )
    REFINEMENT_SYSTEM_PROMPT = """Appraisal refinement
Refine the primary Appraisal using the requested and newly retrieved NPC context.
Treat finalized Perception as authoritative; never reinterpret the player's intent.
Preserve supported aspects, refine or remove contradicted provisional aspects, and add
new personally meaningful aspects where warranted. Update the personal-significance
summary and immediate emotional state together, from the NPC's subjective perspective.
Use concise prose and emotion components, without numerical ratings or emotion hierarchy.
Emotion summary is authoritative. Do not formulate conversational goals, plans, actions or dialogue.
Return complete final JSON: unchanged raw_prompt, appraisal {summary, aspects:
[{aspect, evaluation}]}, emotion {components, summary}, and retrieval_topics: [].
No further retrieval is possible; preserve uncertainty where evidence is insufficient.
Do not output stage_prompt, context_request or gap_reason.
""" + "\n".join(GROUNDING_GUIDANCE)

    def get_prompt(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
    ) -> str:
        return GapAnalysis.prompt(format_prompt(
            "Evaluate what the final perceived player message means for this NPC, then derive the immediate emotional reaction. Do not generate dialogue or a response strategy.",
            [
                ("Player input", perception.raw_prompt),
                ("Final perception", format_perception_core(perception.core)),
                ("Situation", initial_context.situation),
                ("Current sentiment towards player", initial_context.sentiment),
                ("Relationship summary", initial_context.relationship_summary),
                ("Persistent goals and motivations", "\n".join(initial_context.active_goals)),
                ("Belief state", "\n".join(initial_context.belief_state)),
                ("Recent conversation state", "\n".join(initial_context.recent_turns)),
                ("Character definition", initial_context.character_definition),
                ("Retrieved context", retrieved_context.combined_context),
                (
                    "Appraisal boundaries",
                    "\n".join([
                        "Appraisal answers what the perceived situation means for the NPC personally.",
                        "Never redo finalized Perception. Conflicting evidence may produce emotional tension or confusion without correcting it.",
                        *self.GROUNDING_GUIDANCE,
                        "Emotion answers how the NPC feels as a result of that appraisal.",
                        "Do not create the immediate conversational goal; that belongs to the strategy stage.",
                        "Evaluate from the NPC's subjective perspective using character, beliefs, history, relationships, goals and available memories.",
                        "Use concise meaningful prose. Appraisal summary is the compact downstream conclusion.",
                        "Identify only personally activated aspects; preserve mixed implications without numerical ratings or a fixed taxonomy.",
                        "Emotion summary is authoritative: describe immediate feelings, targets, tensions and relative strength naturally in prose.",
                        "Emotion components are supporting keywords, without a primary/secondary hierarchy or universal intensity.",
                        "Identify supported personal significance, including benefits, costs, tensions, and uncertainty; never conversational goals, plans, actions, or dialogue.",
                        "Request descriptive retrieval_topics only for missing personal significance context, such as prior promises or current motivations.",
                        "Never retrieve to reinterpret player intent or decide how to exploit the situation; Perception is authoritative and Strategy decides behavior.",
                    ]),
                ),
                (
                    "Expected JSON output",
                    "\n".join([
                        "{",
                        '  "raw_prompt": "exact original player input",',
                        '  "appraisal": {"summary": "personal significance", "aspects": [{"aspect": "self_image", "evaluation": "contextual personal meaning"}]},',
                        '  "emotion": {"components": ["emotion keyword"], "summary": "immediate emotional consequence in contextual prose"},',
                        '  "retrieval_topics": []',
                        "}",
                    ]),
                ),
                ("Expected result", "Return only valid JSON matching the shown cognitive structure including retrieval_topics. Do not return markdown, prose outside JSON, or code fences."),
            ],
        ), retrieved_context)

    def run(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
    ) -> tuple[AppraisalResult, EmotionResult]:
        prior_context = retrieved_context.combined_context
        return self.enrichment.run(
            stage_name="AppraisalStage", initial_context=initial_context, context=retrieved_context,
            build_prompt=lambda: self.get_prompt(initial_context, perception, retrieved_context),
            parse_result=lambda payload, response, prompt: self.parse_result(payload, response, prompt, perception.raw_prompt),
            retrieval_topics=True,
            build_refinement_prompt=lambda: self.get_refinement_prompt(initial_context, perception, prior_context),
            refinement_system_prompt=self.REFINEMENT_SYSTEM_PROMPT,
        )

    def get_refinement_prompt(self, initial_context, perception, prior_context):
        # Omit the primary call's schema example and long instructions. The draft
        # supplies the schema and unaffected conclusions; new evidence is appended
        # by the shared helper. Keep complete NPC state and authoritative perception.
        return format_prompt(
            "Refine the NPC's personal appraisal and immediate emotion, never Perception or Strategy. "
            "Return raw_prompt unchanged, appraisal {summary, aspects: [{aspect, evaluation}]}, "
            "emotion {components, summary}, retrieval_topics: []. Use meaningful prose; no scores, "
            "emotion hierarchy, conversational goals, plans or dialogue. Emotion summary is authoritative. "
            "Retain supported draft conclusions; resolve missing personal context and internal appraisal contradictions without reopening finalized Perception.",
            [("Previously available context", prior_context),
             ("NPC state", json.dumps(asdict(initial_context), ensure_ascii=False)),
             ("Finalized perception", json.dumps(asdict(perception), ensure_ascii=False))],
        )

    def parse_result(self, parsed_response, response=None, stage_prompt=None, raw_prompt=None):
        from workflow.appraisal_contract import validate_appraisal
        return validate_appraisal(parsed_response, raw_prompt)
