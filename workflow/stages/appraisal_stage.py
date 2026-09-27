from workflow.enrichment import GapAnalysis
from workflow.perception_contract import format_perception_core
from typing import Any
import math

from logger import get_logger
from workflow.models import AppraisalResult, EmotionResult, InitialContext, PerceptionResult, RetrievedContext
from workflow.stages.base import EnrichedStage
from workflow.stages.prompting import format_prompt

logger = get_logger(__name__)


class AppraisalStage(EnrichedStage):
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
                        "Emotion answers how the NPC feels as a result of that appraisal.",
                        "Do not create the immediate conversational goal; that belongs to the strategy stage.",
                        "Use relevance, threat, control, attribution_responsibility, and emotion intensity in the range 0..1.",
                        "Use valence, goal_impact, and social_self_impact in the range -1..1.",
                    ]),
                ),
                (
                    "Expected JSON output",
                    "\n".join([
                        "{",
                        '  "appraisal": {',
                        '    "relevance": 0.0,',
                        '    "valence": 0.0,',
                        '    "goal_impact": 0.0,',
                        '    "social_self_impact": 0.0,',
                        '    "threat": 0.0,',
                        '    "control": 0.5,',
                        '    "attribution": {"source": "unknown", "responsibility": 0.0},',
                        '    "summary": "short prose description"',
                        "  },",
                        '  "emotion": {',
                        '    "primary": "neutral",',
                        '    "secondary": [],',
                        '    "intensity": 0.0',
                        "  }",
                        "}",
                    ]),
                ),
                ("Expected result", "Return only valid JSON matching the shown cognitive structure plus the enrichment fields below. Do not return markdown, prose outside JSON, or code fences."),
            ],
        ), retrieved_context)

    def run(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
    ) -> tuple[AppraisalResult, EmotionResult]:
        return self.enrichment.run(
            stage_name="AppraisalStage", initial_context=initial_context, context=retrieved_context,
            build_prompt=lambda: self.get_prompt(initial_context, perception, retrieved_context),
            parse_result=self.parse_result,
        )

    def parse_result(self, parsed_response, response, stage_prompt):
        if not isinstance(parsed_response.get("appraisal"), dict) or not isinstance(parsed_response.get("emotion"), dict):
            raise ValueError("Appraisal requires appraisal and emotion objects")
        appraisal_payload = self.read_dict(parsed_response.get("appraisal"))
        emotion_payload = self.read_dict(parsed_response.get("emotion"))
        attribution_payload = self.read_dict(appraisal_payload.get("attribution"))
        for payload, key in ((appraisal_payload, "summary"), (emotion_payload, "primary"), (attribution_payload, "source")):
            if not isinstance(payload.get(key), str) or not payload[key].strip():
                raise ValueError(f"Appraisal requires a nonblank {key}")
        secondary = emotion_payload.get("secondary")
        if not isinstance(secondary, list) or not all(isinstance(item, str) and item.strip() for item in secondary):
            raise ValueError("Emotion secondary must be a list of nonblank strings")
        for payload, key, minimum in [
            *((appraisal_payload, key, 0) for key in ("relevance", "threat", "control")),
            *((appraisal_payload, key, -1) for key in ("valence", "goal_impact", "social_self_impact")),
            (emotion_payload, "intensity", 0), (attribution_payload, "responsibility", 0),
        ]:
            value = payload.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= 1:
                raise ValueError(f"Appraisal {key} must be a finite number in {minimum}..1")

        appraisal = AppraisalResult(
            relevance=self.read_float(appraisal_payload, "relevance", 0.0, 0.0, 1.0),
            valence=self.read_float(appraisal_payload, "valence", 0.0, -1.0, 1.0),
            goal_impact=self.read_float(appraisal_payload, "goal_impact", 0.0, -1.0, 1.0),
            social_self_impact=self.read_float(appraisal_payload, "social_self_impact", 0.0, -1.0, 1.0),
            threat=self.read_float(appraisal_payload, "threat", 0.0, 0.0, 1.0),
            control=self.read_float(appraisal_payload, "control", 0.5, 0.0, 1.0),
            attribution_source=self.read_string(attribution_payload, "source", "unknown"),
            attribution_responsibility=self.read_float(attribution_payload, "responsibility", 0.0, 0.0, 1.0),
            summary=self.read_string(appraisal_payload, "summary", ""),
            stage_prompt=stage_prompt,
        )
        emotion = EmotionResult(
            primary=self.read_string(emotion_payload, "primary", "neutral"),
            secondary=self.read_string_list(emotion_payload, "secondary"),
            intensity=self.read_float(emotion_payload, "intensity", 0.0, 0.0, 1.0),
            stage_prompt=stage_prompt,
        )
        return appraisal, emotion

    def read_dict(self, value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    def read_string(self, payload: dict[str, Any], key: str, default: str) -> str:
        value = payload.get(key)
        return str(value).strip() if isinstance(value, str) and value.strip() != "" else default

    def read_string_list(self, payload: dict[str, Any], key: str) -> list[str]:
        value = payload.get(key)
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip() != ""]

    def read_float(self, payload: dict[str, Any], key: str, default: float, minimum: float, maximum: float) -> float:
        value = payload.get(key, default)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, min(maximum, parsed))
