from typing import Any

from logger import get_logger
from workflow.models import AppraisalResult, EmotionResult, InitialContext, PerceptionResult, RetrievedContext
from workflow.stages.base import LLMStage
from workflow.stages.prompting import format_prompt

logger = get_logger(__name__)


class AppraisalStage(LLMStage):
    def get_prompt(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
    ) -> str:
        return format_prompt(
            "Evaluate what the final perceived player message means for this NPC, then derive the immediate emotional reaction. Do not generate dialogue or a response strategy.",
            [
                ("Player input", perception.raw_prompt),
                (
                    "Final perception",
                    "\n".join([
                        f"summary={perception.summary}",
                        f"perceived_intent={', '.join(perception.perceived_intent)}",
                        f"perceived_attitude={', '.join(perception.perceived_attitude)}",
                        f"relevant_topics={', '.join(perception.relevant_topics)}",
                        f"target={', '.join(perception.target)}",
                        f"confidence={perception.confidence}",
                        f"legacy_player_intent={perception.player_intent}",
                        f"legacy_player_emotion={perception.player_emotion}",
                        f"legacy_topic={perception.topic}",
                    ]),
                ),
                ("Situation", initial_context.situation),
                ("Current sentiment towards player", initial_context.sentiment),
                ("Relationship summary", initial_context.relationship_summary),
                ("Persistent goals and motivations", "\n".join(initial_context.active_goals)),
                ("Belief state", "\n".join(initial_context.belief_state)),
                ("Character definition", initial_context.character_definition),
                ("Retrieved context", retrieved_context.combined_context),
                (
                    "Appraisal boundaries",
                    "\n".join([
                        "Appraisal answers what the perceived situation means for the NPC personally.",
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
                ("Expected result", "Return only valid JSON matching the shown structure. Do not return markdown, prose, explanations, or code fences."),
            ],
        )

    def run(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
    ) -> tuple[AppraisalResult, EmotionResult]:
        logger.verbose("Running appraisal stage")
        stage_prompt = self.get_prompt(initial_context, perception, retrieved_context)
        response = self.character.agent.run_prompt(
            prompt=stage_prompt,
            stage_name="AppraisalStage",
            payload={
                "input_prompt": perception.raw_prompt,
                "stage_prompt": stage_prompt,
            },
        )
        parsed_response = self.character.agent.parse_output(response.content, fallback={})
        appraisal_payload = self.read_dict(parsed_response.get("appraisal"))
        emotion_payload = self.read_dict(parsed_response.get("emotion"))
        attribution_payload = self.read_dict(appraisal_payload.get("attribution"))

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
