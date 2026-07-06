from logger import get_logger
from workflow.models import InitialContext, PerceptionResult, RetrievedContext, TurnInput
from workflow.stages.base import LLMStage
from workflow.stages.prompting import format_prompt

logger = get_logger(__name__)


class PerceptionStage(LLMStage):
    def get_prompt(
        self,
        turn_input: TurnInput,
        initial_context: InitialContext,
        retrieved_context: RetrievedContext | None = None,
    ) -> str:
        return format_prompt(
            f"You are simulating how the NPC {initial_context.character_name} perceives the player's latest message. Do not generate dialogue. Analyze the player's prompt from this NPC's perspective so downstream stages can act on that perception. Evaluate the message through this character's personality, knowledge, sentiment, and social awareness, and only infer signals this character would realistically notice or understand. Reflect on whether this NPC is perceptive enough to recognize subtle manipulation, threat, ambiguity, or emotional subtext rather than assuming perfect insight.",
            [
                ("Situation", initial_context.situation),
                ("Current sentiment towards player", initial_context.sentiment),
                ("Character definition", initial_context.character_definition),
                ("Example dialogues", initial_context.example_dialogues),
                (
                    "Relevant relationship and goals",
                    "\n".join([
                        initial_context.relationship_summary,
                        "Active goals: " + ", ".join(initial_context.active_goals) if len(initial_context.active_goals) > 0 else "",
                    ]).strip(),
                ),
                (
                    "Recent conversation state",
                    "\n".join(initial_context.recent_turns),
                ),
                (
                    "Decision rubric",
                    "\n".join([
                        "Analyze the player's message and infer player_intent as a concise description of what the player is trying to achieve, or unknown if it cannot be determined.",
                        "Analyze the player's message and infer player_emotion, defaulting to neutral when no strong emotional signal is present.",
                        "Classify request_type with a concise category such as general, question, demand, negotiation, threat, social bid, or similar.",
                        "Summarize the main subject of the player's message in topic.",
                        "Set is_ambiguous to true only when the player's message is too unclear, underspecified, or contradictory for a confident interpretation.",
                        "Set threat_signal to none unless the player expresses hostility, danger, intimidation, coercion, or violent intent.",
                        "Set manipulation_signal to none unless the player appears deceptive, coercive, flattering strategically, guilt-inducing, or otherwise manipulative.",
                        "Set topic_sensitivity to normal unless the topic is sensitive, secret, risky, personal, or delicate for this NPC.",
                        "Return strictly valid JSON with exactly these fields: player_intent, player_emotion, request_type, topic, is_ambiguous, threat_signal, manipulation_signal, topic_sensitivity.",
                        'Do not return markdown, prose, explanations, or code fences. Output only the JSON object.',
                    ]),
                ),
                ("Player input", turn_input.prompt),
                (
                    "Additional retrieved context",
                    retrieved_context.combined_context if retrieved_context is not None else "",
                ),
            ],
        )

    def run(
        self,
        turn_input: TurnInput,
        initial_context: InitialContext,
        retrieved_context: RetrievedContext | None = None,
        stage_name: str = "PerceptionStage",
    ) -> PerceptionResult:
        logger.verbose("Running perception stage for prompt length=%s", len(turn_input.prompt))
        stage_prompt = self.get_prompt(turn_input, initial_context, retrieved_context)
        response = self.character.agent.run_prompt(
            prompt=stage_prompt,
            stage_name=stage_name,
            payload={
                "input_prompt": turn_input.prompt,
                "stage_prompt": stage_prompt,
                "has_retrieved_context": retrieved_context is not None and retrieved_context.combined_context != "",
            },
        )
        parsed_response = self.character.agent.parse_output(response.content, fallback={})

        return PerceptionResult(
            raw_prompt=turn_input.prompt,
            stage_prompt=stage_prompt,
            player_intent=self.detect_player_intent(parsed_response),
            player_emotion=self.detect_player_emotion(parsed_response),
            request_type=self.detect_request_type(parsed_response),
            topic=self.detect_topic(parsed_response),
            is_ambiguous=self.detect_ambiguity(parsed_response),
            threat_signal=self.detect_threat_signal(parsed_response),
            manipulation_signal=self.detect_manipulation_signal(parsed_response),
            topic_sensitivity=self.detect_topic_sensitivity(parsed_response),
            tool_calls=[],
            retrieval_reasoning="",
        )

    def detect_player_intent(self, parsed_response: dict) -> str:
        value = parsed_response.get("player_intent")
        return str(value) if isinstance(value, str) and value.strip() != "" else "unknown"

    def detect_player_emotion(self, parsed_response: dict) -> str:
        value = parsed_response.get("player_emotion")
        return str(value) if isinstance(value, str) and value.strip() != "" else "neutral"

    def detect_request_type(self, parsed_response: dict) -> str:
        value = parsed_response.get("request_type")
        return str(value) if isinstance(value, str) and value.strip() != "" else "general"

    def detect_topic(self, parsed_response: dict) -> str:
        value = parsed_response.get("topic")
        return str(value) if isinstance(value, str) else ""

    def detect_ambiguity(self, parsed_response: dict) -> bool:
        value = parsed_response.get("is_ambiguous")
        return value if isinstance(value, bool) else False

    def detect_threat_signal(self, parsed_response: dict) -> str:
        value = parsed_response.get("threat_signal")
        return str(value) if isinstance(value, str) and value.strip() != "" else "none"

    def detect_manipulation_signal(self, parsed_response: dict) -> str:
        value = parsed_response.get("manipulation_signal")
        return str(value) if isinstance(value, str) and value.strip() != "" else "none"

    def detect_topic_sensitivity(self, parsed_response: dict) -> str:
        value = parsed_response.get("topic_sensitivity")
        return str(value) if isinstance(value, str) and value.strip() != "" else "normal"
