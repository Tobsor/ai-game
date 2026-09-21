from logger import get_logger
from workflow.models import InitialContext, NPCPerception, PerceptionResult, PerceptionTopic, RetrievedContext, TurnInput
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
                        "Analyze the player's message and infer npc_perception.player_intent as a concise description of what the player is trying to achieve, or unknown if it cannot be determined.",
                        "Summarize what the NPC subjectively believes is happening in summary.",
                        "Represent npc_perception.perceived_intent and npc_perception.perceived_attitude as compact arrays of labels.",
                        "Analyze the player's message and infer npc_perception.player_emotion, defaulting to neutral when no strong emotional signal is present.",
                        "Classify request_type with a concise category such as general, question, demand, negotiation, threat, social bid, or similar.",
                        "Group all theme and retrieval cues in topic. Use topic.primary for the main subject, topic.related for compact related topic labels, and topic.retrieval_queries for concrete retrieval queries that could help later stages.",
                        "Set npc_perception.threat_signal to none unless the player expresses hostility, danger, intimidation, coercion, or violent intent.",
                        "Set npc_perception.manipulation_signal to none unless the player appears deceptive, coercive, flattering strategically, guilt-inducing, or otherwise manipulative.",
                        "Set npc_perception.topic_sensitivity to normal unless the topic is sensitive, secret, risky, personal, or delicate for this NPC.",
                        "Return strictly valid JSON with exactly these top-level fields: summary, npc_perception, topic, request_type.",
                        'The npc_perception object must contain: perceived_intent, perceived_attitude, threat_signal, manipulation_signal, topic_sensitivity, player_intent, player_emotion.',
                        'The topic object must contain: primary, related, retrieval_queries.',
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

        npc_perception_payload = self.detect_npc_perception_payload(parsed_response)
        return PerceptionResult(
            raw_prompt=turn_input.prompt,
            stage_prompt=stage_prompt,
            summary=self.detect_summary(parsed_response),
            npc_perception=NPCPerception(
                perceived_intent=self.detect_string_list(npc_perception_payload, "perceived_intent"),
                perceived_attitude=self.detect_string_list(npc_perception_payload, "perceived_attitude"),
                player_intent=self.detect_player_intent(npc_perception_payload),
                player_emotion=self.detect_player_emotion(npc_perception_payload),
                threat_signal=self.detect_threat_signal(npc_perception_payload),
                manipulation_signal=self.detect_manipulation_signal(npc_perception_payload),
                topic_sensitivity=self.detect_topic_sensitivity(npc_perception_payload),
            ),
            request_type=self.detect_request_type(parsed_response),
            topic=self.detect_topic(parsed_response),
            tool_calls=[],
            retrieval_reasoning="",
        )

    def detect_summary(self, parsed_response: dict) -> str:
        value = parsed_response.get("summary")
        if isinstance(value, str) and value.strip() != "":
            return value.strip()
        npc_perception_payload = self.detect_npc_perception_payload(parsed_response)
        intent = self.detect_player_intent(npc_perception_payload)
        topic = self.detect_topic(parsed_response).primary
        if topic != "":
            return f"{intent} about {topic}"
        return intent

    def detect_npc_perception_payload(self, parsed_response: dict) -> dict:
        value = parsed_response.get("npc_perception")
        return value if isinstance(value, dict) else {}

    def detect_string_list(self, parsed_response: dict, key: str) -> list[str]:
        value = parsed_response.get(key)
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip() != ""]

    def detect_player_intent(self, parsed_response: dict) -> str:
        value = parsed_response.get("player_intent")
        return str(value) if isinstance(value, str) and value.strip() != "" else "unknown"

    def detect_player_emotion(self, parsed_response: dict) -> str:
        value = parsed_response.get("player_emotion")
        return str(value) if isinstance(value, str) and value.strip() != "" else "neutral"

    def detect_request_type(self, parsed_response: dict) -> str:
        value = parsed_response.get("request_type")
        return str(value) if isinstance(value, str) and value.strip() != "" else "general"

    def detect_topic(self, parsed_response: dict) -> PerceptionTopic:
        value = parsed_response.get("topic")
        if isinstance(value, dict):
            return PerceptionTopic(
                primary=self.detect_topic_value(value.get("primary")),
                related=self.detect_string_list(value, "related"),
                retrieval_queries=self.detect_string_list(value, "retrieval_queries"),
            )
        return PerceptionTopic()

    def detect_topic_value(self, value: object) -> str:
        return str(value).strip() if isinstance(value, str) and value.strip() != "" else ""

    def detect_threat_signal(self, parsed_response: dict) -> str:
        value = parsed_response.get("threat_signal")
        return str(value) if isinstance(value, str) and value.strip() != "" else "none"

    def detect_manipulation_signal(self, parsed_response: dict) -> str:
        value = parsed_response.get("manipulation_signal")
        return str(value) if isinstance(value, str) and value.strip() != "" else "none"

    def detect_topic_sensitivity(self, parsed_response: dict) -> str:
        value = parsed_response.get("topic_sensitivity")
        return str(value) if isinstance(value, str) and value.strip() != "" else "normal"
