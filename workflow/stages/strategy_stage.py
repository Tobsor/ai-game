import json
from dataclasses import asdict
from workflow.enrichment import GapAnalysis
from workflow.perception_contract import format_perception_core
from logger import get_logger
from workflow.models import AppraisalResult, EmotionResult, InitialContext, PerceptionResult, RetrievedContext, StrategyResult
from workflow.stages.base import EnrichedStage
from workflow.stages.prompting import format_prompt

logger = get_logger(__name__)


class StrategyStage(EnrichedStage):
    def get_default_strategy_payload(self) -> dict[str, str]:
        return {
            "intention": "",
            "conversation_goal": "answer_plainly",
            "risk_level": "low",
            "disclosure_level": "normal",
            "social_strategy": "neutral",
            "tone": "in_character",
            "verbosity": "normal",
            "conversation_move": "answer",
        }

    def get_prompt(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
        appraisal: AppraisalResult,
        emotion: EmotionResult,
    ) -> str:
        return GapAnalysis.prompt(format_prompt(
            "Choose the NPC's high-level response strategy based on final perception, appraisal, emotional reaction, and context. Treat finalized appraisal and emotion as authoritative. Do not redo perception, appraisal, or emotion or reconstruct numerical ratings.",
            [
                ("Player input", perception.raw_prompt),
                ("Character definition", initial_context.character_definition),
                ("Situation", initial_context.situation),
                ("Belief state", "\n".join(initial_context.belief_state)),
                ("Compact final perception", format_perception_core(perception.core)),
                ("Appraisal", json.dumps(asdict(appraisal), ensure_ascii=False)),
                ("Emotional reaction", json.dumps(asdict(emotion), ensure_ascii=False)),
                ("Character sentiment", initial_context.sentiment),
                ("Relationship summary", initial_context.relationship_summary),
                ("Persistent goals and motivations", "\n".join(initial_context.active_goals)),
                ("Recent conversation state", "\n".join(initial_context.recent_turns)),
                ("Retrieved context", retrieved_context.combined_context),
                (
                    "Strategy guidance",
                    "\n".join([
                        "Decide what interaction strategy this NPC chooses toward the player in this moment.",
                        "Generate the immediate conversational goal for this turn; do not treat persistent goals as if they are already the conversational goal.",
                        "Focus on what the NPC wants to achieve now, how the NPC wants to treat or position toward the player, what tone or social approach fits, and what would feel natural for this specific character in this situation.",
                        "Consider whether the NPC wants to cooperate, deflect, pressure, help, refuse, test, threaten, bargain, open up, or guide the conversation in another direction.",
                        "Keep the NPC's intention and strategy portrayal short and concise. Prefer brief labels or compact phrases over elaborate explanations.",
                        "The strategy is not limited to dialogue alone. If carrying out the strategy would naturally involve an immediate in-world action, the NPC may use the provided action tools.",
                        "Use those tools only when the action is plausible for the character and meaningfully supports the chosen strategy.",
                        "The tools are only for immediate_actions. They are not the general strategy output and should not replace the textual evaluation of the NPC's strategy.",
                        "If no special in-world action is needed, simply continue the interaction.",
                    ]),
                ),
                (
                    "Expected JSON output",
                    "\n".join([
                        "{",
                        '  "intention": "short prose description",',
                        '  "conversation_goal": "short prose description",',
                        '  "risk_level": "short prose description",',
                        '  "disclosure_level": "short prose description",',
                        '  "social_strategy": "short prose description",',
                        '  "tone": "short prose description",',
                        '  "verbosity": "short prose description",',
                        '  "conversation_move": "short prose description"',
                        "}",
                    ]),
                ),
                ("Expected result", "Return valid JSON matching the shown cognitive structure plus the enrichment fields below. Keep textual fields concise, and use tools only to express immediate_actions."),
            ],
        ), retrieved_context)

    def run(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
        appraisal: AppraisalResult,
        emotion: EmotionResult,
    ) -> StrategyResult:
        return self.enrichment.run(
            stage_name="StrategyStage", initial_context=initial_context, context=retrieved_context,
            build_prompt=lambda: self.get_prompt(initial_context, perception, retrieved_context, appraisal, emotion),
            parse_result=self.parse_result,
            tools=[self.keep_talking, self.end_conversation, self.open_trade, self.offer_quest,
                   self.alert_guards, self.attack_player, self.flee, self.call_for_help],
        )

    def parse_result(self, parsed_response, response, stage_prompt):
        for key in self.get_default_strategy_payload():
            if not isinstance(parsed_response.get(key), str) or not parsed_response[key].strip():
                raise ValueError(f"Strategy requires a nonblank {key}")
        immediate_actions: list[str] = []
        new_sentiment = None
        sentiment_reasoning = ""

        for tool_call in response.tool_calls or []:
            tool = tool_call.function.name

            if tool == "keep_talking":
                immediate_actions.append("keep_talking")
            elif tool == "end_conversation":
                immediate_actions.append("end_conversation")
            elif tool == "open_trade":
                immediate_actions.append("open_trade")
            elif tool == "offer_quest":
                immediate_actions.append("offer_quest")
            elif tool == "alert_guards":
                immediate_actions.append("alert_guards")
            elif tool in {"attack_player", "flee", "call_for_help"}:
                immediate_actions.append(tool)

        if len(immediate_actions) == 0:
            immediate_actions.append("keep_talking")

        return StrategyResult(
            intention=str(parsed_response.get("intention", "")),
            conversation_goal=str(parsed_response.get("conversation_goal", "answer_plainly")),
            risk_level=str(parsed_response.get("risk_level", "low")),
            disclosure_level=str(parsed_response.get("disclosure_level", "normal")),
            social_strategy=str(parsed_response.get("social_strategy", "neutral")),
            tone=str(parsed_response.get("tone", "in_character")),
            verbosity=str(parsed_response.get("verbosity", "normal")),
            conversation_move=str(parsed_response.get("conversation_move", "answer")),
            immediate_actions=immediate_actions,
            new_sentiment=str(new_sentiment) if new_sentiment is not None else None,
            sentiment_reasoning=sentiment_reasoning,
        )

    def keep_talking(self, reasoning: str) -> dict[str, str]:
        return {"action": "keep_talking", "reasoning": reasoning}

    def end_conversation(self, reasoning: str) -> dict[str, str]:
        return {"action": "end_conversation", "reasoning": reasoning}

    def open_trade(self, reasoning: str) -> dict[str, str]:
        return {"action": "open_trade", "reasoning": reasoning}

    def offer_quest(self, reasoning: str) -> dict[str, str]:
        return {"action": "offer_quest", "reasoning": reasoning}

    def alert_guards(self, reasoning: str) -> dict[str, str]:
        return {"action": "alert_guards", "reasoning": reasoning}

    def attack_player(self, reasoning: str) -> dict[str, str]:
        """Choose an immediate physical attack against the player."""
        # TODO: Integrate combat execution; this currently records intent only.
        return {"action": "attack_player", "reasoning": reasoning}

    def flee(self, reasoning: str) -> dict[str, str]:
        """Choose to escape the current encounter."""
        # TODO: Integrate movement and encounter termination.
        return {"action": "flee", "reasoning": reasoning}

    def call_for_help(self, reasoning: str) -> dict[str, str]:
        """Call nearby people for immediate assistance."""
        # TODO: Integrate nearby NPC reactions; assistance is not guaranteed.
        return {"action": "call_for_help", "reasoning": reasoning}
