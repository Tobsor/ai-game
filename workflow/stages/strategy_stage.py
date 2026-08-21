from logger import get_logger
from workflow.models import AppraisalResult, EmotionResult, InitialContext, PerceptionResult, RetrievedContext, StrategyResult
from workflow.stages.base import LLMStage
from workflow.stages.prompting import format_prompt

logger = get_logger(__name__)


class StrategyStage(LLMStage):
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
        return format_prompt(
            "Choose the NPC's high-level response strategy based on final perception, appraisal, emotional reaction, and context. Do not redo perception, appraisal, or emotion.",
            [
                ("Player input", perception.raw_prompt),
                (
                    "Compact final perception",
                    "\n".join([
                        f"summary={perception.summary}",
                        f"perceived_intent={', '.join(perception.perceived_intent)}",
                        f"perceived_attitude={', '.join(perception.perceived_attitude)}",
                        f"relevant_topics={', '.join(perception.relevant_topics)}",
                        f"target={', '.join(perception.target)}",
                        f"confidence={perception.confidence}",
                    ]),
                ),
                (
                    "Appraisal",
                    "\n".join([
                        f"relevance={appraisal.relevance}",
                        f"valence={appraisal.valence}",
                        f"goal_impact={appraisal.goal_impact}",
                        f"social_self_impact={appraisal.social_self_impact}",
                        f"threat={appraisal.threat}",
                        f"control={appraisal.control}",
                        f"attribution_source={appraisal.attribution_source}",
                        f"attribution_responsibility={appraisal.attribution_responsibility}",
                        f"summary={appraisal.summary}",
                    ]),
                ),
                (
                    "Emotional reaction",
                    "\n".join([
                        f"primary={emotion.primary}",
                        f"secondary={', '.join(emotion.secondary)}",
                        f"intensity={emotion.intensity}",
                    ]),
                ),
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
                ("Expected result", "Return valid JSON matching the shown structure. Keep the textual fields concise, and use tools only to express immediate_actions."),
            ],
        )

    def run(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
        appraisal: AppraisalResult,
        emotion: EmotionResult,
    ) -> StrategyResult:
        logger.verbose("Running strategy stage")
        stage_prompt = self.get_prompt(initial_context, perception, retrieved_context, appraisal, emotion)
        response = self.character.agent.run_prompt(
            prompt=stage_prompt,
            tools=[
                self.keep_talking,
                self.end_conversation,
                self.open_trade,
                self.offer_quest,
                self.alert_guards,
            ],
            stage_name="StrategyStage",
            payload={
                "input_prompt": perception.raw_prompt,
                "stage_prompt": stage_prompt,
            },
        )

        parsed_response = self.character.agent.parse_output(
            response.content,
            fallback=self.get_default_strategy_payload(),
        )
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
