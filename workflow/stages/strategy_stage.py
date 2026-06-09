from logger import get_logger
from workflow.models import InitialContext, PerceptionResult, RetrievedContext, StrategyResult
from workflow.stages.base import LLMStage
from workflow.stages.prompting import format_prompt

logger = get_logger(__name__)


class StrategyStage(LLMStage):
    def get_prompt(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
    ) -> str:
        # TODO: Add the NPC goals to the prompt
        return format_prompt(
            "Choose the NPC's high-level response strategy based on the interpreted player input and the context retrieved so far.",
            [
                ("Player input", perception.raw_prompt),
                ("Character sentiment", initial_context.sentiment),
                ("Relationship summary", initial_context.relationship_summary),
                ("Retrieved context", retrieved_context.combined_context),
                (
                    "Strategy guidance",
                    "\n".join([
                        "Decide what interaction strategy this NPC chooses toward the player in this moment.",
                        "Focus on what the NPC wants to achieve, how the NPC wants to treat or position toward the player, what tone or social approach fits, and what would feel natural for this specific character in this situation.",
                        "Consider whether the NPC wants to cooperate, deflect, pressure, help, refuse, test, threaten, bargain, open up, or guide the conversation in another direction.",
                        "The strategy is not limited to dialogue alone. If carrying out the strategy would naturally involve an immediate in-world action, the NPC may use the provided action tools.",
                        "Use those tools only when the action is plausible for the character and meaningfully supports the chosen strategy.",
                        "If no special in-world action is needed, simply continue the interaction.",
                    ]),
                ),
                ("Expected result", "A strategy covering conversation goal, risk, disclosure, social approach, tone, verbosity, conversation move, and any immediate actions."),
            ],
        )

    def run(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
    ) -> StrategyResult:
        logger.verbose("Running strategy stage")
        stage_prompt = self.get_prompt(initial_context, perception, retrieved_context)
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

        intention = self.select_intention(initial_context, perception, retrieved_context)
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
            intention=intention,
            conversation_goal=self.select_conversation_goal(initial_context, perception, retrieved_context),
            risk_level=self.assess_risk(initial_context, perception, retrieved_context),
            disclosure_level=self.decide_disclosure(initial_context, perception, retrieved_context),
            social_strategy=self.select_social_strategy(initial_context, perception, retrieved_context),
            tone=self.select_tone(initial_context, perception, retrieved_context),
            verbosity=self.select_verbosity(initial_context, perception, retrieved_context),
            conversation_move=self.select_conversation_move(initial_context, perception, retrieved_context),
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

    def select_intention(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
    ) -> str:
        # TODO: Infer the NPC's conversational intention from strategy reasoning.
        return "answer_plainly"

    def select_conversation_goal(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
    ) -> str:
        # TODO: Choose the concrete conversation goal from structured reasoning.
        return "answer_plainly"

    def assess_risk(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
    ) -> str:
        # TODO: Compute risk level from threat, sensitivity, and social context.
        return "low"

    def decide_disclosure(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
    ) -> str:
        # TODO: Decide how much information the NPC should reveal.
        return "normal"

    def select_social_strategy(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
    ) -> str:
        # TODO: Select a social strategy such as trust-building or deception.
        return "neutral"

    def select_tone(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
    ) -> str:
        # TODO: Select the response tone from emotion and social strategy.
        return "in_character"

    def select_verbosity(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
    ) -> str:
        # TODO: Choose how concise or expansive the reply should be.
        return "normal"

    def select_conversation_move(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
    ) -> str:
        # TODO: Choose the next conversational move from the strategy result.
        return "answer"
