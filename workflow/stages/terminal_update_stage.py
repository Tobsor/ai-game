from logger import get_logger
from workflow.models import (
    InitialContext,
    PerceptionResult,
    ResponseResult,
    RetrievedContext,
    StateUpdate,
    StrategyResult,
    TerminalUpdateResult,
)
from workflow.stages.base import LLMStage
from workflow.stages.prompting import format_prompt

logger = get_logger(__name__)


class TerminalUpdateStage(LLMStage):
    def get_prompt(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
        strategy: StrategyResult,
        response: ResponseResult,
    ) -> str:
        return format_prompt(
            "Determine which durable NPC state updates should happen after the response has been produced.",
            [
                ("Player input", perception.raw_prompt),
                ("Response", response.reply),
                ("Current sentiment", initial_context.sentiment),
                ("Strategy", f"actions={', '.join(strategy.immediate_actions)}, new_sentiment={strategy.new_sentiment}"),
                ("Expected result", "Terminal updates covering sentiment, relationship changes, belief updates, goal updates, memory storage, and external actions."),
            ],
        )

    def run(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
        strategy: StrategyResult,
        response: ResponseResult,
    ) -> TerminalUpdateResult:
        logger.verbose("Running terminal update stage")
        return TerminalUpdateResult(
            sentiment=strategy.new_sentiment,
            sentiment_reasoning=strategy.sentiment_reasoning,
            immediate_actions=list(strategy.immediate_actions),
            relationship_update=self.update_relationship(initial_context, perception, response),
            belief_update=self.update_beliefs(initial_context, perception, retrieved_context, response),
            goal_update=self.update_goals(initial_context, perception, retrieved_context, response),
            store_memory=self.store_memory(initial_context, perception, response),
            external_actions=self.trigger_external_actions(initial_context, perception, response, strategy),
        )

    def update_relationship(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        response: ResponseResult,
    ) -> StateUpdate:
        # TODO: Persist relationship changes for future turns.
        return StateUpdate()

    def update_beliefs(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
        response: ResponseResult,
    ) -> StateUpdate:
        # TODO: Persist or enqueue belief changes for the NPC state model.
        return StateUpdate()

    def update_goals(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
        response: ResponseResult,
    ) -> StateUpdate:
        # TODO: Persist or enqueue goal changes for future planning.
        return StateUpdate()

    def store_memory(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        response: ResponseResult,
    ) -> bool:
        # TODO: Store notable memories for future retrieval.
        return False

    def trigger_external_actions(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        response: ResponseResult,
        strategy: StrategyResult,
    ) -> list[str]:
        # TODO: Replace this placeholder passthrough with real game-side effect dispatch.
        return list(strategy.immediate_actions)
