from logger import get_logger
from workflow.models import TurnInput, TurnResult
from workflow.stages import (
    GapAnalysisStage,
    InitialContextStage,
    PerceptionStage,
    ResponseStage,
    RetrievalStage,
    StrategyStage,
    TerminalUpdateStage,
)

logger = get_logger(__name__)


class TurnPipeline:
    def __init__(self, character):
        self.character = character
        self.initial_context_stage = InitialContextStage(character)
        self.perception_stage = PerceptionStage(character)
        self.gap_analysis_stage = GapAnalysisStage(character)
        self.retrieval_stage = RetrievalStage(character)
        self.strategy_stage = StrategyStage(character)
        self.response_stage = ResponseStage(character)
        self.terminal_update_stage = TerminalUpdateStage(character)

    def run(self, turn_input: TurnInput) -> TurnResult:
        initial_context = self.initial_context_stage.run(turn_input)
        perception = self.perception_stage.run(turn_input, initial_context)
        gap_analysis = self.gap_analysis_stage.run(perception)
        retrieval_plan = self.retrieval_stage.create_plan(perception, gap_analysis)
        retrieved_context = self.retrieval_stage.run(retrieval_plan, gap_analysis)
        strategy = self.strategy_stage.run(initial_context, perception, retrieved_context)
        response = self.response_stage.run(initial_context, perception, retrieved_context, strategy)
        terminal_update = self.terminal_update_stage.run(
            initial_context,
            perception,
            retrieved_context,
            strategy,
            response,
        )

        logger.debug("Turn pipeline completed for %s", self.character.name)

        return TurnResult(
            initial_context=initial_context,
            perception=perception,
            gap_analysis=gap_analysis,
            retrieval_plan=retrieval_plan,
            retrieved_context=retrieved_context,
            strategy=strategy,
            response=response,
            terminal_update=terminal_update,
        )
