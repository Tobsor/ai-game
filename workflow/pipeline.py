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

    def _log_stage_start(self, stage_name: str, payload) -> None:
        logger.verbose("%s started", stage_name)
        logger.conversation_event(
            stage_name=stage_name,
            event="stage_started",
            payload=payload,
        )

    def _log_stage_completion(self, stage_name: str, result, summary: str) -> None:
        logger.verbose("%s completed successfully: %s", stage_name, summary)
        logger.conversation_event(
            stage_name=stage_name,
            event="stage_completed",
            result=result,
        )

    def _log_stage_failure(self, stage_name: str, exc: Exception) -> None:
        logger.error("%s failed: %s", stage_name, exc)
        logger.conversation_event(
            stage_name=stage_name,
            event="stage_failed",
            result={"error": str(exc)},
            status="error",
        )

    def run(self, turn_input: TurnInput) -> TurnResult:
        stage_name = "InitialContextStage"
        try:
            self._log_stage_start(stage_name, turn_input)
            initial_context = self.initial_context_stage.run(turn_input)
            self._log_stage_completion(
                stage_name,
                initial_context,
                f"relationship_summary={initial_context.relationship_summary != ''}, active_goals={len(initial_context.active_goals)}",
            )

            stage_name = "PerceptionStage"
            self._log_stage_start(stage_name, {"prompt": turn_input.prompt})
            perception = self.perception_stage.run(turn_input, initial_context)
            self._log_stage_completion(
                stage_name,
                perception,
                f"tool_calls={len(perception.tool_calls)}, jailbreak_detected={perception.jailbreak_detected}",
            )

            stage_name = "GapAnalysisStage"
            self._log_stage_start(stage_name, perception)
            gap_analysis = self.gap_analysis_stage.run(perception)
            self._log_stage_completion(
                stage_name,
                gap_analysis,
                f"needs_retrieval={gap_analysis.needs_retrieval}, actions={len(gap_analysis.retrieval_actions)}",
            )

            stage_name = "RetrievalStage.create_plan"
            self._log_stage_start(stage_name, gap_analysis)
            retrieval_plan = self.retrieval_stage.create_plan(perception, gap_analysis)
            self._log_stage_completion(
                stage_name,
                retrieval_plan,
                f"requires_retrieval={retrieval_plan.requires_retrieval}, filters={len(retrieval_plan.filters)}",
            )

            stage_name = "RetrievalStage.run"
            self._log_stage_start(stage_name, retrieval_plan)
            retrieved_context = self.retrieval_stage.run(retrieval_plan, gap_analysis)
            self._log_stage_completion(
                stage_name,
                retrieved_context,
                f"combined_context_length={len(retrieved_context.combined_context)}, belief_updates={len(retrieved_context.belief_updates)}",
            )

            stage_name = "StrategyStage"
            self._log_stage_start(stage_name, {"prompt": perception.normalized_prompt})
            strategy = self.strategy_stage.run(initial_context, perception, retrieved_context)
            self._log_stage_completion(
                stage_name,
                strategy,
                f"goal={strategy.conversation_goal}, action={strategy.immediate_action}",
            )

            stage_name = "ResponseStage"
            self._log_stage_start(stage_name, {"prompt": perception.normalized_prompt})
            response = self.response_stage.run(initial_context, perception, retrieved_context, strategy)
            self._log_stage_completion(
                stage_name,
                response,
                f"reply_length={len(response.reply)}",
            )

            stage_name = "TerminalUpdateStage"
            self._log_stage_start(stage_name, response)
            terminal_update = self.terminal_update_stage.run(
                initial_context,
                perception,
                retrieved_context,
                strategy,
                response,
            )
            self._log_stage_completion(
                stage_name,
                terminal_update,
                f"immediate_action={terminal_update.immediate_action}, store_memory={terminal_update.store_memory}",
            )
        except Exception as exc:
            self._log_stage_failure(stage_name, exc)
            raise

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
