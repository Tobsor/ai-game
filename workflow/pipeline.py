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

    def _log_stage_failure(self, stage_name: str, exc: Exception, payload=None) -> None:
        logger.error("%s failed: %s", stage_name, exc)
        logger.conversation_event(
            stage_name=stage_name,
            event="stage_failed",
            payload=payload,
            result={"error": str(exc)},
            status="error",
        )

    def run(self, turn_input: TurnInput) -> TurnResult:
        stage_name = "InitialContextStage"
        stage_payload = turn_input
        try:
            self._log_stage_start(stage_name, stage_payload)
            initial_context = self.initial_context_stage.run(turn_input)
            self._log_stage_completion(
                stage_name,
                initial_context,
                f"relationship_summary={initial_context.relationship_summary != ''}, active_goals={len(initial_context.active_goals)}",
            )

            stage_name = "PerceptionStage"
            stage_payload = {"prompt": turn_input.prompt}
            self._log_stage_start(stage_name, stage_payload)
            perception = self.perception_stage.run(turn_input, initial_context)
            self._log_stage_completion(
                stage_name,
                perception,
                f"tool_calls={len(perception.tool_calls)}, raw_prompt_length={len(perception.raw_prompt)}",
            )

            stage_name = "GapAnalysisStage"
            stage_payload = perception
            self._log_stage_start(stage_name, stage_payload)
            gap_analysis = self.gap_analysis_stage.run(perception)
            self._log_stage_completion(
                stage_name,
                gap_analysis,
                f"tool_calls={len(gap_analysis.tool_calls)}",
            )

            stage_name = "RetrievalStage.run"
            stage_payload = gap_analysis
            self._log_stage_start(stage_name, stage_payload)
            retrieved_context = self.retrieval_stage.run(perception, gap_analysis)
            self._log_stage_completion(
                stage_name,
                retrieved_context,
                f"combined_context_length={len(retrieved_context.combined_context)}",
            )

            if len(gap_analysis.tool_calls) > 0:
                stage_name = "PerceptionStage.reinterpret"
                stage_payload = {
                    "prompt": perception.raw_prompt,
                    "retrieved_context_length": len(retrieved_context.combined_context),
                }
                self._log_stage_start(stage_name, stage_payload)
                perception = self.perception_stage.run(
                    turn_input,
                    initial_context,
                    retrieved_context=retrieved_context,
                    stage_name=stage_name,
                )
                self._log_stage_completion(
                    stage_name,
                    perception,
                    f"tool_calls={len(perception.tool_calls)}, raw_prompt_length={len(perception.raw_prompt)}",
                )

            stage_name = "StrategyStage"
            stage_payload = {"prompt": perception.raw_prompt}
            self._log_stage_start(stage_name, stage_payload)
            strategy = self.strategy_stage.run(initial_context, perception, retrieved_context)
            self._log_stage_completion(
                stage_name,
                strategy,
                f"goal={strategy.conversation_goal}, actions={','.join(strategy.immediate_actions)}",
            )

            stage_name = "ResponseStage"
            stage_payload = {"prompt": perception.raw_prompt}
            self._log_stage_start(stage_name, stage_payload)
            response = self.response_stage.run(initial_context, perception, retrieved_context, strategy)
            self._log_stage_completion(
                stage_name,
                response,
                f"reply_length={len(response.reply)}",
            )

            stage_name = "TerminalUpdateStage"
            stage_payload = response
            self._log_stage_start(stage_name, stage_payload)
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
                f"immediate_actions={','.join(terminal_update.immediate_actions)}, store_memory={terminal_update.store_memory}",
            )

            logger.conversation_event(
                stage_name="TurnPipeline",
                event="final_response_output",
                payload={
                    "reply": response.reply,
                    "external_actions": terminal_update.external_actions,
                    "store_memory": terminal_update.store_memory,
                },
                result={
                    "response": response,
                    "terminal_update": terminal_update,
                },
            )
        except Exception as exc:
            self._log_stage_failure(stage_name, exc, stage_payload)
            raise

        logger.debug("Turn pipeline completed for %s", self.character.name)

        return TurnResult(
            initial_context=initial_context,
            perception=perception,
            gap_analysis=gap_analysis,
            retrieved_context=retrieved_context,
            strategy=strategy,
            response=response,
            terminal_update=terminal_update,
        )
