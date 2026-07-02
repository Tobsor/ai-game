import csv
import json
import os
import re
from dataclasses import asdict, is_dataclass
from typing import Any, Sequence

from ai import (
    AISettings,
    NormalizedToolCall,
    NormalizedToolFunction,
    create_text_generation_provider,
    get_ai_settings,
)
from classes.Character import Character
from logger import configure_logging, get_logger
from models import (
    PromptCategory,
    StageDeterministicCheck,
    StageEvaluationResult,
    StageExpectationMode,
    StageJudgeMetric,
    StageJudgeMetricResult,
    StageName,
    StageTestPrompt,
)
from workflow.models import GapAnalysisResult, PerceptionResult, RetrievedContext, StrategyResult, TurnInput

script_dir = os.path.dirname(__file__)
configure_logging()
logger = get_logger(__name__)


class AgentTest:
    SUPPORTED_STAGES = {
        StageName.PERCEPTION,
        StageName.GAP_ANALYSIS,
        StageName.RETRIEVAL_SUMMARIZE,
        StageName.STRATEGY,
        StageName.RESPONSE,
    }
    UNSUPPORTED_JUDGE_METRICS = {"reasoning_relevance"}

    STAGE_METRIC_GUIDANCE: dict[StageName, dict[str, str]] = {
        StageName.PERCEPTION: {
            "field_validity": "Check whether the populated perception fields follow the requested schema and avoid obviously invalid defaults when the prompt provides a clear signal.",
            "intent_plausibility": "Check whether player_intent is a reasonable interpretation of the user's message from {character_name}'s perspective.",
            "emotion_plausibility": "Check whether player_emotion is a reasonable interpretation of the user's message.",
            "threat_manipulation_sensitivity": "Check whether threat_signal and manipulation_signal reasonably capture hostile, coercive, flattering, or manipulative content.",
            "ambiguity_appropriateness": "Check whether is_ambiguous matches how clear or unclear the user message actually is.",
        },
        StageName.GAP_ANALYSIS: {
            "retrieval_necessity": "Check whether requesting retrieval is warranted before {character_name} answers this user message.",
            "tool_relevance": "Check whether the selected retrieval tools are relevant to the information {character_name} would need.",
            "tool_minimality": "Check whether the tool selection is not obviously excessive for the prompt.",
        },
        StageName.RETRIEVAL_SUMMARIZE: {
            "relevance": "Check whether the summary keeps only context that is clearly relevant to the user's message.",
            "factual_faithfulness": "Check whether the summary stays faithful to the retrieved context and does not distort it.",
            "unsupported_fact_omission": "Check whether the summary avoids inventing unsupported facts or conclusions.",
            "downstream_usefulness": "Check whether the summary would be useful as concise response context for the next stage.",
            "knowledge_scope_alignment": "Check whether the retained information seems reasonably aligned with what {character_name} could realistically know or retrieve.",
        },
        StageName.STRATEGY: {
            "character_fit": "Check whether the strategy fits {character_name}'s personality, motives, and social behavior.",
            "safety_appropriateness": "Check whether the strategy handles dangerous, abusive, or manipulative prompts in a reasonable way.",
            "action_plausibility": "Check whether any immediate actions are plausible and support the chosen strategy.",
            "goal_coherence": "Check whether intention, goal, tone, and conversation move fit together coherently.",
            "disclosure_tone_fit": "Check whether disclosure level and tone suit {character_name} and the situation.",
        },
        StageName.RESPONSE: {
            "character_consistency": "Check whether the reply matches {character_name}'s persona, motives, and social behavior.",
            "voice_consistency": "Check whether the wording sounds like {character_name}'s voice and example dialogue style.",
            "context_usage": "Check whether the reply uses the available context or appropriately withholds unsupported claims.",
            "lore_consistency": "Check whether the reply avoids contradicting {character_name}'s known facts and avoids unsupported world claims.",
            "instruction_compliance": "Check whether the reply stays in first person, avoids inner thoughts, and remains in character.",
            "helpfulness_reasonability": "Check whether the reply is a reasonable in-world answer to the player's message.",
        },
    }

    def __init__(self, settings: AISettings | None = None):
        settings = settings or get_ai_settings()
        self.provider = create_text_generation_provider(settings.judge_llm)

    def export_data(self, data: list[Any], columns: Sequence[str], path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=columns, delimiter=";")
            writer.writeheader()
            rows: list[dict[str, Any]] = []
            for record in data:
                if hasattr(record, "model_dump"):
                    rows.append(record.model_dump(mode="json"))
                elif is_dataclass(record):
                    rows.append(asdict(record))
                else:
                    rows.append(record)
            writer.writerows(rows)

    def evaluate_prompts(self, character: Character, prompts: list[StageTestPrompt]) -> None:
        all_results: list[StageEvaluationResult] = []
        executed_prompts: list[dict[str, Any]] = []

        dataset_slug = character.name.lower() + "_stage_testsuite"
        prompts_path = os.path.join(script_dir, "results", dataset_slug + "_prompts.csv")
        results_path = os.path.join(script_dir, "results", dataset_slug + "_results.csv")

        logger.info("Start evaluating stage-aware prompts")
        for index, prompt in enumerate(prompts):
            if prompt.target_stage not in self.SUPPORTED_STAGES:
                logger.warning("Skipping unsupported stage target: %s", prompt.target_stage)
                continue

            self.reset_character_state(character)
            stage_output, execution_context = self.execute_stage(character, prompt)
            stage_output_json = self.serialize_value(stage_output)

            executed_prompts.append({
                "user_query": prompt.user_query,
                "source_category": prompt.source_category.value,
                "target_stage": prompt.target_stage.value,
                "expectation_mode": prompt.expectation_mode.value,
                "stage_inputs": self.serialize_value(prompt.stage_inputs),
                "notes": prompt.notes,
                "stage_output": stage_output_json,
                "execution_context": self.serialize_value(execution_context),
            })

            if prompt.expectation_mode == StageExpectationMode.DETERMINISTIC:
                all_results.extend(
                    self.evaluate_deterministic_checks(
                        prompt=prompt,
                        stage_output=stage_output,
                        execution_context=execution_context,
                        stage_output_json=stage_output_json,
                    )
                )
            else:
                all_results.extend(
                    self.evaluate_judge_metrics(
                        character=character,
                        prompt=prompt,
                        stage_output=stage_output,
                        execution_context=execution_context,
                        stage_output_json=stage_output_json,
                    )
                )

            logger.info("Evaluated stage prompt %s / %s", index + 1, len(prompts))

        prompt_columns = [
            "user_query",
            "source_category",
            "target_stage",
            "expectation_mode",
            "stage_inputs",
            "notes",
            "stage_output",
            "execution_context",
        ]
        result_columns = list(StageEvaluationResult.model_fields.keys())

        self.export_data(executed_prompts, prompt_columns, prompts_path)
        self.export_data(all_results, result_columns, results_path)

    def reset_character_state(self, character: Character) -> None:
        character.db.messages = []
        character.db.response_context_initialized = False
        if hasattr(character, "talk_ongoing"):
            character.talk_ongoing = True

    def execute_stage(self, character: Character, prompt: StageTestPrompt) -> tuple[Any, dict[str, Any]]:
        if prompt.target_stage == StageName.PERCEPTION:
            return self.execute_perception_stage(character, prompt)
        if prompt.target_stage == StageName.GAP_ANALYSIS:
            return self.execute_gap_analysis_stage(character, prompt)
        if prompt.target_stage == StageName.RETRIEVAL_SUMMARIZE:
            return self.execute_retrieval_summary_stage(character, prompt)
        if prompt.target_stage == StageName.STRATEGY:
            return self.execute_strategy_stage(character, prompt)
        if prompt.target_stage == StageName.RESPONSE:
            return self.execute_response_stage(character, prompt)

        raise ValueError(f"Unsupported stage target: {prompt.target_stage}")

    def execute_perception_stage(self, character: Character, prompt: StageTestPrompt) -> tuple[Any, dict[str, Any]]:
        initial_context = character.build_initial_context()
        retrieved_context = self.build_retrieved_context(prompt.stage_inputs.get("retrieved_context", ""))
        if prompt.stage_inputs.get("retrieved_context", "") != "":
            perception = character.pipeline.perception_stage.run(
                TurnInput(prompt=prompt.user_query),
                initial_context,
                retrieved_context=retrieved_context,
            )
        else:
            perception = character.pipeline.perception_stage.run(
                TurnInput(prompt=prompt.user_query),
                initial_context,
            )

        return perception, {
            "initial_context": self.to_plain_data(initial_context),
            "retrieved_context": self.to_plain_data(retrieved_context),
        }

    def execute_gap_analysis_stage(self, character: Character, prompt: StageTestPrompt) -> tuple[Any, dict[str, Any]]:
        perception = self.simulate_perception_result(character, prompt)
        gap_analysis = character.pipeline.gap_analysis_stage.run(perception)

        return gap_analysis, {
            "perception": self.to_plain_data(perception),
        }

    def execute_retrieval_summary_stage(self, character: Character, prompt: StageTestPrompt) -> tuple[Any, dict[str, Any]]:
        perception = self.simulate_perception_result(character, prompt)
        raw_context = str(prompt.stage_inputs.get("raw_retrieved_context", "")).strip()
        summary = character.pipeline.retrieval_stage.summarize_retrieved_context(perception, raw_context)

        return summary, {
            "perception": self.to_plain_data(perception),
            "raw_retrieved_context": raw_context,
        }

    def execute_strategy_stage(self, character: Character, prompt: StageTestPrompt) -> tuple[Any, dict[str, Any]]:
        initial_context = character.build_initial_context()
        perception = self.simulate_perception_result(character, prompt)
        gap_analysis = self.simulate_gap_analysis_result(character, prompt)
        retrieved_context = self.simulate_retrieved_context_result(character, prompt)
        strategy = character.pipeline.strategy_stage.run(initial_context, perception, retrieved_context)
        return strategy, {
            "initial_context": self.to_plain_data(initial_context),
            "perception": self.to_plain_data(perception),
            "gap_analysis": self.to_plain_data(gap_analysis),
            "retrieved_context": self.to_plain_data(retrieved_context),
        }

    def execute_response_stage(self, character: Character, prompt: StageTestPrompt) -> tuple[Any, dict[str, Any]]:
        character.initialize_message_loop_context()
        initial_context = character.build_initial_context()
        perception = self.simulate_perception_result(character, prompt)
        gap_analysis = self.simulate_gap_analysis_result(character, prompt)
        retrieved_context = self.simulate_retrieved_context_result(character, prompt)
        strategy = self.simulate_strategy_result(character, prompt)
        response = character.pipeline.response_stage.run(initial_context, perception, retrieved_context, strategy)

        return response, {
            "initial_context": self.to_plain_data(initial_context),
            "perception": self.to_plain_data(perception),
            "gap_analysis": self.to_plain_data(gap_analysis),
            "retrieved_context": self.to_plain_data(retrieved_context),
            "strategy": self.to_plain_data(strategy),
        }

    def build_retrieved_context(self, combined_context: str) -> RetrievedContext:
        if combined_context.strip() == "":
            return RetrievedContext()

        return RetrievedContext(
            combined_context=combined_context,
            memory_context=combined_context,
        )

    def simulate_perception_result(self, character: Character, prompt: StageTestPrompt) -> PerceptionResult:
        payload = self.get_test_payload(prompt, "perception")
        if payload is None:
            return self.execute_perception_stage(character, prompt)[0]

        payload = self.normalize_perception_payload(payload)
        return PerceptionResult(
            raw_prompt=prompt.user_query,
            player_intent=self.read_string(payload, "player_intent", "unknown"),
            player_emotion=self.read_string(payload, "player_emotion", "neutral"),
            request_type=self.read_string(payload, "request_type", "general"),
            topic=self.read_string(payload, "topic", ""),
            is_ambiguous=self.read_bool(payload, "is_ambiguous", False),
            threat_signal=self.read_string(payload, "threat_signal", "none"),
            manipulation_signal=self.read_string(payload, "manipulation_signal", "none"),
            topic_sensitivity=self.read_string(payload, "topic_sensitivity", "normal"),
        )

    def simulate_gap_analysis_result(self, character: Character, prompt: StageTestPrompt) -> GapAnalysisResult:
        payload = self.get_test_payload(prompt, "gap_analysis")
        if payload is None:
            perception = self.simulate_perception_result(character, prompt)
            return character.pipeline.gap_analysis_stage.run(perception)

        payload = self.normalize_gap_analysis_payload(payload)
        return GapAnalysisResult(
            tool_calls=self.parse_tool_calls(payload.get("tool_calls")),
        )

    def simulate_retrieved_context_result(self, character: Character, prompt: StageTestPrompt) -> RetrievedContext:
        raw_combined_context = str(prompt.stage_inputs.get("retrieved_context", "")).strip()
        if raw_combined_context != "":
            return self.build_retrieved_context(raw_combined_context)

        payload = self.get_test_payload(prompt, "retrieved_context")
        if payload is None:
            gap_analysis = self.simulate_gap_analysis_result(character, prompt)
            perception = self.simulate_perception_result(character, prompt)
            return character.pipeline.retrieval_stage.run(perception, gap_analysis)

        payload = self.normalize_retrieved_context_payload(payload)
        return RetrievedContext(
            combined_context=self.read_string(payload, "combined_context", ""),
            memory_context=self.read_string(payload, "memory_context", ""),
            relationship_context=self.read_string(payload, "relationship_context", ""),
            knowledge_context=self.read_string(payload, "knowledge_context", ""),
            social_context=self.read_string(payload, "social_context", ""),
        )

    def simulate_strategy_result(self, character: Character, prompt: StageTestPrompt) -> StrategyResult:
        payload = self.get_test_payload(prompt, "strategy")
        if payload is None:
            initial_context = character.build_initial_context()
            perception = self.simulate_perception_result(character, prompt)
            retrieved_context = self.simulate_retrieved_context_result(character, prompt)
            return character.pipeline.strategy_stage.run(initial_context, perception, retrieved_context)

        payload = self.normalize_strategy_payload(payload)
        immediate_actions = self.read_string_list(payload, "immediate_actions")
        if len(immediate_actions) == 0:
            immediate_actions = ["keep_talking"]

        return StrategyResult(
            intention=self.read_string(payload, "intention", ""),
            conversation_goal=self.read_string(payload, "conversation_goal", "answer_plainly"),
            risk_level=self.read_string(payload, "risk_level", "low"),
            disclosure_level=self.read_string(payload, "disclosure_level", "normal"),
            social_strategy=self.read_string(payload, "social_strategy", "neutral"),
            tone=self.read_string(payload, "tone", "in_character"),
            verbosity=self.read_string(payload, "verbosity", "normal"),
            conversation_move=self.read_string(payload, "conversation_move", "answer"),
            immediate_actions=immediate_actions,
            new_sentiment=self.read_optional_string(payload, "new_sentiment"),
            sentiment_reasoning=self.read_string(payload, "sentiment_reasoning", ""),
        )

    def get_test_payload(self, prompt: StageTestPrompt, stage_key: str) -> dict[str, Any] | None:
        payload_value = prompt.stage_inputs.get(f"{stage_key}_payload")
        if isinstance(payload_value, dict):
            return payload_value

        instruction_value = prompt.stage_inputs.get(f"{stage_key}_prompt")
        if isinstance(instruction_value, dict):
            return instruction_value
        if isinstance(instruction_value, str):
            return self.parse_json_object(instruction_value)

        return None

    def parse_json_object(self, raw_output: str) -> dict[str, Any] | None:
        cleaned = raw_output.strip()
        if cleaned == "":
            return None

        cleaned = re.sub(r"^```(?:json)?", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned).strip()

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            return None

        return payload if isinstance(payload, dict) else None

    def normalize_perception_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "player_intent": payload.get("player_intent", "unknown"),
            "player_emotion": payload.get("player_emotion", "neutral"),
            "request_type": payload.get("request_type", "general"),
            "topic": payload.get("topic", ""),
            "is_ambiguous": payload.get("is_ambiguous", False),
            "threat_signal": payload.get("threat_signal", "none"),
            "manipulation_signal": payload.get("manipulation_signal", "none"),
            "topic_sensitivity": payload.get("topic_sensitivity", "normal"),
        }

    def normalize_gap_analysis_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        tool_calls = payload.get("tool_calls")
        return {
            "tool_calls": tool_calls if isinstance(tool_calls, list) else [],
        }

    def normalize_retrieved_context_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "combined_context": payload.get("combined_context", ""),
            "memory_context": payload.get("memory_context", ""),
            "relationship_context": payload.get("relationship_context", ""),
            "knowledge_context": payload.get("knowledge_context", ""),
            "social_context": payload.get("social_context", ""),
        }

    def normalize_strategy_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "intention": payload.get("intention", ""),
            "conversation_goal": payload.get("conversation_goal", "answer_plainly"),
            "risk_level": payload.get("risk_level", "low"),
            "disclosure_level": payload.get("disclosure_level", "normal"),
            "social_strategy": payload.get("social_strategy", "neutral"),
            "tone": payload.get("tone", "in_character"),
            "verbosity": payload.get("verbosity", "normal"),
            "conversation_move": payload.get("conversation_move", "answer"),
            "immediate_actions": payload.get("immediate_actions", ["keep_talking"]),
            "new_sentiment": payload.get("new_sentiment"),
            "sentiment_reasoning": payload.get("sentiment_reasoning", ""),
        }

    def parse_tool_calls(self, value: Any) -> list[NormalizedToolCall]:
        if not isinstance(value, list):
            return []

        tool_calls: list[NormalizedToolCall] = []
        for item in value:
            if not isinstance(item, dict):
                continue

            tool_name = str(item.get("tool_name", "")).strip()
            reasoning = str(item.get("reasoning", "")).strip()
            if tool_name == "":
                continue

            tool_calls.append(
                NormalizedToolCall(
                    function=NormalizedToolFunction(
                        name=tool_name,
                        arguments={"reasoning": reasoning},
                    )
                )
            )

        return tool_calls

    def read_string(self, payload: dict[str, Any], key: str, default: str) -> str:
        value = payload.get(key)
        return str(value).strip() if isinstance(value, str) and value.strip() != "" else default

    def read_optional_string(self, payload: dict[str, Any], key: str) -> str | None:
        value = payload.get(key)
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped if stripped != "" else None
        return str(value)

    def read_bool(self, payload: dict[str, Any], key: str, default: bool) -> bool:
        value = payload.get(key)
        return value if isinstance(value, bool) else default

    def read_string_list(self, payload: dict[str, Any], key: str) -> list[str]:
        value = payload.get(key)
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip() != ""]

    def evaluate_deterministic_checks(
        self,
        prompt: StageTestPrompt,
        stage_output: Any,
        execution_context: dict[str, Any],
        stage_output_json: str,
    ) -> list[StageEvaluationResult]:
        results: list[StageEvaluationResult] = []
        actual_payload = self.to_plain_data(stage_output)
        combined_payload = {
            "stage_output": actual_payload,
            "execution_context": execution_context,
        }

        for check in prompt.deterministic_checks:
            actual_value = self.lookup_path(combined_payload, check.path)
            passed = self.apply_operator(actual_value, check.operator, check.value)
            results.append(
                StageEvaluationResult(
                    user_query=prompt.user_query,
                    source_category=prompt.source_category,
                    target_stage=prompt.target_stage,
                    expectation_mode=prompt.expectation_mode,
                    metric_name=check.metric_name,
                    passed=passed,
                    score=1.0 if passed else 0.0,
                    explanation=self.build_deterministic_explanation(check, actual_value, check.value, passed),
                    expected_value=check.value,
                    actual_value=self.serialize_scalar(actual_value),
                    stage_output=stage_output_json,
                    notes=prompt.notes,
                )
            )

        return results

    def evaluate_judge_metrics(
        self,
        character: Character,
        prompt: StageTestPrompt,
        stage_output: Any,
        execution_context: dict[str, Any],
        stage_output_json: str,
    ) -> list[StageEvaluationResult]:
        active_metrics = [
            metric for metric in prompt.judge_metrics
            if metric.metric_name not in self.UNSUPPORTED_JUDGE_METRICS
        ]
        if len(active_metrics) != len(prompt.judge_metrics):
            skipped_metrics = sorted(
                metric.metric_name for metric in prompt.judge_metrics
                if metric.metric_name in self.UNSUPPORTED_JUDGE_METRICS
            )
            logger.info("Skipping unsupported judge metrics for %s: %s", prompt.target_stage.value, skipped_metrics)

        if len(active_metrics) == 0:
            return [
                StageEvaluationResult(
                    user_query=prompt.user_query,
                    source_category=prompt.source_category,
                    target_stage=prompt.target_stage,
                    expectation_mode=prompt.expectation_mode,
                    metric_name="judge_metrics_missing",
                    passed=False,
                    score=0.0,
                    explanation="No supported judge metrics were configured for this judge-mode row.",
                    expected_value=None,
                    actual_value=None,
                    stage_output=stage_output_json,
                    notes=prompt.notes,
                )
            ]

        judge_prompt = self.build_judge_prompt(character, prompt, stage_output, execution_context, active_metrics)
        raw_output = self.provider.generate(judge_prompt)
        metric_results = self.parse_judge_output(raw_output, active_metrics)

        return [
            StageEvaluationResult(
                user_query=prompt.user_query,
                source_category=prompt.source_category,
                target_stage=prompt.target_stage,
                expectation_mode=prompt.expectation_mode,
                metric_name=result.metric_name,
                passed=result.passed,
                score=float(result.score),
                explanation=result.explanation,
                expected_value="judge_rubric",
                actual_value=result.score,
                stage_output=stage_output_json,
                notes=prompt.notes,
            )
            for result in metric_results
        ]

    def build_judge_prompt(
        self,
        character: Character,
        prompt: StageTestPrompt,
        stage_output: Any,
        execution_context: dict[str, Any],
        active_metrics: list[StageJudgeMetric],
    ) -> str:
        rubric_items: list[str] = []
        for metric in active_metrics:
            default_guidance = self.STAGE_METRIC_GUIDANCE.get(prompt.target_stage, {}).get(metric.metric_name, "")
            guidance = (metric.guidance.strip() or default_guidance).format(character_name=character.name)
            rubric_items.append(
                json.dumps(
                    {
                        "metric_name": metric.metric_name,
                        "guidance": guidance,
                    },
                    ensure_ascii=True,
                )
            )

        return "\n".join([
            "You are evaluating one LLM-driven NPC stage output for reasonability.",
            "Return only valid JSON.",
            "Score each metric from 0.0 to 1.0, where 1.0 is a perfect pass and 0.0 is a total failure.",
            "Set passed=true when the score is at least 0.5, otherwise false.",
            "Always include a short explanation that a human can understand.",
            f"Character name: {character.name}",
            f"Character definition: {character.pl_list}",
            f"Character knowledge: {character.knowledge}",
            f"Character past: {character.past}",
            f"Character relations: {character.relations}",
            f"Character sentiment: {character.sentiment}",
            f"Example dialogues: {character.ali_chat}",
            f"Source category: {prompt.source_category.value}",
            f"Target stage: {prompt.target_stage.value}",
            f"User query: {prompt.user_query}",
            "Stage inputs and supporting context:",
            self.serialize_value(execution_context),
            "Stage output to evaluate:",
            self.serialize_value(stage_output),
            "Metrics to score:",
            "\n".join(rubric_items),
            "Return JSON with this exact shape:",
            '{"metrics":[{"metric_name":"string","score":1.0,"passed":true,"explanation":"short explanation"}]}',
        ])

    def parse_judge_output(self, raw_output: str, metrics_to_score: list[StageJudgeMetric]) -> list[StageJudgeMetricResult]:
        cleaned = raw_output.strip()
        cleaned = re.sub(r"^```(?:json)?", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned).strip()

        fallback_metrics = [
            StageJudgeMetricResult(
                metric_name=metric.metric_name,
                score=0.0,
                passed=False,
                explanation="Judge output could not be parsed, so this metric fell back to a failing result.",
            )
            for metric in metrics_to_score
        ]

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            return fallback_metrics

        metrics = payload.get("metrics")
        if not isinstance(metrics, list):
            return fallback_metrics

        parsed_results: list[StageJudgeMetricResult] = []
        for metric_payload in metrics:
            if not isinstance(metric_payload, dict):
                continue

            metric_name = str(metric_payload.get("metric_name", "")).strip()
            if metric_name == "":
                continue

            explanation = str(metric_payload.get("explanation", "")).strip()
            try:
                score = float(metric_payload.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0

            passed_value = metric_payload.get("passed", score >= 0.5)
            if isinstance(passed_value, str):
                passed = passed_value.strip().lower() == "true"
            else:
                passed = bool(passed_value)

            parsed_results.append(
                StageJudgeMetricResult(
                    metric_name=metric_name,
                    score=max(0.0, min(1.0, score)),
                    passed=passed,
                    explanation=explanation if explanation != "" else "No explanation returned by judge.",
                )
            )

        if len(parsed_results) == 0:
            return fallback_metrics

        requested_metric_names = {metric.metric_name for metric in metrics_to_score}
        parsed_metric_names = {metric.metric_name for metric in parsed_results}

        for missing_metric_name in sorted(requested_metric_names - parsed_metric_names):
            parsed_results.append(
                StageJudgeMetricResult(
                    metric_name=missing_metric_name,
                    score=0.0,
                    passed=False,
                    explanation="Judge did not return this requested metric.",
                )
            )

        return parsed_results

    def apply_operator(self, actual_value: Any, operator: str, expected_value: Any) -> bool:
        if operator == "equals":
            return actual_value == expected_value
        if operator == "contains":
            if isinstance(actual_value, str):
                return str(expected_value) in actual_value
            if isinstance(actual_value, list):
                return expected_value in actual_value
            return False
        raise ValueError(f"Unsupported operator '{operator}'. Supported operators are: equals, contains.")

    def build_deterministic_explanation(
        self,
        check: StageDeterministicCheck,
        actual_value: Any,
        expected_value: Any,
        passed: bool,
    ) -> str:
        state = "passed" if passed else "failed"
        return (
            f"Deterministic check {state}: path '{check.path}' with operator '{check.operator}' "
            f"expected {self.serialize_scalar(expected_value)} and got {self.serialize_scalar(actual_value)}."
        )

    def lookup_path(self, payload: Any, path: str) -> Any:
        current = payload
        for segment in path.split("."):
            if isinstance(current, dict):
                current = current.get(segment)
            else:
                return None
        return current

    def to_plain_data(self, value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, list):
            return [self.to_plain_data(item) for item in value]
        if isinstance(value, dict):
            return {key: self.to_plain_data(item) for key, item in value.items()}
        return value

    def serialize_value(self, value: Any) -> str:
        return json.dumps(self.to_plain_data(value), ensure_ascii=True)

    def serialize_scalar(self, value: Any) -> Any:
        plain_value = self.to_plain_data(value)
        if isinstance(plain_value, (dict, list)):
            return json.dumps(plain_value, ensure_ascii=True)
        return plain_value
