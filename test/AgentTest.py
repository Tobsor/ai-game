import csv
import json
import os
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
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
from workflow.models import AppraisalResult, EmotionResult, GapAnalysisResult, NPCPerception, PerceptionResult, PerceptionTopic, RetrievedContext, StrategyResult, TurnInput
from test.stage_test_utils import (
    extend_vector_state_from_config,
    load_vector_state_config,
    save_vector_snapshot,
)

script_dir = os.path.dirname(__file__)
configure_logging()
logger = get_logger(__name__)


class AgentTest:
    SUPPORTED_STAGES = {
        StageName.PERCEPTION,
        StageName.GAP_ANALYSIS,
        StageName.RETRIEVAL_RUN,
        StageName.APPRAISAL,
        StageName.STRATEGY,
        StageName.RESPONSE,
    }

    STAGE_METRIC_GUIDANCE: dict[StageName, dict[str, str]] = {
        StageName.PERCEPTION: {
            "author_note": "Compare the perception stage output for the user input against the author note, which defines the expected outcome of the stage process. Score semantic agreement with that expected interpretation, including its nuances, rather than exact wording. If no author note is provided, return score 0.0, passed=false, and explain that the expected outcome is missing and alignment cannot be evaluated.",
            "field_validity": "Check whether the populated perception fields follow the requested schema and avoid obviously invalid defaults when the prompt provides a clear signal.",
            "intent_plausibility": "Check whether player_intent is a reasonable interpretation of the user's message from {character_name}'s perspective.",
            "emotion_plausibility": "Check whether player_emotion is a reasonable interpretation of the user's message.",
            "threat_manipulation_sensitivity": "Check whether threat_signal and manipulation_signal reasonably capture hostile, coercive, flattering, or manipulative content.",
            "ambiguity_appropriateness": "Check whether is_ambiguous matches how clear or unclear the user message actually is.",
        },
        StageName.GAP_ANALYSIS: {
            "memory_necessity": "Check whether missing prior events or personal memories require recall_memory, and whether unnecessary memory retrieval is avoided.",
            "knowledge_necessity": "Check whether missing world, faction, or topic facts require recall_knowledge, and whether unnecessary knowledge retrieval is avoided.",
            "interaction_memory_necessity": "Check whether past player interactions need retrieval. Specific events may require recall_memory; trust, relationship history, and shared context may require recall_relationship. Allow overlap when justified by the missing information.",
            "context_completeness": "Check whether available context fully, partially, or inadequately answers the request. Complete context should produce no calls; partial context should trigger only remaining gaps; irrelevant context must not suppress retrieval.",
            "author_note": "Compare the complete retrieval decision against the author note, which defines the expected outcome. Judge semantic agreement, not exact wording. If the note is missing, return score 0.0, passed=false, and explain that alignment cannot be evaluated.",
            "tool_relevance": "Check whether the selected retrieval tools are relevant to the information {character_name} would need.",
            "tool_minimality": "Check whether tool calls avoid unrelated, redundant, or duplicate retrieval requests.",
        },
        StageName.RETRIEVAL_RUN: {
            "fetch_relevance": "Assess whether the supplied retrieval operations address the information {character_name} needs for the player's request, using their resulting context as supporting evidence.",
            "context_relevance": "Assess the relevance of the individual retrieved context fields to the player's request.",
            "summary_quality": "Assess whether combined_context accurately and concisely summarizes the individual retrieved context fields, retains relevant facts, excludes irrelevant material, and avoids distortion or unsupported additions. Judge correctness against those source fields. Do not penalize the summary for omitting facts retrieval never supplied; evaluate those discrepancies under author_note instead. Return one score and explain material omissions, inaccuracies, unsupported claims, or unnecessary content.",
            "knowledge_scope_alignment": "Assess whether the retrieved and summarized information is reasonably within {character_name}'s knowledge or retrieval scope, given the supplied character and test context.",
            "author_note": "Assess semantic agreement between the complete retrieval outcome and the author note, which defines the expected outcome. Distinguish expected retrieved facts from expected summary content when specified. If the note is missing or blank, return score 0.0, passed=false, and explain that expected-outcome alignment cannot be assessed.",
        },
        StageName.APPRAISAL: {
            "appraisal_plausibility": "Check whether the appraisal reasonably evaluates the final perception against {character_name}'s values, goals, relationship state, and situation.",
            "emotion_coherence": "Check whether the emotional reaction follows naturally from the appraisal and fits {character_name}.",
            "attribution_reasonability": "Check whether the attribution source and responsibility are reasonable for the perceived event.",
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
            vector_state_context = self.prepare_vector_state(character, prompt)
            stage_output, execution_context = self.execute_stage(character, prompt)
            execution_context["vector_state"] = vector_state_context
            snapshot_context = self.save_vector_state_snapshot(character, prompt)
            if snapshot_context is not None:
                execution_context["vector_snapshot"] = snapshot_context
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
        if prompt.target_stage == StageName.RETRIEVAL_RUN:
            return self.execute_retrieval_run_stage(character, prompt)
        if prompt.target_stage == StageName.APPRAISAL:
            return self.execute_appraisal_stage(character, prompt)
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
        if self.get_test_payload(prompt, "perception") is None:
            raise ValueError("Gap analysis tests require an explicit perception_payload.")
        perception = self.simulate_perception_result(character, prompt)
        available_context = prompt.stage_inputs.get("available_context", "")
        if not isinstance(available_context, str):
            raise ValueError("available_context must be a string.")
        gap_analysis = character.pipeline.gap_analysis_stage.run(perception, available_context=available_context)
        tool_names = [call.function.name for call in gap_analysis.tool_calls]

        return gap_analysis, {
            "perception": self.to_plain_data(perception),
            "available_context": available_context,
            "gap_analysis_tool_names": sorted(set(tool_names)),
            "gap_analysis_duplicate_count": len(tool_names) - len(set(tool_names)),
            "gap_analysis_arguments_valid": all(
                isinstance(call.function.arguments, dict)
                and isinstance(call.function.arguments.get("reasoning"), str)
                and bool(call.function.arguments["reasoning"].strip())
                for call in gap_analysis.tool_calls
            ),
        }

    def execute_retrieval_run_stage(self, character: Character, prompt: StageTestPrompt) -> tuple[Any, dict[str, Any]]:
        missing_inputs = [
            f"{stage_key}_payload"
            for stage_key in ("perception", "gap_analysis")
            if self.get_test_payload(prompt, stage_key) is None
        ]
        if missing_inputs:
            raise ValueError("Retrieval tests require explicit inputs: " + ", ".join(missing_inputs))
        perception = self.simulate_perception_result(character, prompt)
        gap_analysis = self.simulate_gap_analysis_result(character, prompt)
        retrieved_context = character.pipeline.retrieval_stage.run(perception, gap_analysis)

        return retrieved_context, {
            "perception": self.to_plain_data(perception),
            "gap_analysis": self.to_plain_data(gap_analysis),
            "gap_analysis_tool_names": [
                tool_call.function.name for tool_call in gap_analysis.tool_calls
            ],
        }

    def execute_appraisal_stage(self, character: Character, prompt: StageTestPrompt) -> tuple[Any, dict[str, Any]]:
        initial_context = character.build_initial_context()
        perception = self.simulate_perception_result(character, prompt)
        retrieved_context = self.simulate_retrieved_context_result(character, prompt)
        appraisal, emotion = character.pipeline.appraisal_stage.run(initial_context, perception, retrieved_context)

        return {"appraisal": appraisal, "emotion": emotion}, {
            "initial_context": self.to_plain_data(initial_context),
            "perception": self.to_plain_data(perception),
            "retrieved_context": self.to_plain_data(retrieved_context),
        }

    def execute_strategy_stage(self, character: Character, prompt: StageTestPrompt) -> tuple[Any, dict[str, Any]]:
        initial_context = character.build_initial_context()
        perception = self.simulate_perception_result(character, prompt)
        gap_analysis = self.simulate_gap_analysis_result(character, prompt)
        retrieved_context = self.simulate_retrieved_context_result(character, prompt)
        appraisal, emotion = self.simulate_appraisal_emotion_result(character, prompt)
        strategy = character.pipeline.strategy_stage.run(initial_context, perception, retrieved_context, appraisal, emotion)
        return strategy, {
            "initial_context": self.to_plain_data(initial_context),
            "perception": self.to_plain_data(perception),
            "gap_analysis": self.to_plain_data(gap_analysis),
            "retrieved_context": self.to_plain_data(retrieved_context),
            "appraisal": self.to_plain_data(appraisal),
            "emotion": self.to_plain_data(emotion),
        }

    def execute_response_stage(self, character: Character, prompt: StageTestPrompt) -> tuple[Any, dict[str, Any]]:
        character.initialize_message_loop_context()
        initial_context = character.build_initial_context()
        perception = self.simulate_perception_result(character, prompt)
        gap_analysis = self.simulate_gap_analysis_result(character, prompt)
        retrieved_context = self.simulate_retrieved_context_result(character, prompt)
        appraisal, emotion = self.simulate_appraisal_emotion_result(character, prompt)
        strategy = self.simulate_strategy_result(character, prompt)
        response = character.pipeline.response_stage.run(initial_context, perception, retrieved_context, appraisal, emotion, strategy)

        return response, {
            "initial_context": self.to_plain_data(initial_context),
            "perception": self.to_plain_data(perception),
            "gap_analysis": self.to_plain_data(gap_analysis),
            "retrieved_context": self.to_plain_data(retrieved_context),
            "appraisal": self.to_plain_data(appraisal),
            "emotion": self.to_plain_data(emotion),
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
            summary=self.read_string(payload, "summary", ""),
            npc_perception=NPCPerception(
                perceived_intent=self.read_string_list(payload["npc_perception"], "perceived_intent"),
                perceived_attitude=self.read_string_list(payload["npc_perception"], "perceived_attitude"),
                player_intent=self.read_string(payload["npc_perception"], "player_intent", "unknown"),
                player_emotion=self.read_string(payload["npc_perception"], "player_emotion", "neutral"),
                threat_signal=self.read_string(payload["npc_perception"], "threat_signal", "none"),
                manipulation_signal=self.read_string(payload["npc_perception"], "manipulation_signal", "none"),
                topic_sensitivity=self.read_string(payload["npc_perception"], "topic_sensitivity", "normal"),
            ),
            request_type=self.read_string(payload, "request_type", "general"),
            topic=PerceptionTopic(
                primary=self.read_string(payload["topic"], "primary", ""),
                related=self.read_string_list(payload["topic"], "related"),
                retrieval_queries=self.read_string_list(payload["topic"], "retrieval_queries"),
            ),
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

    def simulate_appraisal_emotion_result(self, character: Character, prompt: StageTestPrompt) -> tuple[AppraisalResult, EmotionResult]:
        payload = self.get_test_payload(prompt, "appraisal")
        if payload is None:
            initial_context = character.build_initial_context()
            perception = self.simulate_perception_result(character, prompt)
            retrieved_context = self.simulate_retrieved_context_result(character, prompt)
            return character.pipeline.appraisal_stage.run(initial_context, perception, retrieved_context)

        payload = self.normalize_appraisal_payload(payload)
        emotion_payload = self.normalize_emotion_payload(self.get_test_payload(prompt, "emotion") or payload.get("emotion", {}))
        appraisal_payload = payload.get("appraisal", payload)
        attribution_payload = appraisal_payload.get("attribution", {})
        if not isinstance(attribution_payload, dict):
            attribution_payload = {}
        return (
            AppraisalResult(
                relevance=self.read_float(appraisal_payload, "relevance", 0.0, 0.0, 1.0),
                valence=self.read_float(appraisal_payload, "valence", 0.0, -1.0, 1.0),
                goal_impact=self.read_float(appraisal_payload, "goal_impact", 0.0, -1.0, 1.0),
                social_self_impact=self.read_float(appraisal_payload, "social_self_impact", 0.0, -1.0, 1.0),
                threat=self.read_float(appraisal_payload, "threat", 0.0, 0.0, 1.0),
                control=self.read_float(appraisal_payload, "control", 0.5, 0.0, 1.0),
                attribution_source=self.read_string(attribution_payload, "source", "unknown"),
                attribution_responsibility=self.read_float(attribution_payload, "responsibility", 0.0, 0.0, 1.0),
                summary=self.read_string(appraisal_payload, "summary", ""),
            ),
            EmotionResult(
                primary=self.read_string(emotion_payload, "primary", "neutral"),
                secondary=self.read_string_list(emotion_payload, "secondary"),
                intensity=self.read_float(emotion_payload, "intensity", 0.0, 0.0, 1.0),
            ),
        )

    def simulate_strategy_result(self, character: Character, prompt: StageTestPrompt) -> StrategyResult:
        payload = self.get_test_payload(prompt, "strategy")
        if payload is None:
            initial_context = character.build_initial_context()
            perception = self.simulate_perception_result(character, prompt)
            retrieved_context = self.simulate_retrieved_context_result(character, prompt)
            appraisal, emotion = self.simulate_appraisal_emotion_result(character, prompt)
            return character.pipeline.strategy_stage.run(initial_context, perception, retrieved_context, appraisal, emotion)

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
        npc_perception = payload.get("npc_perception")
        if not isinstance(npc_perception, dict):
            npc_perception = {}

        topic = payload.get("topic")
        if isinstance(topic, dict):
            normalized_topic = {
                "primary": topic.get("primary", ""),
                "related": topic.get("related", []),
                "retrieval_queries": topic.get("retrieval_queries", []),
            }
        else:
            normalized_topic = {
                "primary": "",
                "related": [],
                "retrieval_queries": [],
            }

        return {
            "npc_perception": {
                "player_intent": npc_perception.get("player_intent", "unknown"),
                "player_emotion": npc_perception.get("player_emotion", "neutral"),
                "threat_signal": npc_perception.get("threat_signal", "none"),
                "manipulation_signal": npc_perception.get("manipulation_signal", "none"),
                "topic_sensitivity": npc_perception.get("topic_sensitivity", "normal"),
                "perceived_intent": npc_perception.get("perceived_intent", []),
                "perceived_attitude": npc_perception.get("perceived_attitude", []),
            },
            "request_type": payload.get("request_type", "general"),
            "topic": normalized_topic,
            "summary": payload.get("summary", ""),
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

    def normalize_appraisal_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        appraisal = payload.get("appraisal", payload)
        if not isinstance(appraisal, dict):
            appraisal = {}
        emotion = payload.get("emotion", {})
        return {
            "appraisal": appraisal,
            "emotion": emotion if isinstance(emotion, dict) else {},
        }

    def normalize_emotion_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            payload = {}
        return {
            "primary": payload.get("primary", "neutral"),
            "secondary": payload.get("secondary", []),
            "intensity": payload.get("intensity", 0.0),
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

    def read_float(self, payload: dict[str, Any], key: str, default: float, minimum: float, maximum: float) -> float:
        value = payload.get(key, default)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, min(maximum, parsed))

    def read_string_list(self, payload: dict[str, Any], key: str) -> list[str]:
        value = payload.get(key)
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip() != ""]

    def prepare_vector_state(self, character: Character, prompt: StageTestPrompt) -> dict[str, Any]:
        config_path_value = prompt.stage_inputs.get("vector_state_config")
        if not isinstance(config_path_value, str) or config_path_value.strip() == "":
            return {}

        config_path = self.resolve_test_data_path(config_path_value)
        config = load_vector_state_config(config_path)
        result = extend_vector_state_from_config(character.db, config)
        result["config_path"] = str(config_path)
        return result

    def save_vector_state_snapshot(self, character: Character, prompt: StageTestPrompt) -> dict[str, Any] | None:
        snapshot_path_value = prompt.stage_inputs.get("snapshot_path")
        config_path_value = prompt.stage_inputs.get("vector_state_config")
        if not isinstance(snapshot_path_value, str) or snapshot_path_value.strip() == "":
            return None
        if not isinstance(config_path_value, str) or config_path_value.strip() == "":
            return None

        snapshot_path = self.resolve_test_data_path(snapshot_path_value)
        config_path = self.resolve_test_data_path(config_path_value)
        return save_vector_snapshot(character.db, snapshot_path, config_path)

    def resolve_test_data_path(self, value: str) -> Path:
        raw_path = Path(value)
        if raw_path.is_absolute():
            return raw_path

        repo_root = Path(script_dir).parent
        return repo_root / raw_path

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
                    input_context=execution_context.get("available_context", "") if prompt.target_stage == StageName.GAP_ANALYSIS else "",
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
            StageJudgeMetric(metric_name=name, guidance=guidance)
            for name, guidance in self.STAGE_METRIC_GUIDANCE.get(prompt.target_stage, {}).items()
        ]

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
                    explanation="No judge metrics are defined for this stage in AgentTest.py.",
                    expected_value=None,
                    actual_value=None,
                    stage_output=stage_output_json,
                    notes=prompt.notes,
                    input_context=execution_context.get("available_context", "") if prompt.target_stage == StageName.GAP_ANALYSIS else "",
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
                input_context=execution_context.get("available_context", "") if prompt.target_stage == StageName.GAP_ANALYSIS else "",
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
            "Do not use vague explanations such as 'the interpretation is supported' or 'it passes expectations'. Pinpoint the concrete prompt phrase, output field value, character-context detail, or author-guidance detail that makes the result reasonable or unreasonable.",
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
            *(["Author note (expected stage outcome):", prompt.notes.strip() or "No author note provided."]
              if prompt.target_stage in {StageName.PERCEPTION, StageName.GAP_ANALYSIS, StageName.RETRIEVAL_RUN} else []),
            "Stage inputs and supporting context:",
            self.serialize_value(execution_context),
            "Stage output to evaluate:",
            self.serialize_value(stage_output),
            "Metrics to score:",
            "Return exactly one result for each listed metric and no other metrics.",
            "\n".join(rubric_items),
            "Return JSON with this exact shape:",
            '{"metrics":[{"metric_name":"string","score":1.0,"passed":true,"explanation":"short explanation"}]}',
        ])

    def describe_stage_output_schema(self, stage_name: StageName) -> str:
        if stage_name == StageName.PERCEPTION:
            return "\n".join([
                "Evaluate only these PerceptionStage output fields:",
                "summary: string",
                "npc_perception: object with perceived_intent, perceived_attitude, threat_signal, manipulation_signal, topic_sensitivity, player_intent, and player_emotion",
                "npc_perception.perceived_intent: list[string]",
                "npc_perception.perceived_attitude: list[string]",
                "npc_perception.threat_signal: string",
                "npc_perception.manipulation_signal: string",
                "npc_perception.topic_sensitivity: string",
                "npc_perception.player_intent: string",
                "npc_perception.player_emotion: string",
                "topic: object grouping all theme and retrieval related data",
                "topic.primary: string",
                "topic.related: list[string]",
                "topic.retrieval_queries: list[string]",
                "request_type: string",
                "Do not expect or judge confidence, is_ambiguous, target, relevant_topics, or flat topic strings; they are not part of the produced PerceptionStage contract.",
                "Do not judge raw_prompt, stage_prompt, tool_calls, or retrieval_reasoning as model-produced perception fields; they are runtime or harness metadata.",
            ])

        return "Use the serialized stage output fields shown below as the schema for this stage."

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

        metrics = payload.get("metrics") if isinstance(payload, dict) else None
        if not isinstance(metrics, list):
            return fallback_metrics

        parsed_results: list[StageJudgeMetricResult] = []
        requested_metric_names = {metric.metric_name for metric in metrics_to_score}
        seen_metric_names: set[str] = set()
        for metric_payload in metrics:
            if not isinstance(metric_payload, dict):
                continue

            metric_name = str(metric_payload.get("metric_name", "")).strip()
            if metric_name not in requested_metric_names or metric_name in seen_metric_names:
                continue
            seen_metric_names.add(metric_name)

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

        results_by_name = {result.metric_name: result for result in parsed_results}
        return [results_by_name[metric.metric_name] for metric in metrics_to_score]

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
