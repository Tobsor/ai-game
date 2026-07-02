import unittest
from types import SimpleNamespace

from models import PromptCategory, StageExpectationMode, StageJudgeMetric, StageName, StageTestPrompt
from test.AgentTest import AgentTest
from workflow.models import GapAnalysisResult, InitialContext, ResponseResult

import json


class RecordingGapAnalysisStage:
    def __init__(self):
        self.last_perception = None

    def run(self, perception):
        self.last_perception = perception
        return GapAnalysisResult(tool_calls=[])


class FailingStage:
    def run(self, *args, **kwargs):
        raise AssertionError("This stage should not run during simulated predecessor setup.")


class RecordingResponseStage:
    def __init__(self):
        self.last_args = None

    def run(self, initial_context, perception, retrieved_context, strategy):
        self.last_args = (initial_context, perception, retrieved_context, strategy)
        return ResponseResult(reply="simulated reply", turn_prompt="turn prompt")


class FakeCharacter:
    def __init__(self):
        self.name = "Mira"
        self.pl_list = "Helpful trader"
        self.ali_chat = "Welcome, traveler."
        self.knowledge = "Knows the market"
        self.past = "Grew up nearby"
        self.relations = "Neutral toward the player"
        self.sentiment = "neutral"
        self.situation = "At the market"
        self.db = SimpleNamespace(messages=[], response_context_initialized=False)
        self.initialized = False
        self.pipeline = SimpleNamespace(
            perception_stage=FailingStage(),
            gap_analysis_stage=RecordingGapAnalysisStage(),
            retrieval_stage=FailingStage(),
            strategy_stage=FailingStage(),
            response_stage=RecordingResponseStage(),
        )

    def build_initial_context(self):
        return InitialContext(
            character_name=self.name,
            situation=self.situation,
            sentiment=self.sentiment,
            character_definition=self.pl_list,
            example_dialogues="Welcome, traveler.",
        )

    def initialize_message_loop_context(self):
        self.initialized = True


class AgentTestTests(unittest.TestCase):
    def test_deterministic_checks_are_supported(self):
        agent_test = AgentTest()
        prompt = StageTestPrompt(
            user_query="What are the market rules?",
            source_category=PromptCategory.GENERAL,
            target_stage=StageName.GAP_ANALYSIS,
            expectation_mode=StageExpectationMode.DETERMINISTIC,
            stage_inputs={
                "perception_payload": {
                    "player_intent": "seek_information",
                    "player_emotion": "curious",
                    "request_type": "question",
                    "topic": "market rules",
                }
            },
            deterministic_checks=[
                {
                    "metric_name": "topic_is_market_rules",
                    "path": "execution_context.perception.topic",
                    "operator": "equals",
                    "value": "market rules",
                }
            ],
        )

        stage_output, execution_context = agent_test.execute_gap_analysis_stage(FakeCharacter(), prompt)
        results = agent_test.evaluate_deterministic_checks(
            prompt=prompt,
            stage_output=stage_output,
            execution_context=execution_context,
            stage_output_json=agent_test.serialize_value(stage_output),
        )

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].passed)
        self.assertEqual(results[0].score, 1.0)

    def test_parse_judge_output_normalizes_zero_to_one_scores(self):
        agent_test = AgentTest()
        metrics = [StageJudgeMetric(metric_name="field_validity")]

        results = agent_test.parse_judge_output(
            '{"metrics":[{"metric_name":"field_validity","score":0.75,"passed":true,"explanation":"Strong match."}]}',
            metrics,
        )

        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0].score, 0.75)
        self.assertTrue(results[0].passed)

    def test_reasoning_relevance_metric_is_excluded_from_judge_payload(self):
        agent_test = AgentTest()
        character = FakeCharacter()
        prompt = StageTestPrompt(
            user_query="What should I know?",
            source_category=PromptCategory.GENERAL,
            target_stage=StageName.GAP_ANALYSIS,
            expectation_mode=StageExpectationMode.JUDGE,
            judge_metrics=[
                StageJudgeMetric(metric_name="tool_relevance"),
                StageJudgeMetric(metric_name="reasoning_relevance"),
            ],
        )

        judge_prompt = agent_test.build_judge_prompt(
            character=character,
            prompt=prompt,
            stage_output={},
            execution_context={},
            active_metrics=[StageJudgeMetric(metric_name="tool_relevance")],
        )

        self.assertIn("tool_relevance", judge_prompt)
        self.assertNotIn("reasoning_relevance", judge_prompt)

    def test_gap_analysis_uses_simulated_perception_result(self):
        character = FakeCharacter()
        agent_test = AgentTest()
        prompt = StageTestPrompt(
            user_query="What are the market rules?",
            source_category=PromptCategory.GENERAL,
            target_stage=StageName.GAP_ANALYSIS,
            expectation_mode=StageExpectationMode.JUDGE,
            stage_inputs={"perception_payload": {
                "player_intent": "seek_information",
                "player_emotion": "curious",
                "request_type": "question",
                "topic": "market rules",
                "is_ambiguous": False,
                "threat_signal": "none",
                "manipulation_signal": "none",
                "topic_sensitivity": "normal",
            }},
        )

        result, execution_context = agent_test.execute_gap_analysis_stage(character, prompt)

        self.assertEqual(result.tool_calls, [])
        self.assertEqual(character.pipeline.gap_analysis_stage.last_perception.player_intent, "seek_information")
        self.assertEqual(execution_context["perception"]["topic"], "market rules")

    def test_response_stage_uses_simulated_predecessor_outputs(self):
        character = FakeCharacter()
        agent_test = AgentTest()
        prompt = StageTestPrompt(
            user_query="What do you have for sale?",
            source_category=PromptCategory.GENERAL,
            target_stage=StageName.RESPONSE,
            expectation_mode=StageExpectationMode.JUDGE,
            stage_inputs={
                "perception_payload": {
                "player_intent": "buy_goods",
                "player_emotion": "curious",
                "request_type": "question",
                "topic": "wares",
                "is_ambiguous": False,
                "threat_signal": "none",
                "manipulation_signal": "none",
                "topic_sensitivity": "normal",
                },
                "gap_analysis_payload": {
                    "tool_calls": [
                        {
                            "tool_name": "recall_knowledge",
                            "reasoning": "Need product details before answering.",
                        }
                    ]
                },
                "retrieved_context_payload": {
                    "combined_context": "Mira sells herbs and travel supplies.",
                    "memory_context": "",
                    "relationship_context": "",
                    "knowledge_context": "Mira sells herbs and travel supplies.",
                    "social_context": "",
                },
                "strategy_payload": {
                    "intention": "make a sale",
                    "conversation_goal": "invite trade",
                    "risk_level": "low",
                    "disclosure_level": "normal",
                    "social_strategy": "welcoming",
                    "tone": "friendly",
                    "verbosity": "brief",
                    "conversation_move": "answer_then_offer",
                    "immediate_actions": ["open_trade", "keep_talking"],
                    "new_sentiment": None,
                    "sentiment_reasoning": "",
                },
            },
        )

        result, execution_context = agent_test.execute_response_stage(character, prompt)
        _, perception, retrieved_context, strategy = character.pipeline.response_stage.last_args

        self.assertTrue(character.initialized)
        self.assertEqual(result.reply, "simulated reply")
        self.assertEqual(perception.player_intent, "buy_goods")
        self.assertEqual(retrieved_context.combined_context, "Mira sells herbs and travel supplies.")
        self.assertEqual(strategy.immediate_actions, ["open_trade", "keep_talking"])
        self.assertEqual(execution_context["gap_analysis"]["tool_calls"][0]["function"]["name"], "recall_knowledge")

    def test_json_string_stage_instruction_is_shaped_deterministically(self):
        character = FakeCharacter()
        agent_test = AgentTest()
        prompt = StageTestPrompt(
            user_query="What do you have for sale?",
            source_category=PromptCategory.GENERAL,
            target_stage=StageName.RESPONSE,
            expectation_mode=StageExpectationMode.JUDGE,
            stage_inputs={
                "perception_prompt": json.dumps({
                    "player_intent": "buy_goods",
                    "player_emotion": "curious",
                    "request_type": "question",
                    "topic": "wares",
                }),
                "gap_analysis_prompt": json.dumps({
                    "tool_calls": [
                        {
                            "tool_name": "recall_knowledge",
                            "reasoning": "debug only",
                        }
                    ]
                }),
                "retrieved_context_prompt": json.dumps({
                    "combined_context": "Mira sells herbs and travel supplies.",
                    "knowledge_context": "Mira sells herbs and travel supplies.",
                }),
                "strategy_prompt": json.dumps({
                    "conversation_goal": "invite trade",
                    "immediate_actions": ["open_trade"],
                }),
            },
        )

        _, execution_context = agent_test.execute_response_stage(character, prompt)

        self.assertEqual(execution_context["perception"]["player_intent"], "buy_goods")
        self.assertEqual(execution_context["retrieved_context"]["knowledge_context"], "Mira sells herbs and travel supplies.")
        self.assertEqual(execution_context["strategy"]["conversation_goal"], "invite trade")
        self.assertEqual(execution_context["gap_analysis"]["tool_calls"][0]["function"]["arguments"]["reasoning"], "debug only")

    def test_retrieval_guidance_includes_knowledge_scope_alignment(self):
        agent_test = AgentTest()

        self.assertIn(
            "knowledge_scope_alignment",
            agent_test.STAGE_METRIC_GUIDANCE[StageName.RETRIEVAL_SUMMARIZE],
        )


if __name__ == "__main__":
    unittest.main()
