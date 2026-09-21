import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from models import StageTestPrompt
from test.AgentTest import AgentTest
from test.build_strategy_fixtures import build_rows
from workflow.stages.strategy_stage import StrategyStage


def fixture_prompts():
    for row in build_rows():
        yield StageTestPrompt(**{**row, "stage_inputs": json.loads(row["stage_inputs"]), "deterministic_checks": []})


class StrategyStageTests(unittest.TestCase):
    def setUp(self):
        self.harness = AgentTest.__new__(AgentTest)
        self.agent = Mock()
        self.agent.parse_output.side_effect = lambda content, fallback: json.loads(content)
        self.agent.run_prompt.return_value = SimpleNamespace(content='{"intention":"protect pride"}', tool_calls=[])
        self.character = SimpleNamespace(name="Tom", agent=self.agent)
        self.stage = StrategyStage(self.character)
        self.character.pipeline = SimpleNamespace(strategy_stage=self.stage)

    def test_all_ten_fixtures_run_without_any_predecessor_or_context_builder(self):
        prompts = list(fixture_prompts())
        self.assertEqual(len(prompts), 10)
        for prompt in prompts:
            with self.subTest(scenario=prompt.stage_inputs["scenario_id"]):
                output, context = self.harness.execute_strategy_stage(self.character, prompt)
                self.assertEqual(output.immediate_actions, ["keep_talking"])
                self.assertEqual(set(context), {"initial_context", "perception", "retrieved_context", "appraisal", "emotion"})
                generated = self.agent.run_prompt.call_args.kwargs["prompt"]
                self.assertIn(context["initial_context"]["situation"], generated)
                self.assertIn(context["initial_context"]["character_definition"], generated)
                self.assertNotIn(prompt.notes, generated)
        self.assertEqual(self.agent.run_prompt.call_count, 10)

    def test_missing_or_malformed_inputs_fail_before_generation(self):
        original = next(fixture_prompts())
        for key in ("initial_context_payload", "perception_payload", "retrieved_context_payload", "appraisal_payload", "emotion_payload"):
            for invalid in (None, "{}", []):
                with self.subTest(key=key, invalid=invalid):
                    prompt = copy.deepcopy(original)
                    prompt.stage_inputs[key] = invalid
                    with self.assertRaises(ValueError):
                        self.harness.execute_strategy_stage(self.character, prompt)
        self.agent.run_prompt.assert_not_called()

    def test_nested_appraisal_shape_is_rejected_instead_of_silently_defaulted(self):
        prompt = next(fixture_prompts())
        prompt.stage_inputs["appraisal_payload"] = {"appraisal": prompt.stage_inputs["appraisal_payload"]}
        with self.assertRaisesRegex(ValueError, "Unknown appraisal_payload fields"):
            self.harness.execute_strategy_stage(self.character, prompt)
        self.agent.run_prompt.assert_not_called()

    def test_action_tools_are_offered_and_invocations_are_preserved(self):
        actions = ("keep_talking", "end_conversation", "open_trade", "offer_quest", "alert_guards", "attack_player", "flee", "call_for_help")
        for action in actions:
            with self.subTest(action=action):
                self.agent.run_prompt.return_value.tool_calls = [SimpleNamespace(function=SimpleNamespace(name=action))]
                output, _ = self.harness.execute_strategy_stage(self.character, next(fixture_prompts()))
                self.assertEqual(output.immediate_actions, [action])
                offered = {tool.__name__: tool for tool in self.agent.run_prompt.call_args.kwargs["tools"]}
                self.assertEqual(set(offered), set(actions))
                self.assertEqual(offered[action]("fixture reasoning"), {"action": action, "reasoning": "fixture reasoning"})

    def test_multiple_calls_preserved_and_unknown_call_uses_existing_fallback(self):
        for names, expected in [(["call_for_help", "end_conversation"], ["call_for_help", "end_conversation"]), (["unknown_tool"], ["keep_talking"])]:
            self.agent.run_prompt.return_value.tool_calls = [SimpleNamespace(function=SimpleNamespace(name=name)) for name in names]
            output, _ = self.harness.execute_strategy_stage(self.character, next(fixture_prompts()))
            self.assertEqual(output.immediate_actions, expected)

    def test_author_note_is_separate_and_export_uses_existing_results_format(self):
        from test.test_agent_test import FakeCharacter
        character = FakeCharacter()
        character.name = "Tom"
        character.pipeline.strategy_stage = self.stage
        prompt = next(fixture_prompts())
        names = list(self.harness.STAGE_METRIC_GUIDANCE[prompt.target_stage])
        self.assertEqual(len(names), 6)
        captured = []

        def judge(text):
            captured.append(text)
            return json.dumps({"metrics": [{"metric_name": name, "score": 0 if name == "author_note" else 1, "explanation": "Fixture evaluation."} for name in names]})

        self.harness.provider = SimpleNamespace(generate=judge)
        self.harness.export_data = Mock()
        self.harness.evaluate_prompts(character, [prompt])
        self.assertIn(prompt.notes, captured[0])
        exports = self.harness.export_data.call_args_list
        self.assertEqual(len(exports), 2)
        results = exports[1].args[0]
        self.assertEqual(len(results), 6)
        scores = {result.metric_name: result.passed for result in results}
        self.assertTrue(scores["character_fit"])
        self.assertFalse(scores["author_note"])
        self.assertTrue(exports[1].args[2].endswith("tom_stage_testsuite_results.csv"))


if __name__ == "__main__":
    unittest.main()
