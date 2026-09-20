import csv
import unittest
from pathlib import Path
from types import SimpleNamespace

from ai import NormalizedToolCall, NormalizedToolFunction
from models import StageName
from test.AgentTest import AgentTest
from test.TestRunner import read_stage_prompts
from test.test_agent_test import FakeCharacter
from workflow.stages.gap_analysis_stage import GapAnalysisStage


class GapAnalysisTests(unittest.TestCase):
    def test_context_and_notes_are_separated_and_tools_are_normalized(self):
        character = FakeCharacter()
        prompts = read_stage_prompts(SimpleNamespace(name="tom"))
        prompt = next(p for p in prompts if p.target_stage == StageName.GAP_ANALYSIS)
        prompt.stage_inputs = {"perception_payload": {"topic": "market"}, "available_context": "Market closes at sunset."}
        prompt.notes = "AUTHOR EXPECTATION ONLY"
        captured = []
        calls = [NormalizedToolCall(function=NormalizedToolFunction(name=name, arguments={"reasoning": "Need missing facts"}))
                 for name in ["recall_memory", "recall_knowledge", "recall_memory"]]
        def run_prompt(**kwargs):
            captured.append(kwargs["prompt"])
            return SimpleNamespace(tool_calls=calls)
        character.agent = SimpleNamespace(run_prompt=run_prompt)
        character.pipeline.gap_analysis_stage = GapAnalysisStage(character)
        runner = AgentTest()
        output, context = runner.execute_gap_analysis_stage(character, prompt)
        self.assertIn("Market closes at sunset.", captured[0])
        self.assertNotIn(prompt.notes, captured[0])
        self.assertEqual(context["gap_analysis_tool_names"], ["recall_knowledge", "recall_memory"])
        self.assertEqual(context["gap_analysis_duplicate_count"], 1)
        self.assertTrue(context["gap_analysis_arguments_valid"])
        calls[0].function.arguments["reasoning"] = " "
        self.assertFalse(runner.execute_gap_analysis_stage(character, prompt)[1]["gap_analysis_arguments_valid"])
        judge_prompts = []
        runner.provider = SimpleNamespace(generate=lambda text: judge_prompts.append(text) or '{"metrics":[]}')
        results = runner.evaluate_judge_metrics(character, prompt, output, context, "{}")
        self.assertIn(prompt.notes, judge_prompts[0])
        self.assertIn("Market closes at sunset.", judge_prompts[0])
        self.assertEqual(len(results), 7)
        self.assertTrue(all(not result.passed for result in results))
        prompt.stage_inputs = {}
        with self.assertRaisesRegex(ValueError, "explicit perception_payload"):
            runner.execute_gap_analysis_stage(character, prompt)

    def test_csv_gap_fixtures(self):
        path = Path("data/test_data/tom_stage_testsuite.csv")
        with path.open(encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source, delimiter=";"):
                self.assertNotIn(None, row)
                self.assertNotIn(None, row.values())
        prompts = read_stage_prompts(SimpleNamespace(name="tom"))
        gaps = [p for p in prompts if p.target_stage == StageName.GAP_ANALYSIS]
        self.assertEqual(len(gaps), 17)
        for prompt in gaps:
            self.assertTrue(prompt.notes.strip())
            payload = prompt.stage_inputs["perception_payload"]
            self.assertNotIn("npc_perception", payload)
            self.assertIsInstance(payload["topic"], str)
            self.assertIsInstance(prompt.stage_inputs["available_context"], str)


if __name__ == "__main__":
    unittest.main()
