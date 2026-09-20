import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from models import StageName, StageTestPrompt
from test.AgentTest import AgentTest
from test.TestRunner import get_stage_options, read_stage_prompts
from test.test_agent_test import FakeCharacter
from workflow.stages.retrieval_stage import RetrievalStage


class RetrievalStageTestTests(unittest.TestCase):
    METRICS = ["fetch_relevance", "context_relevance", "summary_quality",
               "knowledge_scope_alignment", "author_note"]

    def setUp(self):
        self.runner = AgentTest.__new__(AgentTest)
        self.runner.provider = Mock()
        self.character = FakeCharacter()
        self.character.pipeline.perception_stage = Mock()
        self.character.pipeline.gap_analysis_stage = Mock()
        self.character.pipeline.retrieval_stage = RetrievalStage(self.character)
        self.character.db.query_text = Mock(return_value="The market closes at sunset.")
        self.character.agent = SimpleNamespace(
            run_prompt=Mock(return_value=SimpleNamespace(content="Market closes at sunset.")))

    def prompt(self, inputs=None, notes="EXPECTED OUTCOME: mention the closing time."):
        return StageTestPrompt(
            user_query="When does the market close?", source_category="general",
            target_stage=StageName.RETRIEVAL_RUN, expectation_mode="judge", notes=notes,
            stage_inputs=inputs if inputs is not None else {
                "perception_payload": {"player_intent": "seek_information", "topic": "market"},
                "gap_analysis_payload": {"tool_calls": [
                    {"tool_name": "recall_knowledge", "reasoning": "Need market hours"}]},
            })

    def test_missing_inputs_fail_before_any_stage_runs(self):
        cases = [({}, ["perception_payload", "gap_analysis_payload"]),
                 ({"perception_payload": {}}, ["gap_analysis_payload"]),
                 ({"gap_analysis_payload": {"tool_calls": []}}, ["perception_payload"]),
                 ({"perception_prompt": "invalid JSON"}, ["perception_payload", "gap_analysis_payload"])]
        for inputs, missing in cases:
            with self.subTest(inputs=inputs):
                with self.assertRaises(ValueError) as error:
                    self.runner.execute_stage(self.character, self.prompt(inputs))
                for name in missing:
                    self.assertIn(name, str(error.exception))
        self.character.pipeline.perception_stage.run.assert_not_called()
        self.character.pipeline.gap_analysis_stage.run.assert_not_called()
        self.character.db.query_text.assert_not_called()
        self.character.agent.run_prompt.assert_not_called()

    def test_aliases_and_empty_calls_skip_predecessors_and_summary_model(self):
        for encode in (lambda value: value, json.dumps):
            with self.subTest(encode=encode):
                prompt = self.prompt({
                    "perception_prompt": encode({"topic": "market"}),
                    "gap_analysis_prompt": encode({"tool_calls": []}),
                })
                output, context = self.runner.execute_stage(self.character, prompt)
                self.assertEqual(output.combined_context, "no information")
                self.assertEqual(context["perception"]["topic"], "market")
                self.assertEqual(context["gap_analysis_tool_names"], [])
        self.character.pipeline.perception_stage.run.assert_not_called()
        self.character.pipeline.gap_analysis_stage.run.assert_not_called()
        self.character.db.query_text.assert_not_called()
        self.character.agent.run_prompt.assert_not_called()

    def test_retrieval_summary_judge_and_csv_flow(self):
        prompt = self.prompt()
        self.runner.provider.generate.return_value = json.dumps({"metrics": [
            {"metric_name": name, "score": 1, "passed": True, "explanation": "Matches expectation"}
            for name in self.METRICS]})
        with tempfile.TemporaryDirectory() as directory:
            with patch("test.AgentTest.script_dir", directory):
                self.runner.evaluate_prompts(self.character, [prompt])
            with (Path(directory) / "results/mira_stage_testsuite_results.csv").open(
                    encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source, delimiter=";"))
            self.assertEqual([row["metric_name"] for row in rows], self.METRICS)
            self.assertTrue(all(row["notes"] == prompt.notes for row in rows))
            output = json.loads(rows[0]["stage_output"])
            self.assertEqual(output["knowledge_context"], "The market closes at sunset.")
            self.assertEqual(output["combined_context"], "Market closes at sunset.")

        query = self.character.db.query_text.call_args.kwargs["prompt"]
        summary = self.character.agent.run_prompt.call_args.kwargs
        self.assertIn("Need market hours", query)
        self.assertIn(prompt.user_query, query)
        self.assertIn("The market closes at sunset.", summary["prompt"])
        self.assertEqual(summary["stage_name"], "RetrievalStage.summarize")
        self.assertNotIn(prompt.notes, query)
        self.assertNotIn(prompt.notes, summary["prompt"])
        judge = self.runner.provider.generate.call_args.args[0]
        self.assertIn(prompt.notes, judge)
        self.assertIn('"knowledge_context": "The market closes at sunset."', judge)
        self.assertIn('"combined_context": "Market closes at sunset."', judge)
        self.assertEqual(list(self.runner.STAGE_METRIC_GUIDANCE[StageName.RETRIEVAL_RUN]), self.METRICS)
        self.character.pipeline.perception_stage.run.assert_not_called()
        self.character.pipeline.gap_analysis_stage.run.assert_not_called()

    def test_blank_notes_and_missing_judge_metrics(self):
        self.runner.provider.generate.return_value = '{"metrics":[]}'
        results = self.runner.evaluate_judge_metrics(self.character, self.prompt(notes=" "), {}, {}, "{}")
        self.assertEqual([result.metric_name for result in results], self.METRICS)
        self.assertTrue(all(result.score == 0 and not result.passed for result in results))
        judge = self.runner.provider.generate.call_args.args[0]
        self.assertIn("No author note provided.", judge)
        self.assertIn("If the note is missing or blank, return score 0.0, passed=false", judge)

    def test_fixture_migration_and_retired_target(self):
        prompts = read_stage_prompts(SimpleNamespace(name="tom"))
        self.assertNotIn("RetrievalStage.summarize", get_stage_options(prompts))
        with self.assertRaises(ValueError):
            StageName("RetrievalStage.summarize")
        prompt = next(p for p in prompts if any(
            check.metric_name == "no_information_passthrough" for check in p.deterministic_checks))
        self.assertEqual(prompt.target_stage, StageName.RETRIEVAL_RUN)
        output, context = self.runner.execute_stage(self.character, prompt)
        results = self.runner.evaluate_deterministic_checks(prompt, output, context, self.runner.serialize_value(output))
        self.assertTrue(all(result.passed for result in results))
        self.character.agent.run_prompt.assert_not_called()


if __name__ == "__main__":
    unittest.main()
