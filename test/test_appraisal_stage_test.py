import csv
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from models import StageJudgeMetric, StageName
from test.AgentTest import AgentTest
from test.TestRunner import read_stage_prompts
from test.appraisal_support import (
    AppraisalFixtureError, AppraisalJudgeError, parse_appraisal_judgment, resolve_appraisal_inputs,
)
from test.run_appraisal import main as run_appraisal
from workflow.models import GapAnalysisResult
from workflow.stages.appraisal_stage import AppraisalStage


class AppraisalStageTests(unittest.TestCase):
    METRICS = ["character_circumstance_fit", "emotional_coherence", "goals_beliefs_alignment",
               "grounding_attribution", "downstream_usefulness", "scenario_expectation_alignment"]

    def setUp(self):
        self.runner = AgentTest.__new__(AgentTest)
        self.runner.provider = Mock()
        self.runner.provider.generate.return_value = self.judgment()
        self.cases = [p for p in read_stage_prompts(SimpleNamespace(name="tom"))
                      if p.target_stage == StageName.APPRAISAL]
        self.character = SimpleNamespace(
            name="Tom", pl_list="LIVE DEFINITION", knowledge="LIVE KNOWLEDGE", past="LIVE PAST",
            relations="LIVE RELATIONS", sentiment="LIVE SENTIMENT", ali_chat="LIVE DIALOGUE",
            build_initial_context=Mock(side_effect=AssertionError("Live context forbidden")),
            db=Mock(), agent=Mock(), pipeline=SimpleNamespace(),
        )
        for name in ("initial_context", "perception", "gap_analysis", "retrieval", "strategy", "response"):
            setattr(self.character.pipeline, name + "_stage", Mock())
        self.character.pipeline.appraisal_stage = AppraisalStage(self.character)
        self.character.agent.run_prompt.return_value = SimpleNamespace(content="ignored")
        self.character.agent.parse_output.return_value = {
            "appraisal": {"summary": "Recognition affirms my hard-earned fishing skill.", "relevance": 0.7},
            "emotion": {"primary": "pride", "intensity": 0.6},
        }

    def judgment(self, score=0.8):
        return json.dumps({"metrics": [{"metric_name": name, "score": score, "passed": False,
                                        "explanation": "The supplied context supports this appraisal."}
                                       for name in self.METRICS]})

    def prompt(self):
        return self.cases[0].model_copy(deep=True)

    def assert_isolated(self):
        self.character.build_initial_context.assert_not_called()
        self.assertEqual(self.character.db.mock_calls, [])
        for name in ("initial_context", "perception", "gap_analysis", "retrieval", "strategy", "response"):
            getattr(self.character.pipeline, name + "_stage").run.assert_not_called()

    def test_all_ten_fixtures_load_with_unique_ids(self):
        self.assertEqual(len(self.cases), 10)
        self.assertEqual(len({p.stage_inputs["scenario_id"] for p in self.cases}), 10)
        for prompt in self.cases:
            initial, perception, retrieval = resolve_appraisal_inputs(prompt)
            self.assertEqual(initial.character_name, "Tom")
            self.assertTrue(initial.active_goals)
            self.assertEqual(perception.raw_prompt, prompt.user_query)
            self.assertTrue(prompt.notes)
        # Repeated prompts are independent fixtures with genuinely different consumed context.
        repeated = [p for p in self.cases if p.user_query == self.cases[5].user_query]
        self.assertEqual(len(repeated), 2)
        first, second = [resolve_appraisal_inputs(p)[0] for p in repeated]
        self.assertNotEqual(first.relationship_summary, second.relationship_summary)
        self.assertNotIn("he does not know {{user}}", first.character_definition)

    def test_resolved_inputs_are_exactly_executed_and_exported(self):
        prompt = self.prompt()
        output, context = self.runner.execute_stage(self.character, prompt)
        resolved = resolve_appraisal_inputs(prompt)
        self.assertEqual(context["initial_context"], asdict(resolved[0]))
        self.assertEqual(context["perception"], asdict(resolved[1]))
        self.assertEqual(context["retrieved_context"], asdict(resolved[2]))
        stage_prompt = self.character.agent.run_prompt.call_args.kwargs["prompt"]
        self.assertIn(resolved[2].combined_context, stage_prompt)
        self.assertNotIn(prompt.notes, stage_prompt)
        self.assertNotIn("LIVE DEFINITION", stage_prompt)
        self.assertIn("appraisal", output)
        self.assert_isolated()

    def test_each_missing_or_null_fixture_fails_before_any_model(self):
        for key in ("scenario_id", "initial_context_fixture", "perception_payload", "retrieved_context_payload"):
            for null in (False, True):
                prompt = self.prompt()
                if null:
                    prompt.stage_inputs[key] = None
                else:
                    del prompt.stage_inputs[key]
                with self.subTest(key=key, null=null), self.assertRaisesRegex(AppraisalFixtureError, key):
                    self.runner.execute_stage(self.character, prompt)
        self.character.agent.run_prompt.assert_not_called()
        self.assert_isolated()

    def test_explicit_empty_retrieval_is_valid_for_object_and_json(self):
        for value in ({}, "{}", {"combined_context": ""}):
            prompt = self.prompt()
            prompt.stage_inputs["retrieved_context_payload"] = value
            _, context = self.runner.execute_stage(self.character, prompt)
            self.assertTrue(all(value == "" for value in context["retrieved_context"].values()))
        self.assert_isolated()

    def test_malformed_payloads_fail_instead_of_defaulting(self):
        for key, value in [
            ("perception_payload", "broken JSON"), ("perception_payload", []),
            ("perception_payload", {"confidence": "high"}),
            ("perception_payload", {"perceived_intent": "greet"}),
            ("perception_payload", {"raw_prompt": "different query"}),
            ("retrieved_context_payload", ""), ("retrieved_context_payload", {"combined_context": 42}),
            ("retrieved_context_payload", {"memory_context": "Hidden evidence"}),
            ("retrieved_context_payload", {"typo": "fact"}),
        ]:
            prompt = self.prompt()
            prompt.stage_inputs[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(AppraisalFixtureError):
                self.runner.execute_stage(self.character, prompt)
        self.character.agent.run_prompt.assert_not_called()
        self.assert_isolated()

    def test_bad_context_file_and_list_cells_are_fixture_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "context.csv"
            prompt = self.prompt()
            context = asdict(resolve_appraisal_inputs(prompt)[0])
            prompt.stage_inputs["initial_context_fixture"] = str(path)
            with self.assertRaisesRegex(AppraisalFixtureError, "initial-context fixture"):
                resolve_appraisal_inputs(prompt)
            context["active_goals"] = "not JSON"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(context), delimiter=";")
                writer.writeheader()
                writer.writerow(context)
            with self.assertRaisesRegex(AppraisalFixtureError, "initial-context fixture"):
                resolve_appraisal_inputs(prompt)

    def test_deterministic_mode_and_vector_setup_are_rejected(self):
        prompt = self.prompt()
        prompt.expectation_mode = "deterministic"
        with self.assertRaisesRegex(AppraisalFixtureError, "judge mode"):
            resolve_appraisal_inputs(prompt)
        prompt = self.prompt()
        prompt.stage_inputs["vector_state_config"] = "unwanted.json"
        with self.assertRaisesRegex(AppraisalFixtureError, "static retrieval"):
            resolve_appraisal_inputs(prompt)

    def test_judge_uses_resolved_context_and_notes_without_live_background(self):
        prompt = self.prompt()
        output, context = self.runner.execute_stage(self.character, prompt)
        results = self.runner.evaluate_judge_metrics(self.character, prompt, output, context, "{}")
        judge_prompt = self.runner.provider.generate.call_args.args[0]
        self.assertIn(prompt.notes, judge_prompt)
        self.assertIn(self.runner.serialize_value(context), judge_prompt)
        self.assertNotIn("LIVE ", judge_prompt)
        self.assertIn("does not independently verify raw output validity", judge_prompt)
        self.assertEqual([result.metric_name for result in results], self.METRICS)
        self.assertTrue(all(result.passed for result in results))
        self.assertEqual(json.loads(results[0].input_context), context)

    def test_threshold_ignores_inconsistent_judge_pass_flags(self):
        metrics = [StageJudgeMetric(metric_name=name) for name in self.METRICS]
        for score, expected in ((0.49, False), (0.5, True), (1.0, True)):
            results = parse_appraisal_judgment(self.judgment(score), metrics)
            self.assertTrue(all(result.passed == expected for result in results))

    def test_invalid_judge_results_are_errors_not_semantic_zeroes(self):
        metrics = [StageJudgeMetric(metric_name=name) for name in self.METRICS]
        bad = ["not JSON", '{"metrics":[]}', self.judgment(float("nan")), self.judgment(2)]
        missing = json.loads(self.judgment())
        missing["metrics"][0].pop("explanation")
        bad.append(json.dumps(missing))
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(AppraisalJudgeError):
                parse_appraisal_judgment(raw, metrics)

    def test_csv_traceability_and_errors_continue_to_next_scenario(self):
        broken = self.prompt()
        broken.stage_inputs["scenario_id"] = "broken_fixture"
        del broken.stage_inputs["retrieved_context_payload"]
        execution = self.prompt()
        execution.stage_inputs["scenario_id"] = "broken_execution"
        judge = self.prompt()
        judge.stage_inputs["scenario_id"] = "broken_judge"
        valid = self.prompt()
        valid.stage_inputs["scenario_id"] = "semantic_failure"
        self.character.agent.run_prompt.side_effect = [RuntimeError("stage unavailable"), SimpleNamespace(content="ok"), SimpleNamespace(content="ok")]
        self.runner.provider.generate.side_effect = ["invalid", self.judgment(0.3)]
        with tempfile.TemporaryDirectory() as directory, patch("test.AgentTest.script_dir", directory):
            self.runner.evaluate_prompts(self.character, [broken, execution, judge, valid])
            with (Path(directory) / "results/tom_stage_testsuite_prompts.csv").open(encoding="utf-8", newline="") as handle:
                prompts = list(csv.DictReader(handle, delimiter=";"))
            with (Path(directory) / "results/tom_stage_testsuite_results.csv").open(encoding="utf-8", newline="") as handle:
                results = list(csv.DictReader(handle, delimiter=";"))
        self.assertEqual([p["status"] for p in prompts], ["fixture_error", "execution_error", "judge_error", "evaluated"])
        self.assertEqual(len(results), 9)
        for result in results[:3]:
            self.assertEqual(result["score"], "")
            self.assertEqual(result["passed"], "")
            self.assertEqual(result["metric_name"], "")
            self.assertTrue(result["error_type"])
        self.assertEqual({r["scenario_id"] for r in results[3:]}, {"semantic_failure"})
        self.assertTrue(all(r["passed"] == "False" for r in results[3:]))
        self.assertTrue(all(r["initial_context_fixture"] == valid.stage_inputs["initial_context_fixture"] for r in results))
        self.assertIn("initial_context", json.loads(prompts[1]["execution_context"]))
        self.assertIsNotNone(json.loads(prompts[2]["stage_output"]))
        self.assert_isolated()

    def test_mixed_stage_export_keeps_non_appraisal_behavior(self):
        other = next(p for p in read_stage_prompts(SimpleNamespace(name="tom")) if p.target_stage == StageName.GAP_ANALYSIS)
        self.character.pipeline.gap_analysis_stage.run.return_value = GapAnalysisResult(tool_calls=[])
        with tempfile.TemporaryDirectory() as directory, patch("test.AgentTest.script_dir", directory):
            self.runner.evaluate_prompts(self.character, [other, self.prompt()])
            with (Path(directory) / "results/tom_stage_testsuite_results.csv").open(encoding="utf-8", newline="") as handle:
                results = list(csv.DictReader(handle, delimiter=";"))
        gap = [r for r in results if r["target_stage"] == "GapAnalysisStage"]
        self.assertTrue(gap)
        self.assertTrue(all(r["scenario_id"] == "" for r in gap))

    def test_runner_constructs_only_appraisal_and_uses_existing_export(self):
        with patch("test.run_appraisal.NPCAgent") as agent, patch("test.run_appraisal.AgentTest") as runner:
            run_appraisal()
        character, prompts = runner.return_value.evaluate_prompts.call_args.args
        self.assertFalse(hasattr(character, "db"))
        self.assertEqual(set(vars(character.pipeline)), {"appraisal_stage"})
        self.assertEqual(len(prompts), 10)
        self.assertIs(character.agent, agent.return_value)


if __name__ == "__main__":
    unittest.main()
