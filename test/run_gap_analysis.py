"""Run CSV gap fixtures three times: python -m test.run_gap_analysis."""
import json
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

from classes.NpcAgent import NPCAgent
from models import StageExpectationMode, StageName
from test.AgentTest import AgentTest
from test.TestRunner import read_characters, read_stage_prompts
from workflow.stages.gap_analysis_stage import GapAnalysisStage


def main():
    # A lightweight character avoids initializing Chroma or running predecessor stages.
    data = next(row for row in read_characters(Path("data/character_data_cop.csv")) if row["name"].lower() == "tom")
    character = SimpleNamespace(**{key: data.get(key, "") for key in
                                  ("name", "pl_list", "ali_chat", "knowledge", "past", "relations", "sentiment")})
    character.agent = NPCAgent()
    character.pipeline = SimpleNamespace(gap_analysis_stage=GapAnalysisStage(character))
    runner = AgentTest()
    prompts = [p for p in read_stage_prompts(character) if p.target_stage == StageName.GAP_ANALYSIS]
    report = []
    path = Path("test/results/tom_gap_analysis_repeated.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    for index, prompt in enumerate(prompts, 1):
        case = {"case": index, "query": prompt.user_query, "mode": prompt.expectation_mode.value,
                "notes": prompt.notes, "stage_inputs": prompt.stage_inputs, "runs": []}
        report.append(case)
        for repetition in range(1, 4):
            try:
                output, context = runner.execute_gap_analysis_stage(character, prompt)
                kwargs = dict(prompt=prompt, stage_output=output, execution_context=context,
                              stage_output_json=runner.serialize_value(output))
                results = (runner.evaluate_deterministic_checks(**kwargs)
                           if prompt.expectation_mode == StageExpectationMode.DETERMINISTIC
                           else runner.evaluate_judge_metrics(character=character, **kwargs))
                case["runs"].append({"run": repetition, "passed": all(r.passed for r in results),
                                     "results": [r.model_dump(mode="json") for r in results]})
            except Exception as error:
                case["runs"].append({"run": repetition, "passed": False, "error_type": type(error).__name__})
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        case["pass_rate"] = mean(run["passed"] for run in case["runs"])
        names = {r["metric_name"] for run in case["runs"] for r in run.get("results", [])}
        case["mean_metric_scores"] = {name: mean(r["score"] for run in case["runs"]
            for r in run.get("results", []) if r["metric_name"] == name) for name in sorted(names)}
        case["acceptance_passed"] = all(run["passed"] for run in case["runs"])
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Case {index}/{len(prompts)}: {case['pass_rate']:.0%} passed", flush=True)
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
