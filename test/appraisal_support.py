"""Static inputs and error-aware exports for isolated appraisal evaluations."""
import csv
import json
import math
import re
from dataclasses import asdict, fields
from pathlib import Path
from typing import get_type_hints

from models import StageExpectationMode, StageJudgeMetricResult
from workflow.models import InitialContext, PerceptionResult, RetrievedContext


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACE_COLUMNS = ["scenario_id", "initial_context_fixture", "status", "error_type", "error_message"]


class AppraisalFixtureError(ValueError):
    pass


class AppraisalJudgeError(ValueError):
    pass


def fixture_object(value, label):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise AppraisalFixtureError(f"{label} must contain a JSON object: {error.msg}") from error
    if not isinstance(value, dict):
        raise AppraisalFixtureError(f"{label} must be an explicit object (an empty object is allowed).")
    return dict(value)


def construct_fixture(cls, payload, label):
    """Validate fixture types rather than silently repairing authoring mistakes."""
    hints = get_type_hints(cls)
    unknown = set(payload) - {field.name for field in fields(cls)}
    if unknown:
        raise AppraisalFixtureError(f"{label} has unknown fields: {', '.join(sorted(unknown))}")
    for name, value in payload.items():
        kind = hints[name]
        valid = True
        if kind is str:
            valid = isinstance(value, str)
        elif kind is bool:
            valid = isinstance(value, bool)
        elif kind is float:
            valid = type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 1
        elif kind == list[str]:
            valid = isinstance(value, list) and all(isinstance(item, str) for item in value)
        elif name == "tool_calls":
            valid = value == []
        if not valid:
            raise AppraisalFixtureError(f"Invalid {label}.{name}: expected {kind}.")
    try:
        return cls(**payload)
    except TypeError as error:
        raise AppraisalFixtureError(f"Invalid {label}: {error}") from error


def resolve_appraisal_inputs(prompt):
    if prompt.expectation_mode != StageExpectationMode.JUDGE or prompt.deterministic_checks:
        raise AppraisalFixtureError("Appraisal tests require judge mode without deterministic checks.")
    inputs = prompt.stage_inputs
    for name in ("scenario_id", "initial_context_fixture", "perception_payload", "retrieved_context_payload"):
        if name not in inputs or inputs[name] is None:
            raise AppraisalFixtureError(f"Appraisal tests require explicit {name}.")
    for name in ("scenario_id", "initial_context_fixture"):
        if not isinstance(inputs[name], str) or not inputs[name].strip():
            raise AppraisalFixtureError(f"{name} must be a nonblank string.")
    if not prompt.notes.strip():
        raise AppraisalFixtureError("Appraisal tests require author notes.")
    if any(inputs.get(name) for name in ("vector_state_config", "vector_snapshot_path")):
        raise AppraisalFixtureError("Appraisal tests use static retrieval and cannot configure vector state.")

    path = Path(inputs["initial_context_fixture"])
    if not path.is_absolute():
        path = REPO_ROOT / path
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=";")
            expected = {field.name for field in fields(InitialContext)}
            if set(reader.fieldnames or []) != expected:
                raise AppraisalFixtureError("Initial-context CSV must contain exactly the InitialContext columns.")
            rows = list(reader)
        if len(rows) != 1:
            raise AppraisalFixtureError("Initial-context CSV must contain exactly one complete context row.")
        context = rows[0]
        for name in ("active_goals", "belief_state", "recent_turns"):
            context[name] = json.loads(context[name])
        initial = construct_fixture(InitialContext, context, "initial_context")
    except (OSError, ValueError, TypeError) as error:
        raise AppraisalFixtureError(f"Invalid initial-context fixture {path}: {error}") from error
    if initial.character_name != "Tom":
        raise AppraisalFixtureError("Appraisal fixtures must describe Tom.")

    perception = fixture_object(inputs["perception_payload"], "perception_payload")
    if "raw_prompt" in perception and perception["raw_prompt"] != prompt.user_query:
        raise AppraisalFixtureError("perception_payload.raw_prompt must match user_query.")
    perception["raw_prompt"] = prompt.user_query
    perception = construct_fixture(PerceptionResult, perception, "perception_payload")
    retrieval = construct_fixture(
        RetrievedContext, fixture_object(inputs["retrieved_context_payload"], "retrieved_context_payload"),
        "retrieved_context_payload",
    )
    if not retrieval.combined_context.strip() and any(
        getattr(retrieval, name).strip()
        for name in ("memory_context", "relationship_context", "knowledge_context", "social_context")
    ):
        raise AppraisalFixtureError("Appraisal consumes combined_context; source evidence requires an explicit combined_context.")
    return initial, perception, retrieval


def appraisal_execution_context(prompt, resolved):
    return {
        "scenario_id": prompt.stage_inputs["scenario_id"],
        "initial_context_fixture": prompt.stage_inputs["initial_context_fixture"],
        **{name: asdict(value) for name, value in zip(
            ("initial_context", "perception", "retrieved_context"), resolved
        )},
    }


def run_appraisal(character, resolved):
    appraisal, emotion = character.pipeline.appraisal_stage.run(*resolved)
    return {"appraisal": appraisal, "emotion": emotion}


def parse_appraisal_judgment(raw_output, active_metrics):
    """Judge transport/schema errors are not low semantic scores."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_output.strip())
    try:
        payload = json.loads(cleaned)
        metrics = payload["metrics"]
        if not isinstance(metrics, list):
            raise ValueError("metrics must be an array")
        results = {}
        requested = {metric.metric_name for metric in active_metrics}
        for item in metrics:
            if not isinstance(item, dict):
                raise ValueError("metric entries must be objects")
            name = item.get("metric_name")
            if name not in requested or name in results:
                continue
            score = item.get("score")
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"{name} requires a finite score in 0..1")
            explanation = item.get("explanation")
            if not isinstance(explanation, str) or not explanation.strip():
                raise ValueError(f"{name} requires an explanation")
            results[name] = StageJudgeMetricResult(
                metric_name=name, score=score, passed=score >= 0.5, explanation=explanation,
            )
        missing = requested - results.keys()
        if missing:
            raise ValueError(f"Missing metrics: {', '.join(sorted(missing))}")
        return [results[metric.metric_name] for metric in active_metrics]
    except (ValueError, KeyError, TypeError) as error:
        raise AppraisalJudgeError(f"Invalid appraisal judge response: {error}") from error


def evaluate_appraisal_case(runner, character, prompt):
    trace = {name: prompt.stage_inputs.get(name, "") for name in TRACE_COLUMNS[:2]}
    trace.update(status="fixture_error", error_type="", error_message="")
    context = dict(trace)
    output = None
    results = []
    try:
        resolved = resolve_appraisal_inputs(prompt)
        context = appraisal_execution_context(prompt, resolved)
        trace["status"] = "execution_error"
        output = run_appraisal(character, resolved)
        trace["status"] = "judge_error"
        results = runner.evaluate_judge_metrics(
            character, prompt, output, context, runner.serialize_value(output),
        )
        trace["status"] = "evaluated"
    except Exception as error:
        trace.update(error_type=type(error).__name__, error_message=str(error))

    record = {
        "user_query": prompt.user_query, "source_category": prompt.source_category.value,
        "target_stage": prompt.target_stage.value, "expectation_mode": prompt.expectation_mode.value,
        "stage_inputs": runner.serialize_value(prompt.stage_inputs), "notes": prompt.notes,
        "stage_output": runner.serialize_value(output),
        "execution_context": runner.serialize_value(context), **trace,
    }
    if trace["status"] == "evaluated":
        exported = [{**result.model_dump(mode="json"), **trace} for result in results]
    else:
        exported = [{
            key: record[key] for key in ("user_query", "source_category", "target_stage", "expectation_mode", "notes", "stage_output")
        } | {
            "metric_name": "", "score": None, "passed": None,
            "explanation": trace["error_message"], "input_context": runner.serialize_value(context), **trace,
        }]
    return record, exported
