"""Validate self-contained Response fixtures without invoking any earlier stage."""
import math
from dataclasses import fields, is_dataclass

from pydantic import TypeAdapter

from workflow.models import (
    AppraisalResult, EmotionResult, InitialContext, PerceptionResult,
    RetrievedContext, StrategyResult,
)


FIXTURE_TYPES = {
    "initial_context": InitialContext,
    "perception": PerceptionResult,
    "retrieved_context": RetrievedContext,
    "appraisal": AppraisalResult,
    "emotion": EmotionResult,
    "strategy": StrategyResult,
}
# Only runtime metadata and fields unused by Response may use dataclass defaults.
OPTIONAL_FIELDS = {
    PerceptionResult: {"stage_prompt", "tool_calls", "retrieval_reasoning"},
    RetrievedContext: {field.name for field in fields(RetrievedContext)},
    AppraisalResult: {"stage_prompt"},
    EmotionResult: {"stage_prompt"},
    StrategyResult: {"new_sentiment", "sentiment_reasoning"},
}


def construct_fixture(kind, payload, label):
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an explicit JSON object.")
    schema = {field.name: field.type for field in fields(kind)}
    unknown = set(payload) - set(schema)
    missing = set(schema) - OPTIONAL_FIELDS.get(kind, set()) - set(payload)
    if unknown:
        raise ValueError(f"Unknown {label} fields: {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"Missing {label} fields: {', '.join(sorted(missing))}")
    values = {}
    for name, value in payload.items():
        field_type = schema[name]
        field_label = f"{label}.{name}"
        if is_dataclass(field_type):
            values[name] = construct_fixture(field_type, value, field_label)
        else:
            try:
                values[name] = TypeAdapter(field_type).validate_python(value, strict=True)
            except ValueError as error:
                raise ValueError(f"Invalid {field_label}: {error}") from error
            if field_type is float and (isinstance(value, bool) or not math.isfinite(value)):
                raise ValueError(f"{field_label} must be a finite number.")
    return kind(**values)


def resolve_response_inputs(prompt, character_name):
    """Fail on missing context, stale keys, or coercible-but-invalid field types."""
    inputs = prompt.stage_inputs
    scenario = inputs.get("scenario_id")
    if not isinstance(scenario, str) or not scenario.strip():
        raise ValueError("Response tests require a nonblank scenario_id.")
    try:
        allowed = {"scenario_id", *(name + "_payload" for name in FIXTURE_TYPES)}
        unknown = set(inputs) - allowed
        if unknown:
            raise ValueError(f"Unknown Response stage_inputs: {', '.join(sorted(unknown))}")
        if not prompt.user_query.strip():
            raise ValueError("Response tests require a nonblank user_query.")
        resolved = {}
        for name, kind in FIXTURE_TYPES.items():
            key = name + "_payload"
            payload = inputs.get(key)
            if name == "perception" and isinstance(payload, dict):
                payload = dict(payload)
                if "raw_prompt" in payload and payload["raw_prompt"] != prompt.user_query:
                    raise ValueError("perception_payload.raw_prompt must match user_query.")
                payload["raw_prompt"] = prompt.user_query
            resolved[name] = construct_fixture(kind, payload, key)
        initial = resolved["initial_context"]
        if initial.character_name != character_name:
            raise ValueError("Response fixture character_name must match the selected character.")
        required_text = {
            "initial_context": ("character_definition", "situation"),
            "perception": ("summary", "request_type"),
            "appraisal": ("summary",),
            "emotion": ("primary",),
            "strategy": ("intention", "conversation_goal", "risk_level", "disclosure_level",
                         "social_strategy", "tone", "verbosity", "conversation_move"),
        }
        for name, names in required_text.items():
            for field_name in names:
                if not getattr(resolved[name], field_name).strip():
                    raise ValueError(f"{name}_payload.{field_name} must not be blank.")
        retrieval = resolved["retrieved_context"]
        if not retrieval.combined_context.strip() and any(
            getattr(retrieval, name).strip() for name in OPTIONAL_FIELDS[RetrievedContext]
            if name != "combined_context"
        ):
            raise ValueError("Response consumes combined_context; source evidence requires an explicit combined_context.")
        return resolved
    except ValueError as error:
        raise ValueError(f"Response scenario {scenario}: {error}") from error
