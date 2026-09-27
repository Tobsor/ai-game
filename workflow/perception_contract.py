"""Strict perception boundary shared by model responses and stage fixtures."""
import json
from dataclasses import asdict, fields, is_dataclass
from typing import Literal, get_args, get_origin, get_type_hints

from workflow.models import PerceptionCore, PerceptionResult


class PerceptionValidationError(ValueError):
    pass


def validate_perception(payload: object, raw_prompt: str) -> PerceptionResult:
    def validate(kind, value, path):
        if is_dataclass(kind):
            if not isinstance(value, dict):
                expected = ", ".join(f.name for f in fields(kind))
                received = repr(value)
                if len(received) > 160:
                    received = received[:157] + "..."
                raise PerceptionValidationError(
                    f"{path}: expected an object with fields [{expected}]; "
                    f"received {type(value).__name__}: {received}."
                )
            names = {f.name for f in fields(kind)}
            missing, unknown = names - value.keys(), value.keys() - names
            if missing:
                raise PerceptionValidationError(f"{path}: missing fields: {', '.join(sorted(missing))}.")
            if unknown:
                raise PerceptionValidationError(f"{path}: unknown fields: {', '.join(sorted(unknown))}.")
            hints = get_type_hints(kind)
            return kind(**{name: validate(hints[name], value[name], f"{path}.{name}") for name in hints})
        if kind is str:
            if not isinstance(value, str) or not value.strip():
                raise PerceptionValidationError(f"{path}: expected a nonblank string.")
        elif get_origin(kind) is Literal:
            if not isinstance(value, str) or value not in get_args(kind):
                raise PerceptionValidationError(f"{path}: expected one of {get_args(kind)}.")
        elif kind == list[str]:
            if not isinstance(value, list):
                raise PerceptionValidationError(f"{path}: expected a list of nonblank strings.")
            return [validate(str, item, f"{path}[{index}]") for index, item in enumerate(value)]
        return value

    result = validate(PerceptionResult, payload, "perception")
    if result.raw_prompt != raw_prompt:
        raise PerceptionValidationError("perception.raw_prompt: must match the original player input exactly.")
    return result


def parse_perception(content: str, raw_prompt: str) -> PerceptionResult:
    try:
        payload = json.loads(content)
    except (ValueError, TypeError) as error:
        raise PerceptionValidationError(f"perception: expected valid JSON: {error}") from error
    return validate_perception(payload, raw_prompt)


def format_perception_core(core: PerceptionCore) -> str:
    return json.dumps(asdict(core), ensure_ascii=False)
