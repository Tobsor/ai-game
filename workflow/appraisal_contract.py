"""Strict semantic Appraisal validation shared by model outputs and fixtures."""
from workflow.models import AppraisalResult, EmotionResult


def validate_appraisal(payload, raw_prompt=None):
    if not isinstance(payload, dict) or set(payload) != {"raw_prompt", "appraisal", "emotion"}:
        raise ValueError("Appraisal requires raw_prompt, appraisal and emotion after retrieval extraction")
    if not isinstance(payload["raw_prompt"], str) or (raw_prompt is not None and payload["raw_prompt"] != raw_prompt):
        raise ValueError("Appraisal raw_prompt must match original player input")
    appraisal, emotion = payload["appraisal"], payload["emotion"]
    if not isinstance(appraisal, dict) or set(appraisal) != {"summary", "aspects"}:
        raise ValueError("Appraisal requires summary and aspects")
    if not isinstance(emotion, dict) or set(emotion) != {"components", "summary"}:
        raise ValueError("Emotion requires components and summary")
    def text(value):
        return isinstance(value, str) and bool(value.strip())
    if not text(appraisal["summary"]) or not text(emotion["summary"]):
        raise ValueError("Appraisal and emotion summaries must be nonblank strings")
    if not isinstance(appraisal["aspects"], list) or any(
        not isinstance(item, dict) or set(item) != {"aspect", "evaluation"}
        or not all(text(value) for value in item.values()) for item in appraisal["aspects"]
    ):
        raise ValueError("Aspects require nonblank aspect and evaluation strings")
    if not isinstance(emotion["components"], list) or not all(text(item) for item in emotion["components"]):
        raise ValueError("Emotion components must be nonblank strings")
    return AppraisalResult(**appraisal), EmotionResult(**emotion)
