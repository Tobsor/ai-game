from typing import Dict, Any, Callable
import re
import json

from ai import AISettings, ChatCompletionResult, create_chat_provider, get_ai_settings
from logger import get_logger

logger = get_logger(__name__)

class NPCAgent:
    def __init__(self, settings: AISettings | None = None):
        settings = settings or get_ai_settings()
        self.provider = create_chat_provider(settings.decision_llm)

    def parse_output(self, raw_output: str, fallback: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Attempt to parse JSON from the model output.
        Falls back to the provided defaults on error.
        """
        cleaned = raw_output.strip()
        cleaned = re.sub(r"^```(?:json)?", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned).strip()

        try:
            data = json.loads(cleaned)
            return data if isinstance(data, dict) else (fallback or {})
        except json.JSONDecodeError:
            return fallback or {}

    def run_prompt(
        self,
        prompt: str,
        stage_name: str = "PerceptionStage",
        tools: list[Callable] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> ChatCompletionResult:
        system_message = {"role": "user", "content": prompt}

        res = self.provider.chat(
            messages=[system_message],
            tools=tools
        )

        logger.debug("Generated payload: %s", res)
        logger.conversation_event(
            stage_name=stage_name,
            event="decision_model",
            payload=payload or {"prompt": prompt},
            ai_request={
                "messages": [system_message],
                "tools": [tool.__name__ for tool in tools] if tools is not None else [],
            },
            ai_response={
                "content": res.content,
                "tool_calls": res.tool_calls,
            },
            result={"content": res.content, "tool_calls": res.tool_calls},
        )

        return res
