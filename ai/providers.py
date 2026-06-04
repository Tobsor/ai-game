import json
import os
from dataclasses import dataclass
import inspect
from typing import Any, Protocol
from urllib import request

import ollama

from ai.settings import RoleProviderConfig


@dataclass
class NormalizedToolFunction:
    name: str
    arguments: dict[str, Any]


@dataclass
class NormalizedToolCall:
    function: NormalizedToolFunction


@dataclass
class ChatCompletionResult:
    content: str
    tool_calls: list[NormalizedToolCall]


class ChatProvider(Protocol):
    def chat(self, messages: list[dict[str, Any]], tools: list[Any] | None = None) -> ChatCompletionResult:
        ...


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[list[float]]:
        ...


class TextGenerationProvider(Protocol):
    def generate(self, prompt: str) -> str:
        ...


def _normalize_tool_calls(raw_tool_calls: Any) -> list[NormalizedToolCall]:
    if raw_tool_calls is None:
        return []

    normalized_calls: list[NormalizedToolCall] = []
    for raw_tool_call in raw_tool_calls:
        function = getattr(raw_tool_call, "function", None)
        if function is not None:
            name = getattr(function, "name", "")
            arguments = getattr(function, "arguments", {})
        else:
            function_payload = raw_tool_call.get("function", {})
            name = function_payload.get("name", "")
            arguments = function_payload.get("arguments", {})

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

        normalized_calls.append(
            NormalizedToolCall(
                function=NormalizedToolFunction(
                    name=str(name),
                    arguments=arguments if isinstance(arguments, dict) else {},
                )
            )
        )

    return normalized_calls


class OllamaChatProvider:
    def __init__(self, config: RoleProviderConfig):
        self.config = config

    def chat(self, messages: list[dict[str, Any]], tools: list[Any] | None = None) -> ChatCompletionResult:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
        }
        if tools is not None:
            kwargs["tools"] = tools

        result = ollama.chat(**kwargs)
        message = result["message"] if isinstance(result, dict) else result.message
        content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
        tool_calls = message.get("tool_calls") if isinstance(message, dict) else getattr(message, "tool_calls", None)

        return ChatCompletionResult(
            content=content,
            tool_calls=_normalize_tool_calls(tool_calls),
        )


class OllamaEmbeddingProvider:
    def __init__(self, config: RoleProviderConfig):
        self.config = config

    def embed(self, text: str) -> list[list[float]]:
        response = ollama.embed(model=self.config.model, input=text)
        embeddings = response.embeddings if hasattr(response, "embeddings") else response["embeddings"]
        return embeddings


class OllamaTextGenerationProvider:
    def __init__(self, config: RoleProviderConfig):
        self.config = config

    def generate(self, prompt: str) -> str:
        return ollama.generate(model=self.config.model, prompt=prompt)["response"]


class HostedChatProvider:
    def __init__(self, config: RoleProviderConfig):
        self.config = config

    def chat(self, messages: list[dict[str, Any]], tools: list[Any] | None = None) -> ChatCompletionResult:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
        }
        if tools is not None:
            payload["tools"] = _normalize_tool_definitions(tools)

        response = _post_json(
            url=self.config.base_url,
            payload=payload,
            config=self.config,
        )
        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})

        return ChatCompletionResult(
            content=message.get("content", "") or "",
            tool_calls=_normalize_tool_calls(message.get("tool_calls")),
        )


class HostedEmbeddingProvider:
    def __init__(self, config: RoleProviderConfig):
        self.config = config

    def embed(self, text: str) -> list[list[float]]:
        response = _post_json(
            url=self.config.base_url,
            payload={
                "model": self.config.model,
                "input": text,
            },
            config=self.config,
        )

        if "data" in response:
            return [item.get("embedding", []) for item in response["data"]]

        if "embeddings" in response:
            return response["embeddings"]

        return []


class HostedTextGenerationProvider:
    def __init__(self, config: RoleProviderConfig):
        self.config = config
        self.chat_provider = HostedChatProvider(config)

    def generate(self, prompt: str) -> str:
        result = self.chat_provider.chat(messages=[{"role": "user", "content": prompt}])
        return result.content


def _annotation_to_json_type(annotation: Any) -> str:
    if annotation in {int, float}:
        return "number"
    if annotation is bool:
        return "boolean"
    if annotation in {list, tuple, set}:
        return "array"
    if annotation is dict:
        return "object"
    return "string"


def _normalize_tool_definitions(tools: list[Any]) -> list[dict[str, Any]]:
    normalized_tools: list[dict[str, Any]] = []

    for tool in tools:
        if not callable(tool):
            continue

        signature = inspect.signature(tool)
        properties: dict[str, Any] = {}
        required: list[str] = []

        for name, parameter in signature.parameters.items():
            annotation = parameter.annotation if parameter.annotation is not inspect._empty else str
            properties[name] = {
                "type": _annotation_to_json_type(annotation),
            }
            if parameter.default is inspect._empty:
                required.append(name)

        normalized_tools.append({
            "type": "function",
            "function": {
                "name": tool.__name__,
                "description": inspect.getdoc(tool) or "",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        })

    return normalized_tools


def _build_headers(config: RoleProviderConfig) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
    }

    if config.api_key_env != "":
        api_key = os.getenv(config.api_key_env, "")
        if api_key != "":
            headers["Authorization"] = f"Bearer {api_key}"

    return headers


def _post_json(url: str, payload: dict[str, Any], config: RoleProviderConfig) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=url,
        data=body,
        headers=_build_headers(config),
        method="POST",
    )

    with request.urlopen(req, timeout=config.timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def create_chat_provider(config: RoleProviderConfig) -> ChatProvider:
    if config.provider == "ollama":
        return OllamaChatProvider(config)

    if config.provider in {"huggingface", "openai_compatible"}:
        return HostedChatProvider(config)

    raise ValueError(f"Unsupported chat provider '{config.provider}'")


def create_embedding_provider(config: RoleProviderConfig) -> EmbeddingProvider:
    if config.provider == "ollama":
        return OllamaEmbeddingProvider(config)

    if config.provider in {"huggingface", "openai_compatible"}:
        return HostedEmbeddingProvider(config)

    raise ValueError(f"Unsupported embedding provider '{config.provider}'")


def create_text_generation_provider(config: RoleProviderConfig) -> TextGenerationProvider:
    if config.provider == "ollama":
        return OllamaTextGenerationProvider(config)

    if config.provider in {"huggingface", "openai_compatible"}:
        return HostedTextGenerationProvider(config)

    raise ValueError(f"Unsupported text provider '{config.provider}'")
