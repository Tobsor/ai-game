import inspect
import json
import os
from dataclasses import dataclass
from typing import Any, Protocol
from urllib import request

import ollama

from ai.settings import RoleProviderConfig

try:
    from huggingface_hub import InferenceClient
except ImportError:  # pragma: no cover - exercised only when dependency is missing at runtime
    InferenceClient = None  # type: ignore[assignment]


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


def _extract_message_content(message: Any) -> str:
    if message is None:
        return ""

    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
            else:
                text_value = getattr(item, "text", "")
                if text_value != "":
                    text_parts.append(str(text_value))
        return "".join(text_parts)

    return str(content)


def _extract_message_tool_calls(message: Any) -> Any:
    if message is None:
        return None

    if isinstance(message, dict):
        return message.get("tool_calls")

    return getattr(message, "tool_calls", None)


def _flatten_messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    prompt_parts: list[str] = []

    for message in messages:
        role = str(message.get("role", "user")).strip() or "user"
        content = _extract_message_content(message).strip()
        if content == "":
            continue
        prompt_parts.append(f"{role}: {content}")

    return "\n\n".join(prompt_parts)


def _coerce_embedding_payload(embedding: Any) -> list[float]:
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()

    if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
        first_row = embedding[0]
        return [float(value) for value in first_row]

    if isinstance(embedding, list):
        return [float(value) for value in embedding]

    return []


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

        return ChatCompletionResult(
            content=_extract_message_content(message),
            tool_calls=_normalize_tool_calls(_extract_message_tool_calls(message)),
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


class OpenAICompatibleChatProvider:
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
            content=_extract_message_content(message),
            tool_calls=_normalize_tool_calls(_extract_message_tool_calls(message)),
        )


class OpenAICompatibleEmbeddingProvider:
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


class OpenAICompatibleTextGenerationProvider:
    def __init__(self, config: RoleProviderConfig):
        self.chat_provider = OpenAICompatibleChatProvider(config)

    def generate(self, prompt: str) -> str:
        result = self.chat_provider.chat(messages=[{"role": "user", "content": prompt}])
        return result.content


class HuggingFaceInferenceProviderBase:
    def __init__(self, config: RoleProviderConfig):
        self.config = config
        self.client = self._create_client()

    def _create_client(self) -> Any:
        if InferenceClient is None:
            raise ImportError(
                "huggingface_hub is required for provider='huggingface'. "
                "Install it with `pip install huggingface_hub`."
            )

        api_key = ""
        if self.config.api_key_env != "":
            api_key = self.config.api_key_env

        client_kwargs: dict[str, Any] = {
            "timeout": float(self.config.timeout_seconds),
        }
        if self.config.hf_provider != "":
            client_kwargs["provider"] = self.config.hf_provider
        if self.config.base_url != "":
            client_kwargs["base_url"] = self.config.base_url
        if api_key != "":
            client_kwargs["api_key"] = api_key

        return InferenceClient(**client_kwargs)


class HuggingFaceChatProvider(HuggingFaceInferenceProviderBase):
    def chat(self, messages: list[dict[str, Any]], tools: list[Any] | None = None) -> ChatCompletionResult:
        if self.config.hf_provider == "featherless-ai":
            prompt = _flatten_messages_to_prompt(messages)
            result = self.client.text_generation(
                prompt=prompt,
                model=self.config.model,
                max_new_tokens=500
            )
            content = result if isinstance(result, str) else str(result)
            return ChatCompletionResult(content=content, tool_calls=[])

        kwargs: dict[str, Any] = {
            "messages": messages,
            "model": self.config.model,
        }
        if tools is not None:
            kwargs["tools"] = _normalize_tool_definitions(tools)

        result = self.client.chat_completion(**kwargs)
        message = getattr(result, "choices", [None])[0]
        normalized_message = getattr(message, "message", None)

        return ChatCompletionResult(
            content=_extract_message_content(normalized_message),
            tool_calls=_normalize_tool_calls(_extract_message_tool_calls(normalized_message)),
        )


class HuggingFaceEmbeddingProvider(HuggingFaceInferenceProviderBase):
    def embed(self, text: str) -> list[list[float]]:
        embedding = self.client.feature_extraction(
            text=text,
            model=self.config.model,
        )
        return [_coerce_embedding_payload(embedding)]


class HuggingFaceTextGenerationProvider(HuggingFaceInferenceProviderBase):
    def generate(self, prompt: str) -> str:
        result = self.client.text_generation(
            prompt=prompt,
            model=self.config.model,
        )

        return result if isinstance(result, str) else str(result)


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
        api_key = config.api_key_env
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

    if config.provider == "huggingface":
        return HuggingFaceChatProvider(config)

    if config.provider == "openai_compatible":
        return OpenAICompatibleChatProvider(config)

    raise ValueError(f"Unsupported chat provider '{config.provider}'")


def create_embedding_provider(config: RoleProviderConfig) -> EmbeddingProvider:
    if config.provider == "ollama":
        return OllamaEmbeddingProvider(config)

    if config.provider == "huggingface":
        return HuggingFaceEmbeddingProvider(config)

    if config.provider == "openai_compatible":
        return OpenAICompatibleEmbeddingProvider(config)

    raise ValueError(f"Unsupported embedding provider '{config.provider}'")


def create_text_generation_provider(config: RoleProviderConfig) -> TextGenerationProvider:
    if config.provider == "ollama":
        return OllamaTextGenerationProvider(config)

    if config.provider == "huggingface":
        return HuggingFaceTextGenerationProvider(config)

    if config.provider == "openai_compatible":
        return OpenAICompatibleTextGenerationProvider(config)

    raise ValueError(f"Unsupported text provider '{config.provider}'")
