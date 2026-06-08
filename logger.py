import logging
import os
import json
from contextvars import ContextVar, Token
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Optional, Protocol, cast
from uuid import uuid4

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

_configured = False
VERBOSE_LEVEL = 15
_current_conversation_id: ContextVar[str | None] = ContextVar("current_conversation_id", default=None)


def _serialize_trace_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=False)
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): _serialize_trace_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_trace_value(item) for item in value]
    return value


def _render_trace_value(value: Any) -> str:
    serialized = _serialize_trace_value(value)
    if isinstance(serialized, str):
        return serialized
    return json.dumps(serialized, indent=2, ensure_ascii=True, default=str)


class ConversationTrace:
    def __init__(
        self,
        root_dir: Path,
        character_name: str,
        profile: str,
        providers: dict[str, str],
        persist_enabled: bool,
    ) -> None:
        self.persist_enabled = persist_enabled
        self.character_name = character_name
        self.profile = profile
        self.providers = providers
        self.conversation_id = uuid4().hex
        self.started_at = datetime.now(timezone.utc)
        self._lock = Lock()
        self.path: Path | None = None

        if not self.persist_enabled:
            return

        root_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(char for char in character_name if char.isalnum() or char in {"-", "_"}).strip("_")
        if safe_name == "":
            safe_name = "conversation"
        filename = f"{self.started_at.strftime('%Y%m%d-%H%M%S')}-{safe_name}-{self.conversation_id[:8]}.txt"
        self.path = root_dir / filename
        self._append(self._render_header())

    def _append(self, text: str) -> None:
        if not self.persist_enabled or self.path is None:
            return

        with self._lock:
            with self.path.open("a", encoding="utf-8") as file:
                file.write(text)

    def _render_header(self) -> str:
        lines = [
            "=== Conversation Trace ===",
            f"conversation_id: {self.conversation_id}",
            f"character_name: {self.character_name}",
            f"started_at_utc: {self.started_at.isoformat()}",
            f"ai_profile: {self.profile}",
            f"persist_enabled: {self.persist_enabled}",
            "providers:",
            _render_trace_value(self.providers),
            "",
        ]
        return "\n".join(lines)

    def record_stage_event(
        self,
        stage_name: str,
        event: str,
        payload: Any | None = None,
        ai_request: Any | None = None,
        ai_response: Any | None = None,
        result: Any | None = None,
        status: str = "ok",
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        lines = [
            f"=== Stage: {stage_name} ===",
            f"timestamp_utc: {timestamp}",
            f"event: {event}",
            f"status: {status}",
        ]
        if payload is not None:
            lines.extend(["payload:", _render_trace_value(payload)])
        if ai_request is not None:
            lines.extend(["ai_request:", _render_trace_value(ai_request)])
        if ai_response is not None:
            lines.extend(["ai_response:", _render_trace_value(ai_response)])
        if result is not None:
            lines.extend(["result:", _render_trace_value(result)])
        lines.append("")
        self._append("\n".join(lines))


_conversation_traces: dict[str, ConversationTrace] = {}


class LoggerWithTrace(Protocol):
    def verbose(self, message: str, *args: Any, **kwargs: Any) -> None:
        ...
    def trace(self, message: str, *args, **kwargs) -> None:
        ...
    def debug(self, message: str, *args, **kwargs) -> None:
        ...
    def info(self, message: str, *args, **kwargs) -> None:
        ...
    def warning(self, message: str, *args, **kwargs) -> None:
        ...
    def error(self, message: str, *args, **kwargs) -> None:
        ...
    def critical(self, message: str, *args, **kwargs) -> None:
        ...
    def start_conversation_trace(
        self,
        root_dir: str | Path,
        character_name: str,
        profile: str,
        providers: dict[str, str],
        persist_enabled: bool,
    ) -> Token[str | None]:
        ...
    def reset_conversation_id(self, token: Token[str | None]) -> None:
        ...
    def get_conversation_id(self) -> str | None:
        ...
    def conversation_event(
        self,
        stage_name: str,
        event: str,
        payload: Any | None = None,
        ai_request: Any | None = None,
        ai_response: Any | None = None,
        result: Any | None = None,
        status: str = "ok",
    ) -> None:
        ...
    def get_conversation_trace_path(self, conversation_id: str | None) -> Path | None:
        ...


def _install_verbose_level() -> None:
    if getattr(logging, "VERBOSE", None) is None:
        logging.VERBOSE = VERBOSE_LEVEL  # type: ignore[attr-defined]
        logging.addLevelName(VERBOSE_LEVEL, "VERBOSE")

        def verbose(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
            if self.isEnabledFor(VERBOSE_LEVEL):
                self._log(VERBOSE_LEVEL, message, args, **kwargs)

        def trace(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
            self.verbose(message, *args, **kwargs)  # type: ignore[attr-defined]

        def start_conversation_trace(
            self: logging.Logger,
            root_dir: str | Path,
            character_name: str,
            profile: str,
            providers: dict[str, str],
            persist_enabled: bool,
        ) -> Token[str | None]:
            trace = ConversationTrace(
                root_dir=Path(root_dir),
                character_name=character_name,
                profile=profile,
                providers=providers,
                persist_enabled=persist_enabled,
            )
            _conversation_traces[trace.conversation_id] = trace
            return _current_conversation_id.set(trace.conversation_id)

        def reset_conversation_id(self: logging.Logger, token: Token[str | None]) -> None:
            _current_conversation_id.reset(token)

        def get_conversation_id(self: logging.Logger) -> str | None:
            return _current_conversation_id.get()

        def conversation_event(
            self: logging.Logger,
            stage_name: str,
            event: str,
            payload: Any | None = None,
            ai_request: Any | None = None,
            ai_response: Any | None = None,
            result: Any | None = None,
            status: str = "ok",
        ) -> None:
            conversation_id = _current_conversation_id.get()
            if conversation_id is None:
                return
            trace = _conversation_traces.get(conversation_id)
            if trace is None:
                return
            trace.record_stage_event(
                stage_name=stage_name,
                event=event,
                payload=payload,
                ai_request=ai_request,
                ai_response=ai_response,
                result=result,
                status=status,
            )

        def get_conversation_trace_path(self: logging.Logger, conversation_id: str | None) -> Path | None:
            if conversation_id is None:
                return None
            trace = _conversation_traces.get(conversation_id)
            return None if trace is None else trace.path

        logging.Logger.verbose = verbose  # type: ignore[assignment]
        logging.Logger.trace = trace  # type: ignore[assignment]
        logging.Logger.start_conversation_trace = start_conversation_trace  # type: ignore[assignment]
        logging.Logger.reset_conversation_id = reset_conversation_id  # type: ignore[assignment]
        logging.Logger.get_conversation_id = get_conversation_id  # type: ignore[assignment]
        logging.Logger.conversation_event = conversation_event  # type: ignore[assignment]
        logging.Logger.get_conversation_trace_path = get_conversation_trace_path  # type: ignore[assignment]


def configure_logging(level: Optional[str | int] = None) -> None:
    global _configured
    _install_verbose_level()
    if _configured:
        if level is not None:
            logging.getLogger().setLevel(level)
        return

    if load_dotenv is not None:
        load_dotenv()

    resolved_level = level or os.getenv("LOG_LEVEL", "INFO")
    print("LOGLEVEL: " + str(resolved_level))
    logging.basicConfig(
        level=resolved_level,
        format="%(levelname)s | %(name)s | %(message)s",
    )
    _configured = True


def get_logger(name: str) -> LoggerWithTrace:
    return cast(LoggerWithTrace, logging.getLogger(name))
