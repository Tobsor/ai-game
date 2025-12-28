import logging
import os
from typing import Optional, Protocol, cast

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

_configured = False
TRACE_LEVEL = 15


class LoggerWithTrace(Protocol):
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


def _install_trace_level() -> None:
    if getattr(logging, "TRACE", None) is None:
        logging.TRACE = TRACE_LEVEL  # type: ignore[attr-defined]
        logging.addLevelName(TRACE_LEVEL, "TRACE")

        def trace(self: logging.Logger, message: str, *args, **kwargs) -> None:
            if self.isEnabledFor(TRACE_LEVEL):
                self._log(TRACE_LEVEL, message, args, **kwargs)

        logging.Logger.trace = trace  # type: ignore[assignment]


def configure_logging(level: Optional[str | int] = None) -> None:
    global _configured
    _install_trace_level()
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
