import logging
import os
from typing import Optional, Protocol, cast

_configured = False
TRACE_LEVEL = 15


class LoggerWithTrace(logging.Logger, Protocol):
    def trace(self, message: str, *args, **kwargs) -> None:
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

    resolved_level = level or os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    _configured = True


def get_logger(name: str) -> LoggerWithTrace:
    return cast(LoggerWithTrace, logging.getLogger(name))
