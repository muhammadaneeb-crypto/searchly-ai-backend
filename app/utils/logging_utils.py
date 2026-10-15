import json
import logging
import re

_SECRET_RE = re.compile(r"(gsk_[A-Za-z0-9]{8,}|Bearer\s+[A-Za-z0-9._\-]{8,})")


class RedactFilter(logging.Filter):
    """Removes anything that looks like an API key from log lines."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _SECRET_RE.sub("[REDACTED]", str(record.msg))
        if record.args:
            record.args = tuple(_SECRET_RE.sub("[REDACTED]", str(a)) if isinstance(a, str) else a for a in record.args)
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"time": self.formatTime(record), "level": record.levelname, "logger": record.name, "message": record.getMessage()}
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exception"] = _SECRET_RE.sub("[REDACTED]", self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RedactFilter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
