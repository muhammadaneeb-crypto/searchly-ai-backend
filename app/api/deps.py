import re

from fastapi import Header, Request

from app.container import Container
from app.utils.errors import AppError

_CLIENT_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{8,100}$")


def get_container(request: Request) -> Container:
    return request.app.state.container


def get_client_id(x_client_id: str = Header(default="")) -> str:
    """Anonymous per-install id sent by the app. It keeps one user's chats/documents away from another's."""
    if not _CLIENT_ID_RE.match(x_client_id or ""):
        raise AppError(400, "missing_client_id", "Missing or invalid X-Client-Id header.")
    return x_client_id
