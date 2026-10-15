import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import chat, search, upload, voice
from app.config import Settings, get_settings
from app.container import Container
from app.utils.errors import AppError
from app.utils.logging_utils import setup_logging

log = logging.getLogger("searchly")
VERSION = "1.0.0"


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def _ocr_available() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def create_app(container: Optional[Container] = None, settings: Optional[Settings] = None) -> FastAPI:
    owns_container = container is None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        s = settings or (container.settings if container else get_settings())
        setup_logging(s.log_level)
        for problem in s.config_problems():  # startup validation: loud, but never prints secret values
            log.error("CONFIG PROBLEM: %s", problem)
        app.state.container = container or Container.build(s)
        log.info("Searchly AI backend started (model=%s)", s.groq_model)
        yield
        if owns_container:
            await app.state.container.aclose()

    app = FastAPI(title="Searchly AI Backend", version=VERSION, lifespan=lifespan)

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        rid, start = uuid.uuid4().hex[:8], time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        log.info("%s %s -> %s", request.method, request.url.path, response.status_code,
                 extra={"extra_fields": {"request_id": rid, "ms": round((time.perf_counter() - start) * 1000)}})
        return response

    @app.exception_handler(AppError)
    async def app_error(_: Request, exc: AppError):
        return _error(exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        fields = ", ".join(sorted({str(e["loc"][-1]) for e in exc.errors() if e.get("loc")})) or "request"
        return _error(422, "invalid_request", f"Invalid or missing value for: {fields}.")

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException):
        return _error(exc.status_code, "http_error", "Request could not be completed." if exc.status_code >= 500 else str(exc.detail))

    @app.exception_handler(Exception)
    async def unexpected(_: Request, exc: Exception):
        log.exception("unhandled error")  # full details go to the server log only
        return _error(500, "internal_error", "Something went wrong on the server. Please try again.")

    @app.get("/health")
    async def health(request: Request):
        c: Container = request.app.state.container
        return {"status": "ok", "version": VERSION, "model": c.settings.groq_model,
                "groq_configured": c.settings.groq_configured, "speech_available": c.speech.available(),
                "ocr_available": _ocr_available(), "problems": c.settings.config_problems()}

    for r in (chat.router, search.router, upload.router, voice.router):
        app.include_router(r)
    return app


app = create_app()
