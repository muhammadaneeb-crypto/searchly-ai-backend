import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from app.api.deps import get_client_id, get_container
from app.container import Container
from app.models.schemas import UploadResponse
from app.services.document_service import sanitize_filename
from app.utils.errors import AppError
from app.utils.store import StoredDocument

router = APIRouter(prefix="/api", tags=["upload"])


async def read_limited(file: UploadFile, limit: int, code: str, label: str) -> bytes:
    buf = bytearray()
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > limit:
            raise AppError(413, code, f"The {label} is larger than the allowed limit ({limit // (1024 * 1024)} MB).")
    return bytes(buf)


@router.post("/upload", response_model=UploadResponse)
async def upload(request: Request, file: UploadFile = File(...), question: str = Form(""),
                 conversation_id: Optional[str] = Form(None), response_language: Optional[str] = Form(None),
                 use_web: Optional[bool] = Form(None), c: Container = Depends(get_container),
                 client_id: str = Depends(get_client_id)):
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > c.settings.max_upload_bytes + 1024 * 1024:
        raise AppError(413, "file_too_large", f"The file is larger than the {c.settings.max_upload_mb} MB limit.")
    data = await read_limited(file, c.settings.max_upload_bytes, "file_too_large", "file")
    # Files are processed in memory and never written to disk, so there is nothing to execute or clean up.
    doc = await asyncio.to_thread(c.documents.process, file.filename or "upload", data)
    conv = c.store.get_or_create(conversation_id, client_id)
    c.store.add_document(conv, StoredDocument(doc.filename, doc.file_type, doc.chunks, doc.stats))
    q = question.strip()[:4000] or "Summarize this document and list its key points."
    result = await c.answers.answer(conv, q, response_language=response_language, use_web=use_web,
                                    user_label=f"[File: {sanitize_filename(doc.filename)}] {q}", warnings=doc.warnings)
    return UploadResponse(**result.model_dump(), filename=doc.filename, file_type=doc.file_type, stats=doc.stats)
