from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.deps import get_client_id, get_container
from app.api.upload import read_limited
from app.container import Container
from app.models.schemas import VoiceResponse

router = APIRouter(prefix="/api", tags=["voice"])


@router.post("/voice", response_model=VoiceResponse)
async def voice(file: UploadFile = File(...), language: Optional[str] = Form(None),
                answer: bool = Form(False), conversation_id: Optional[str] = Form(None),
                response_language: Optional[str] = Form(None), c: Container = Depends(get_container),
                client_id: str = Depends(get_client_id)):
    """Transcribe locally. By default only the transcript is returned so the user can check it first.
    Send answer=true to also run the transcript through the normal search pipeline."""
    data = await read_limited(file, c.settings.max_audio_bytes, "audio_too_large", "audio file")
    result = await c.speech.transcribe(file.filename or "audio.m4a", data, language=(language or None))
    out = VoiceResponse(conversation_id=conversation_id, transcription=result.text, language=result.language,
                        language_probability=result.probability)
    if answer:
        conv = c.store.get_or_create(conversation_id, client_id)
        out.conversation_id = conv.id
        out.answer = await c.answers.answer(conv, result.text, response_language=response_language)
    return out
