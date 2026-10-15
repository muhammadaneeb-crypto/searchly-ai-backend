"""Local speech-to-text with faster-whisper (optional dependency). No external STT API is used."""
import asyncio
import logging
import os
import tempfile
import threading
from dataclasses import dataclass
from typing import Optional

from app.config import Settings
from app.utils.errors import AppError

log = logging.getLogger("searchly.speech")
AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".ogg", ".webm", ".aac", ".mp4", ".flac", ".opus", ".3gp"}


@dataclass
class Transcription:
    text: str
    language: Optional[str]
    probability: Optional[float]


class SpeechService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._lock = threading.Lock()

    def available(self) -> bool:
        if not self.settings.whisper_enabled:
            return False
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def _load(self):
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel
                s = self.settings
                self._model = WhisperModel(s.whisper_model, device=s.whisper_device, compute_type=s.whisper_compute_type)
            return self._model

    def validate(self, filename: str, data: bytes) -> str:
        ext = os.path.splitext(filename or "")[1].lower()
        if ext not in AUDIO_EXTS:
            raise AppError(415, "unsupported_audio_type", "Unsupported audio format. Use m4a, mp3, wav, ogg, webm, aac or flac.")
        if not data:
            raise AppError(422, "empty_audio", "The audio file is empty.")
        if len(data) > self.settings.max_audio_bytes:
            raise AppError(413, "audio_too_large", f"The audio is larger than the {self.settings.max_audio_mb} MB limit.")
        return ext

    def _transcribe_blocking(self, path: str, language: Optional[str]) -> Transcription:
        model = self._load()
        segments, info = model.transcribe(path, language=language, vad_filter=True, beam_size=1)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return Transcription(text, getattr(info, "language", None), getattr(info, "language_probability", None))

    async def transcribe(self, filename: str, data: bytes, language: Optional[str] = None) -> Transcription:
        ext = self.validate(filename, data)
        if not self.settings.whisper_enabled:
            raise AppError(503, "speech_disabled", "Voice transcription is turned off on this server (WHISPER_ENABLED=false). Please type your question.")
        if not self.available():
            raise AppError(503, "speech_unavailable",
                           "Voice transcription is not installed on this server (run: pip install faster-whisper). Please type your question instead.")
        fd, path = tempfile.mkstemp(suffix=ext, prefix="searchly_audio_")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            try:
                result = await asyncio.to_thread(self._transcribe_blocking, path, language)
            except AppError:
                raise
            except Exception as exc:
                log.warning("transcription failed: %s", type(exc).__name__)
                raise AppError(500, "transcription_failed",
                               "The audio could not be transcribed. The file may be damaged, or the speech model could not load. Please type your question instead.")
        finally:
            try:
                os.remove(path)  # always delete the temporary audio file
            except OSError:
                pass
        if not result.text:
            raise AppError(422, "no_speech_detected", "No speech was detected in the recording. Please try again or type your question.")
        return result
