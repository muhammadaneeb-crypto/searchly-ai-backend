from dataclasses import dataclass
from typing import Optional

from app.config import Settings
from app.services.answer_service import AnswerService
from app.services.document_service import DocumentService
from app.services.groq_service import GroqService
from app.services.speech_service import SpeechService
from app.services.web_search_service import WebSearchService
from app.services.webpage_extractor import WebPageExtractor
from app.utils.store import ConversationStore


@dataclass
class Container:
    settings: Settings
    groq: GroqService
    search: WebSearchService
    extractor: WebPageExtractor
    documents: DocumentService
    speech: SpeechService
    store: ConversationStore
    answers: AnswerService

    @classmethod
    def build(cls, settings: Settings, *, groq=None, search=None, extractor=None, documents=None, speech=None,
              store: Optional[ConversationStore] = None) -> "Container":
        groq = groq or GroqService(settings)
        search = search or WebSearchService(settings)
        extractor = extractor or WebPageExtractor(settings)
        store = store or ConversationStore()
        return cls(settings, groq, search, extractor, documents or DocumentService(settings),
                   speech or SpeechService(settings), store, AnswerService(settings, groq, search, extractor, store))

    async def aclose(self) -> None:
        for part in (self.groq, self.extractor):
            close = getattr(part, "aclose", None)
            if close:
                await close()
