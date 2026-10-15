"""In-memory conversation store. Each conversation belongs to one client id."""
import secrets
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.models.schemas import Source, StoredMessage
from app.utils.errors import AppError


@dataclass
class StoredDocument:
    filename: str
    file_type: str
    chunks: list  # list[app.services.document_service.Chunk]
    stats: dict = field(default_factory=dict)


@dataclass
class Conversation:
    id: str
    owner: str
    messages: list[StoredMessage] = field(default_factory=list)
    documents: list[StoredDocument] = field(default_factory=list)


class ConversationStore:
    def __init__(self, max_conversations: int = 500, max_messages: int = 100, max_documents: int = 5):
        self._items: "OrderedDict[str, Conversation]" = OrderedDict()
        self._lock = threading.Lock()
        self.max_conversations = max_conversations
        self.max_messages = max_messages
        self.max_documents = max_documents

    def get_or_create(self, conversation_id: str | None, owner: str) -> Conversation:
        with self._lock:
            if conversation_id:
                conv = self._items.get(conversation_id)
                if conv is None:
                    # Unknown id (e.g. backend restarted): start a fresh conversation with that id.
                    conv = Conversation(id=conversation_id, owner=owner)
                    self._items[conversation_id] = conv
                elif conv.owner != owner:
                    # Same message for "not found" and "not yours" so ids cannot be probed.
                    raise AppError(404, "conversation_not_found", "Conversation not found.")
            else:
                conv = Conversation(id=secrets.token_urlsafe(16), owner=owner)
                self._items[conv.id] = conv
            self._items.move_to_end(conv.id)
            while len(self._items) > self.max_conversations:
                self._items.popitem(last=False)
            return conv

    def get(self, conversation_id: str, owner: str) -> Conversation:
        with self._lock:
            conv = self._items.get(conversation_id)
            if conv is None or conv.owner != owner:
                raise AppError(404, "conversation_not_found", "Conversation not found.")
            return conv

    def add_message(self, conv: Conversation, role: str, content: str, sources: list[Source] | None = None) -> None:
        conv.messages.append(StoredMessage(role=role, content=content, sources=sources or [],
                                           created_at=datetime.now(timezone.utc).isoformat()))
        if len(conv.messages) > self.max_messages:
            del conv.messages[: len(conv.messages) - self.max_messages]

    def add_document(self, conv: Conversation, doc: StoredDocument) -> None:
        conv.documents.append(doc)
        if len(conv.documents) > self.max_documents:
            del conv.documents[0]

    def clear(self, conversation_id: str, owner: str) -> None:
        with self._lock:
            conv = self._items.get(conversation_id)
            if conv and conv.owner == owner:
                del self._items[conversation_id]
