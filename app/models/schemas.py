from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Source(BaseModel):
    id: int
    title: str
    url: Optional[str] = None
    snippet: str = ""
    date: Optional[str] = None
    origin: Literal["web", "document"] = "web"
    location: Optional[str] = None  # e.g. "page 3", "sheet Sales, rows 2-40"
    cited: bool = False
    snippet_only: bool = False  # web page could not be opened; only the search snippet was used
    flagged: bool = False  # contained text that looked like a prompt-injection attempt


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: Optional[str] = Field(default=None, max_length=64)
    response_language: Optional[str] = Field(default=None, max_length=40)
    use_web: Optional[bool] = None  # None = let the agent decide

    @field_validator("message")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v.strip()


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    conversation_id: Optional[str] = Field(default=None, max_length=64)
    response_language: Optional[str] = Field(default=None, max_length=40)
    max_results: int = Field(default=6, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v.strip()


class AnswerResponse(BaseModel):
    conversation_id: str
    answer: str
    language: str
    language_name: str
    rtl: bool = False
    used_web: bool = False
    used_documents: bool = False
    searched_queries: list[str] = []
    sources: list[Source] = []
    warnings: list[str] = []


class UploadResponse(AnswerResponse):
    filename: str
    file_type: str
    stats: dict = {}


class VoiceResponse(BaseModel):
    conversation_id: Optional[str] = None
    transcription: str
    language: Optional[str] = None
    language_probability: Optional[float] = None
    answer: Optional[AnswerResponse] = None


class StoredMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    sources: list[Source] = []
    created_at: str


class ConversationOut(BaseModel):
    conversation_id: str
    messages: list[StoredMessage]
    documents: list[str] = []


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody
