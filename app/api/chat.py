from fastapi import APIRouter, Depends

from app.api.deps import get_client_id, get_container
from app.container import Container
from app.models.schemas import AnswerResponse, ChatRequest, ConversationOut

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=AnswerResponse)
async def chat(req: ChatRequest, c: Container = Depends(get_container), client_id: str = Depends(get_client_id)):
    conv = c.store.get_or_create(req.conversation_id, client_id)
    return await c.answers.answer(conv, req.message, response_language=req.response_language, use_web=req.use_web)


@router.get("/conversations/{conversation_id}", response_model=ConversationOut)
async def get_conversation(conversation_id: str, c: Container = Depends(get_container),
                           client_id: str = Depends(get_client_id)):
    conv = c.store.get(conversation_id, client_id)
    return ConversationOut(conversation_id=conv.id, messages=conv.messages, documents=[d.filename for d in conv.documents])


@router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: str, c: Container = Depends(get_container),
                              client_id: str = Depends(get_client_id)):
    c.store.clear(conversation_id, client_id)
