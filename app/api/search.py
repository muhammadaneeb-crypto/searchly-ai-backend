from fastapi import APIRouter, Depends

from app.api.deps import get_client_id, get_container
from app.container import Container
from app.models.schemas import AnswerResponse, SearchRequest

router = APIRouter(prefix="/api", tags=["search"])


@router.post("/search", response_model=AnswerResponse)
async def search(req: SearchRequest, c: Container = Depends(get_container), client_id: str = Depends(get_client_id)):
    """Always performs web research (use_web=True) and answers with citations."""
    conv = c.store.get_or_create(req.conversation_id, client_id)
    return await c.answers.answer(conv, req.query, response_language=req.response_language, use_web=True,
                                  max_sources=req.max_results)
