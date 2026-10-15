"""The agent pipeline: understand -> plan -> search -> read -> rank -> answer with citations."""
import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from app.config import Settings
from app.models.schemas import AnswerResponse, Source
from app.services import prompts
from app.services.document_service import retrieve
from app.services.groq_service import GroqService
from app.services.web_search_service import SearchResult, SearchUnavailable, WebSearchService
from app.services.webpage_extractor import WebPageExtractor
from app.utils.language import detect_language, language_from_code
from app.utils.safety import neutralize, safe_attr
from app.utils.store import Conversation, ConversationStore

log = logging.getLogger("searchly.answer")
_CITATION_RE = re.compile(r"[\[【](\d{1,3}(?:\s*[,;]\s*\d{1,3})*)[\]】]")
_CODE_RE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)


@dataclass
class Plan:
    needs_web: bool
    queries: list[str]
    recency: bool = False


@dataclass
class Evidence:
    source: Source
    text: str


def parse_plan(raw: str, fallback_query: str) -> Plan:
    """Parse the planner's JSON defensively; fall back to searching the user's own words."""
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group(0)) if match else {}
        queries = [str(q).strip() for q in data.get("queries", []) if str(q).strip()][:5]
        return Plan(bool(data.get("needs_web", True)), queries or [fallback_query[:200]], bool(data.get("recency", False)))
    except (ValueError, AttributeError, TypeError):
        return Plan(True, [fallback_query[:200]], False)


def apply_citations(answer: str, valid_ids: set[int]) -> tuple[str, set[int]]:
    """Drop citation numbers that do not match a real source. Code blocks are left untouched."""
    cited: set[int] = set()

    def fix(m: re.Match) -> str:
        nums = [int(n) for n in re.split(r"[,;]", m.group(1))]
        good = [n for n in nums if n in valid_ids]
        cited.update(good)
        return f"[{', '.join(str(n) for n in good)}]" if good else ""

    parts = _CODE_RE.split(answer)
    for i in range(0, len(parts), 2):  # even indexes are normal text, odd are code
        parts[i] = _CITATION_RE.sub(fix, parts[i])
    return "".join(parts), cited


class AnswerService:
    def __init__(self, settings: Settings, groq: GroqService, search: WebSearchService,
                 extractor: WebPageExtractor, store: ConversationStore):
        self.settings, self.groq, self.search, self.extractor, self.store = settings, groq, search, extractor, store

    async def answer(self, conv: Conversation, message: str, *, response_language: Optional[str] = None,
                     use_web: Optional[bool] = None, max_sources: int = 8, user_label: Optional[str] = None,
                     warnings: Optional[list[str]] = None) -> AnswerResponse:
        warnings = list(warnings or [])
        lang = language_from_code(response_language) or detect_language(message)
        has_docs = bool(conv.documents)

        # 1) plan (GroqCloud generates search queries when needed)
        plan = Plan(False, [], False)
        if use_web is not False:
            raw = await self.groq.chat(
                [{"role": "system", "content": prompts.planner_prompt()},
                 {"role": "user", "content": f"has_uploaded_documents: {has_docs}\nforce_web_search: {use_web is True}\nUser message:\n{message[:1500]}"}],
                temperature=0, max_tokens=250, json_mode=True)
            plan = parse_plan(raw, message)
            if use_web is True:
                plan.needs_web = True

        evidence: list[Evidence] = []
        searched: list[str] = []
        flagged_any = False

        # 2) uploaded-document evidence
        if has_docs:
            all_chunks = [(d, c) for d in conv.documents for c in d.chunks]
            picked = retrieve([c for _, c in all_chunks], message, k=8)
            by_id = {id(c): d for d, c in all_chunks}
            if len(all_chunks) > len(picked):
                warnings.append("document_sampled")
            for c in picked:
                text, flagged = neutralize(c.text)
                flagged_any |= flagged
                evidence.append(Evidence(Source(id=len(evidence) + 1, title=by_id[id(c)].filename, origin="document",
                                                location=c.location, snippet=text[:160], flagged=flagged), text))

        # 3) web evidence
        web_failed = False
        if plan.needs_web:
            searched = plan.queries[: self.settings.search_max_queries]
            try:
                results = await self.search.search(searched, message, recency=plan.recency, limit=max_sources)
            except SearchUnavailable:
                results, web_failed = [], True
                warnings.append("web_search_failed")
            if not results and not web_failed:
                warnings.append("no_web_results")
            web_ev, flagged = await self._web_evidence(results, start_id=len(evidence) + 1)
            flagged_any |= flagged
            evidence.extend(web_ev)
            if any(e.source.snippet_only for e in web_ev):
                warnings.append("some_pages_unreadable")
        if flagged_any:
            warnings.append("suspicious_content_removed")

        # 4) answer
        history = self._history(conv)
        if evidence:
            system = prompts.answer_prompt(lang.name)
            user = f"<evidence>\n{self._render(evidence)}\n</evidence>\n\nUser question:\n{message}"
        elif plan.needs_web:
            system, user = prompts.no_evidence_prompt(lang.name), f"User question:\n{message}"
        else:
            system, user = prompts.chat_prompt(lang.name), message
        text = await self.groq.chat([{"role": "system", "content": system}, *history, {"role": "user", "content": user}],
                                    temperature=0.2, max_tokens=1800)

        # 5) verify citations map to real sources
        sources = [e.source for e in evidence]
        if evidence:
            text, cited = apply_citations(text, {s.id for s in sources})
            for s in sources:
                s.cited = s.id in cited
            if not cited:
                warnings.append("answer_without_citations")
        else:
            text, _ = apply_citations(text, set())

        self.store.add_message(conv, "user", user_label or message)
        self.store.add_message(conv, "assistant", text, sources)
        return AnswerResponse(
            conversation_id=conv.id, answer=text, language=lang.code, language_name=lang.name, rtl=lang.rtl,
            used_web=any(s.origin == "web" for s in sources), used_documents=any(s.origin == "document" for s in sources),
            searched_queries=searched, sources=sources, warnings=list(dict.fromkeys(warnings)))

    async def _web_evidence(self, results: list[SearchResult], start_id: int) -> tuple[list[Evidence], bool]:
        if not results:
            return [], False
        n_fetch = self.settings.pages_to_fetch
        pages = await self.extractor.fetch_many([r.url for r in results[:n_fetch]]) if n_fetch else {}
        out, flagged_any = [], False
        for r in results:
            page = pages.get(r.url)
            raw = page.text if page else r.snippet
            if not raw.strip():
                continue
            text, flagged = neutralize(raw)
            flagged_any |= flagged
            title = (page.title if page and page.title else r.title)
            out.append(Evidence(Source(id=start_id + len(out), title=title[:200], url=r.url, snippet=r.snippet[:200],
                                       date=(page.date if page and page.date else r.date), origin="web",
                                       snippet_only=page is None, flagged=flagged), text))
        return out, flagged_any

    def _render(self, evidence: list[Evidence]) -> str:
        budget = max(800, self.settings.max_context_chars // max(len(evidence), 1))
        blocks = []
        for e in evidence:
            s = e.source
            attrs = [f'id="{s.id}"', f'type="{s.origin}"', f'title="{safe_attr(s.title)}"']
            if s.url:
                attrs.append(f'url="{safe_attr(s.url, 300)}"')
            if s.location:
                attrs.append(f'location="{safe_attr(s.location)}"')
            if s.date:
                attrs.append(f'date="{safe_attr(str(s.date), 40)}"')
            if s.snippet_only:
                attrs.append('snippet_only="true"')
            blocks.append(f"<source {' '.join(attrs)}>\n{e.text[:budget]}\n</source>")
        return "\n\n".join(blocks)

    @staticmethod
    def _history(conv: Conversation, turns: int = 6) -> list[dict]:
        return [{"role": m.role, "content": m.content[:1200]} for m in conv.messages[-turns:]]
