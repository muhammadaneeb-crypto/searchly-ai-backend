"""Real web search (DuckDuckGo via the `ddgs` library) with de-duplication and ranking."""
import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Protocol
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from app.config import Settings

log = logging.getLogger("searchly.search")


class SearchUnavailable(Exception):
    """Every search request failed (network down, blocked, timeout)."""


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    date: Optional[str] = None
    query: str = ""
    position: int = 0
    score: float = 0.0


class SearchBackend(Protocol):
    def text(self, query: str, max_results: int, region: str, timelimit: Optional[str]) -> list[dict]: ...
    def news(self, query: str, max_results: int, region: str, timelimit: Optional[str]) -> list[dict]: ...


class DDGSBackend:
    """Blocking calls; always run through asyncio.to_thread."""

    def _ddgs(self):
        try:
            from ddgs import DDGS
        except ImportError:  # older package name
            from duckduckgo_search import DDGS  # type: ignore
        return DDGS()

    def text(self, query, max_results, region, timelimit):
        return list(self._ddgs().text(query, region=region, max_results=max_results, timelimit=timelimit) or [])

    def news(self, query, max_results, region, timelimit):
        return list(self._ddgs().news(query, region=region, max_results=max_results, timelimit=timelimit) or [])


_TRACKING = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "fbclid", "gclid", "ref", "ref_src"}
_LOW_VALUE_HOSTS = {"pinterest.com", "facebook.com", "instagram.com", "tiktok.com"}
_TRUSTED_HOSTS = {
    "wikipedia.org": 1.0, "python.org": 2.0, "mozilla.org": 2.0, "w3.org": 2.0, "who.int": 2.5, "nih.gov": 2.5,
    "nature.com": 2.0, "arxiv.org": 1.5, "github.com": 1.0, "stackoverflow.com": 1.0, "bbc.com": 1.5, "bbc.co.uk": 1.5,
    "reuters.com": 1.5, "apnews.com": 1.5, "flutter.dev": 2.0, "dart.dev": 2.0, "microsoft.com": 1.5,
    "fastapi.tiangolo.com": 2.0, "groq.com": 2.0, "britannica.com": 1.0, "developer.android.com": 2.0,
}


def canonical_url(url: str) -> str:
    p = urlparse(url.strip())
    host = (p.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    query = urlencode([(k, v) for k, v in parse_qsl(p.query) if k.lower() not in _TRACKING])
    port = f":{p.port}" if p.port and p.port not in (80, 443) else ""
    return urlunparse(("https" if p.scheme in ("http", "https") else p.scheme, host + port, p.path.rstrip("/") or "/", "", query, ""))


def host_of(url: str) -> str:
    h = (urlparse(url).hostname or "").lower()
    return h[4:] if h.startswith("www.") else h


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"\w+", text.lower()) if len(t) > 1}


def trust_bonus(url: str) -> float:
    host = host_of(url)
    bonus = 0.0
    if host.endswith((".gov", ".edu", ".int")) or ".gov." in host or ".edu." in host or ".ac." in host:
        bonus += 2.5
    if host.startswith(("docs.", "developer.", "support.", "learn.", "help.")):
        bonus += 1.5
    for domain, b in _TRUSTED_HOSTS.items():
        if host == domain or host.endswith("." + domain):
            bonus += b
            break
    return bonus


def rank_results(results: list[SearchResult], question: str, recency: bool = False) -> list[SearchResult]:
    q = _tokens(question)
    for r in results:
        overlap = len(q & _tokens(f"{r.title} {r.snippet}")) / max(len(q), 1)
        score = overlap * 3 + trust_bonus(r.url) + max(0.0, 1.5 - 0.15 * r.position)
        if recency and r.date:
            score += 1.5
        if host_of(r.url) in _LOW_VALUE_HOSTS:
            score -= 3
        r.score = score
    return sorted(results, key=lambda r: r.score, reverse=True)


def diversify(results: list[SearchResult], limit: int, per_host: int = 2) -> list[SearchResult]:
    """Prefer several different websites rather than many pages from one."""
    picked, counts = [], {}
    for r in results:
        h = host_of(r.url)
        if counts.get(h, 0) >= per_host:
            continue
        counts[h] = counts.get(h, 0) + 1
        picked.append(r)
        if len(picked) >= limit:
            break
    return picked


class WebSearchService:
    def __init__(self, settings: Settings, backend: Optional[SearchBackend] = None):
        self.settings = settings
        self.backend = backend or DDGSBackend()

    async def _one(self, fn, query: str, n: int, timelimit: Optional[str]) -> list[dict]:
        return await asyncio.wait_for(
            asyncio.to_thread(fn, query, n, self.settings.search_region, timelimit),
            timeout=self.settings.search_timeout_seconds,
        )

    async def search(self, queries: list[str], question: str, recency: bool = False,
                     limit: int = 8) -> list[SearchResult]:
        s = self.settings
        timelimit = "m" if recency else None
        jobs, labels = [], []
        for q in queries[: s.search_max_queries]:
            jobs.append(self._one(self.backend.text, q, s.search_results_per_query, timelimit))
            labels.append(q)
            if recency:
                jobs.append(self._one(self.backend.news, q, max(3, s.search_results_per_query // 2), timelimit))
                labels.append(q)
        outcomes = await asyncio.gather(*jobs, return_exceptions=True)

        raw: list[SearchResult] = []
        failures = 0
        for q, out in zip(labels, outcomes):
            if isinstance(out, BaseException):
                failures += 1
                log.warning("search request failed: %s", type(out).__name__)
                continue
            for pos, item in enumerate(out):
                url = item.get("href") or item.get("url") or ""
                title = (item.get("title") or "").strip()
                if not url.startswith(("http://", "https://")) or not title:
                    continue
                raw.append(SearchResult(title=title, url=url, snippet=(item.get("body") or item.get("excerpt") or "").strip(),
                                        date=item.get("date"), query=q, position=pos))
        if failures == len(outcomes) and outcomes:
            raise SearchUnavailable("all search requests failed")

        seen, unique = set(), []
        for r in raw:
            key = canonical_url(r.url)
            if key in seen:
                continue
            seen.add(key)
            unique.append(r)
        ranked = rank_results(unique, question, recency)
        return diversify(ranked, limit)
