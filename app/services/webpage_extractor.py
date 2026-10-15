"""Safely download public web pages and extract readable text.

Safety rules: only http/https, only public IP addresses (blocks SSRF to the
laptop/LAN/cloud metadata), robots.txt respected, size/time limits, no login,
paywall or CAPTCHA bypass (any 401/402/403/429 simply means "skip this page").
"""
import asyncio
import ipaddress
import logging
import socket
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup
from bs4.dammit import UnicodeDammit

from app.config import Settings

log = logging.getLogger("searchly.extractor")
_ALLOWED_TYPES = ("text/html", "application/xhtml+xml", "text/plain")
_MIN_USEFUL_CHARS = 200


@dataclass
class PageContent:
    url: str
    text: str
    title: Optional[str] = None
    date: Optional[str] = None


async def is_public_url(url: str) -> bool:
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        return False
    try:
        infos = await asyncio.to_thread(socket.getaddrinfo, p.hostname, p.port or (443 if p.scheme == "https" else 80),
                                        type=socket.SOCK_STREAM)
    except OSError:
        return False
    for info in infos:
        try:
            if not ipaddress.ip_address(info[4][0]).is_global:
                return False
        except ValueError:
            return False
    return bool(infos)


class WebPageExtractor:
    def __init__(self, settings: Settings, client: Optional[httpx.AsyncClient] = None, check_public: bool = True):
        self.settings = settings
        self._client = client
        self._check_public = check_public
        self._sem = asyncio.Semaphore(settings.max_concurrent_fetches)
        self._cache: dict[str, tuple[float, Optional[PageContent]]] = {}
        self._robots: dict[str, tuple[float, Optional[RobotFileParser]]] = {}

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.settings.page_timeout_seconds, follow_redirects=False,
                headers={"User-Agent": self.settings.user_agent, "Accept": "text/html,application/xhtml+xml,text/plain;q=0.8"})
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def fetch_many(self, urls: list[str]) -> dict[str, Optional[PageContent]]:
        results = await asyncio.gather(*(self.fetch(u) for u in urls), return_exceptions=True)
        return {u: (r if isinstance(r, PageContent) else None) for u, r in zip(urls, results)}

    async def fetch(self, url: str) -> Optional[PageContent]:
        ttl = self.settings.cache_ttl_seconds
        hit = self._cache.get(url)
        if hit and time.monotonic() - hit[0] < ttl:
            return hit[1]
        async with self._sem:
            try:
                page = await asyncio.wait_for(self._fetch(url), timeout=self.settings.page_timeout_seconds * 2)
            except Exception as exc:  # any failure just means "use the snippet instead"
                log.info("page fetch failed (%s): %s", type(exc).__name__, urlparse(url).hostname)
                page = None
        if len(self._cache) > 200:
            self._cache.clear()
        self._cache[url] = (time.monotonic(), page)
        return page

    async def _robots_allows(self, url: str) -> bool:
        if not self.settings.respect_robots:
            return True
        p = urlparse(url)
        origin = f"{p.scheme}://{p.netloc}"
        cached = self._robots.get(origin)
        if cached is None or time.monotonic() - cached[0] > 3600:
            parser: Optional[RobotFileParser] = RobotFileParser()
            try:
                resp = await self._get_client().get(origin + "/robots.txt", timeout=5)
                if resp.status_code in (401, 403):
                    parser.parse(["User-agent: *", "Disallow: /"])
                elif resp.status_code == 200:
                    parser.parse(resp.text.splitlines())
                else:
                    parser = None  # no robots.txt => allowed
            except httpx.HTTPError:
                parser = None
            cached = (time.monotonic(), parser)
            self._robots[origin] = cached
        parser = cached[1]
        return True if parser is None else parser.can_fetch(self.settings.user_agent, url)

    async def _fetch(self, url: str) -> Optional[PageContent]:
        client = self._get_client()
        current = url
        for _ in range(4):  # follow at most 3 redirects, re-checking each hop
            if self._check_public and not await is_public_url(current):
                return None
            if not await self._robots_allows(current):
                log.info("blocked by robots.txt: %s", urlparse(current).hostname)
                return None
            async with client.stream("GET", current) as resp:
                if resp.status_code in (301, 302, 303, 307, 308) and resp.headers.get("location"):
                    current = urljoin(current, resp.headers["location"])
                    continue
                if resp.status_code != 200:
                    return None  # includes 401/402/403/429: we never try to get around these
                ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                if ctype and ctype not in _ALLOWED_TYPES:
                    return None
                raw = bytearray()
                async for chunk in resp.aiter_bytes():
                    raw.extend(chunk)
                    if len(raw) > self.settings.page_max_bytes:
                        break
            return await asyncio.to_thread(self._extract, current, bytes(raw))
        return None

    def _extract(self, url: str, raw: bytes) -> Optional[PageContent]:
        html = UnicodeDammit(raw).unicode_markup or ""
        text, title, date = None, None, None
        try:
            import trafilatura
            text = trafilatura.extract(html, include_comments=False, include_tables=True, url=url)
            meta = trafilatura.extract_metadata(html)
            if meta:
                title, date = meta.title, meta.date
        except Exception:
            text = None
        soup = None
        if not text or len(text) < _MIN_USEFUL_CHARS:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript", "svg"]):
                tag.decompose()
            text = " ".join(soup.get_text(separator=" ").split())
        if not title:
            soup = soup or BeautifulSoup(html, "html.parser")
            title = soup.title.get_text(strip=True) if soup.title else None
        if not text or len(text) < _MIN_USEFUL_CHARS:
            return None
        return PageContent(url=url, text=text[: self.settings.page_text_chars], title=title, date=date)
