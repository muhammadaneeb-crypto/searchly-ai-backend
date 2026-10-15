"""GroqCloud client. GroqCloud is the ONLY external AI API used by this project."""
import asyncio
import logging
from typing import Optional

import httpx

from app.config import Settings
from app.utils.errors import AppError

log = logging.getLogger("searchly.groq")


class GroqError(AppError):
    pass


class GroqService:
    def __init__(self, settings: Settings, client: Optional[httpx.AsyncClient] = None):
        self.settings = settings
        self._client = client
        self._owns_client = client is None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.settings.groq_timeout_seconds)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()

    async def chat(self, messages: list[dict], *, temperature: float = 0.2, max_tokens: int = 1800,
                   json_mode: bool = False) -> str:
        s = self.settings
        if not s.groq_configured:
            raise GroqError(503, "groq_not_configured",
                            "The server has no GroqCloud API key. Add GROQ_API_KEY to backend/.env and restart the backend.")
        body: dict = {"model": s.groq_model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {s.groq_api_key.strip()}", "Content-Type": "application/json"}
        url = s.groq_base_url.rstrip("/") + "/chat/completions"

        last_error: Optional[GroqError] = None
        for attempt in range(s.groq_max_retries + 1):
            try:
                resp = await self._get_client().post(url, json=body, headers=headers, timeout=s.groq_timeout_seconds)
            except httpx.TimeoutException:
                last_error = GroqError(504, "groq_timeout", "GroqCloud took too long to answer. Please try again.")
            except httpx.TransportError:
                last_error = GroqError(503, "groq_unreachable",
                                       "The backend could not reach GroqCloud. Check the server's internet connection.")
            else:
                if resp.status_code == 200:
                    return self._parse(resp)
                last_error = self._http_error(resp)
                if resp.status_code not in (429, 500, 502, 503, 504):
                    raise last_error
            if attempt < s.groq_max_retries:
                delay = 0.5 * (2 ** attempt)
                log.warning("Groq request failed (%s); retrying in %.1fs", last_error.code, delay)
                await asyncio.sleep(delay)
        assert last_error is not None
        raise last_error

    @staticmethod
    def _http_error(resp: httpx.Response) -> GroqError:
        code = resp.status_code
        if code in (401, 403):
            return GroqError(503, "groq_auth_failed",
                             "GroqCloud rejected the API key. Check GROQ_API_KEY in backend/.env (and that the key is active).")
        if code == 429:
            return GroqError(429, "groq_rate_limited", "GroqCloud rate limit reached. Wait a moment and try again.")
        if code in (400, 404):
            return GroqError(502, "groq_request_rejected",
                             "GroqCloud rejected the request. The GROQ_MODEL name may be wrong or retired; check console.groq.com/docs/models.")
        return GroqError(502, "groq_server_error", "GroqCloud returned an error. Please try again shortly.")

    @staticmethod
    def _parse(resp: httpx.Response) -> str:
        try:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise GroqError(502, "groq_bad_response", "GroqCloud sent a response the backend could not understand.")
        if not isinstance(content, str) or not content.strip():
            raise GroqError(502, "groq_bad_response", "GroqCloud returned an empty answer.")
        return content.strip()
