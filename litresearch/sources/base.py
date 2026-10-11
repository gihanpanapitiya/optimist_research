"""Shared plumbing for literature sources.

Each source adapter turns a :class:`QueryPlan` into normalized
:class:`RawHit` records. Normalizing at the adapter boundary is what lets the
rest of the system treat arXiv, PubMed and a web page uniformly, and it is
where the per-source API quirks are meant to stay.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import get_settings
from ..state import Document, Provenance, QueryPlan, RawHit

logger = logging.getLogger(__name__)


class RateLimiter:
    """Minimum-interval limiter, one instance per source.

    Scholarly APIs police request *rate* rather than concurrency, and several
    (arXiv, unauthenticated Semantic Scholar) are strict enough that ignoring
    this gets a run throttled into uselessness.
    """

    def __init__(self, min_interval: float, max_concurrency: int = 4) -> None:
        self.min_interval = min_interval
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._last_call = 0.0

    async def __aenter__(self) -> RateLimiter:
        await self._semaphore.acquire()
        async with self._lock:
            delay = self.min_interval - (time.monotonic() - self._last_call)
            if delay > 0:
                await asyncio.sleep(delay)
            self._last_call = time.monotonic()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self._semaphore.release()


class TransientSourceError(Exception):
    """A source failure worth retrying (timeout, 429, 5xx)."""


_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    """Process-wide HTTP client, so connections are pooled across sources."""
    global _client
    if _client is None or _client.is_closed:
        settings = get_settings()
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.http_timeout_seconds),
            headers={"User-Agent": settings.user_agent},
            follow_redirects=True,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
    return _client


async def close_http_client() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


class SearchSource(ABC):
    """One literature source.

    Subclasses implement :meth:`search`, and optionally the snowball hooks.
    Not every source can do citation traversal -- arXiv's API exposes no
    reference graph, for instance -- so callers must check
    :attr:`supports_backward` / :attr:`supports_forward` first.
    """

    name: str = "base"
    min_request_interval: float = 0.34
    max_concurrency: int = 4
    supports_backward: bool = False
    supports_forward: bool = False
    requires_api_key: bool = False

    def __init__(self) -> None:
        self.settings = get_settings()
        self.limiter = RateLimiter(self.min_request_interval, self.max_concurrency)

    # -- subclass API ------------------------------------------------------

    @abstractmethod
    async def search(self, plan: QueryPlan) -> list[RawHit]:
        """Execute ``plan`` and return normalized hits."""

    async def references(self, doc: Document, limit: int = 40) -> list[RawHit]:
        """Works cited *by* ``doc`` (backward snowballing)."""
        raise NotImplementedError

    async def citations(self, doc: Document, limit: int = 40) -> list[RawHit]:
        """Works citing ``doc`` (forward snowballing)."""
        raise NotImplementedError

    def is_available(self) -> bool:
        return True

    # -- helpers for subclasses -------------------------------------------

    @retry(
        retry=retry_if_exception_type(TransientSourceError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.5, min=1, max=20),
        reraise=False,
    )
    async def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict | None = None,
        json_body: dict | None = None,
        headers: dict | None = None,
    ) -> httpx.Response | None:
        """Issue a rate-limited, retried request.

        Returns ``None`` on definitive failure. A dead source must not abort
        the research run -- the others still have work to do -- so failures
        are logged and degraded rather than raised.
        """
        client = get_http_client()
        async with self.limiter:
            try:
                response = await client.request(
                    method, url, params=params, json=json_body, headers=headers
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                raise TransientSourceError(f"{self.name}: {exc}") from exc

        if response.status_code == 429 or response.status_code >= 500:
            raise TransientSourceError(
                f"{self.name}: HTTP {response.status_code} for {url}"
            )
        if response.status_code >= 400:
            logger.warning(
                "%s: HTTP %s for %s -- %s",
                self.name,
                response.status_code,
                url,
                response.text[:200],
            )
            return None
        return response

    async def get_json(self, url: str, params: dict | None = None, headers: dict | None = None):
        try:
            response = await self._request("GET", url, params=params, headers=headers)
        except Exception as exc:  # noqa: BLE001 - retries exhausted
            logger.warning("%s: giving up on %s (%s)", self.name, url, exc)
            return None
        if response is None:
            return None
        try:
            return response.json()
        except ValueError:
            logger.warning("%s: non-JSON response from %s", self.name, url)
            return None

    async def get_text(self, url: str, params: dict | None = None) -> str | None:
        try:
            response = await self._request("GET", url, params=params)
        except Exception as exc:  # noqa: BLE001 - retries exhausted
            logger.warning("%s: giving up on %s (%s)", self.name, url, exc)
            return None
        return response.text if response is not None else None

    def provenance_for(
        self,
        plan: QueryPlan,
        rank: int,
        *,
        via: str = "search",
        seed_doc_id: str | None = None,
    ) -> Provenance:
        return Provenance(
            subquestion_id=plan.subquestion_id,
            intent_id=plan.intent_id,
            source=self.name,
            query=plan.query,
            rank=rank,
            via=via,  # type: ignore[arg-type]
            seed_doc_id=seed_doc_id,
        )

    @staticmethod
    def snowball_plan(doc: Document, direction: str) -> QueryPlan:
        """A synthetic plan so snowball hits carry sensible provenance."""
        prov = doc.discovered_by[0] if doc.discovered_by else None
        return QueryPlan(
            intent_id=prov.intent_id if prov else "snowball",
            subquestion_id=prov.subquestion_id if prov else "",
            source="snowball",
            query=f"{direction} snowball from {doc.doc_id}: {doc.title[:80]}",
        )
