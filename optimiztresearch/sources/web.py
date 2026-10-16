"""General web search via Tavily.

Covers what the scholarly indexes miss: lab blog posts, framework docs,
technical reports, and very recent work that has not been indexed yet. Hits
from here carry no DOI and are marked as grey literature so the ranking and
critic stages can discount them against peer-reviewed sources.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import date

from ..state import QueryPlan, RawHit
from .base import SearchSource

logger = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"

# Recognize scholarly hosts so a paper landing page found via web search is
# canonicalized against the same paper found via arXiv or Semantic Scholar,
# rather than becoming a duplicate corpus entry.
ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})")
DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:a-z0-9]+)\b", re.IGNORECASE)
PUBMED_RE = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)")

# A year in a date string, not embedded in a longer number.
DATE_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
# A year in a URL only where it is a date: a whole path segment ("/2024/",
# "/2024/05/12/") or one followed by a month ("/2024-05-12"). URLs are full
# of year-like digit runs that are nothing of the kind -- the 2076 in MDPI's
# ISSN path "/2076-3417/", PMC12089777, a ScienceDirect PII, a LinkedIn
# activity id -- and each one became a publication year.
URL_YEAR_RE = re.compile(r"/((?:19|20)\d{2})(?=/|$|-(?:0[1-9]|1[0-2])(?!\d))")


class WebSource(SearchSource):
    name = "web"
    min_request_interval = 0.2
    max_concurrency = 4
    requires_api_key = True

    def __init__(self) -> None:
        super().__init__()
        self.api_key = os.getenv("TAVILY_API_KEY", "").strip()

    def is_available(self) -> bool:
        return self.settings.source_unavailable_reason("web") is None

    async def search(self, plan: QueryPlan) -> list[RawHit]:
        if not self.api_key:
            logger.debug("web: TAVILY_API_KEY not set, skipping")
            return []

        body = {
            "query": plan.query,
            "max_results": min(plan.max_results, 20),
            # "advanced" returns longer content excerpts, which matter because
            # the excerpt is all the evidence extractor gets for a web hit.
            "search_depth": "advanced",
            "include_answer": False,
            "include_raw_content": False,
        }

        try:
            response = await self._request(
                "POST",
                TAVILY_URL,
                json_body=body,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        except Exception as exc:  # noqa: BLE001 - retries exhausted
            logger.warning("web: search failed (%s)", exc)
            return []
        if response is None:
            return []

        try:
            data = response.json()
        except ValueError:
            logger.warning("web: non-JSON response from Tavily")
            return []

        hits = []
        for rank, result in enumerate(data.get("results") or []):
            hit = self._to_hit(result, plan, rank)
            if hit:
                hits.append(hit)
        return hits

    def _to_hit(self, result: dict, plan: QueryPlan, rank: int) -> RawHit | None:
        title = (result.get("title") or "").strip()
        url = (result.get("url") or "").strip()
        if not title or not url:
            return None

        content = (result.get("content") or "").strip() or None

        arxiv_id = None
        if match := ARXIV_RE.search(url):
            arxiv_id = match.group(1)

        doi = None
        if match := DOI_RE.search(url):
            doi = match.group(1).rstrip(".;,)")

        pubmed_id = None
        if match := PUBMED_RE.search(url):
            pubmed_id = match.group(1)

        return RawHit(
            source=self.name,
            source_id=url,
            title=title,
            abstract=content,
            year=self._year(result.get("published_date") or "", url),
            venue=self._host(url),
            doi=doi,
            arxiv_id=arxiv_id,
            pubmed_id=pubmed_id,
            url=url,
            pdf_url=url if url.lower().endswith(".pdf") else None,
            publication_types=["web"],
            is_preprint=False,
            provenance=self.provenance_for(plan, rank),
        )

    @staticmethod
    def _year(published: str, url: str) -> int | None:
        """Publication year from Tavily's date, else a date-shaped URL path.

        No year is far better than a wrong one: recency ranking and the
        report's source-age caveats both trust this field.
        """
        match = DATE_YEAR_RE.search(published)
        if match is None:
            match = URL_YEAR_RE.search(url.split("?", 1)[0].split("#", 1)[0])
        if match is None:
            return None
        year = int(match.group(1))
        return year if 1990 <= year <= date.today().year + 1 else None

    @staticmethod
    def _host(url: str) -> str:
        match = re.match(r"https?://([^/]+)", url)
        return match.group(1) if match else "web"
