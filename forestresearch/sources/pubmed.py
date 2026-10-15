"""PubMed source adapter via NCBI E-utilities.

Two-step protocol: ``esearch`` returns PMIDs, then ``efetch`` returns full
records. We use efetch rather than the lighter esummary because esummary
omits abstracts, and abstracts are what the evidence extractor reads in the
default (non-full-text) mode.
"""

from __future__ import annotations

import logging
import os
import re
from xml.etree import ElementTree

from ..state import QueryPlan, RawHit
from .base import SearchSource

logger = logging.getLogger(__name__)

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

REVIEW_TYPES = {
    "Review",
    "Systematic Review",
    "Meta-Analysis",
    "Editorial",
    "Letter",
    "Comment",
    "News",
}


class PubMedSource(SearchSource):
    name = "pubmed"

    def __init__(self) -> None:
        self.api_key = os.getenv("NCBI_API_KEY", "").strip()
        # NCBI allows 3 req/s unauthenticated, 10 req/s with a key.
        self.min_request_interval = 0.11 if self.api_key else 0.35
        self.max_concurrency = 3 if self.api_key else 1
        super().__init__()

    def _common_params(self) -> dict:
        params = {"db": "pubmed", "tool": "forestresearch"}
        if self.settings.contact_email:
            params["email"] = self.settings.contact_email
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    async def search(self, plan: QueryPlan) -> list[RawHit]:
        pmids = await self._esearch(plan)
        if not pmids:
            return []
        return await self._efetch(pmids, plan)

    async def _esearch(self, plan: QueryPlan) -> list[str]:
        params = self._common_params()
        params.update(
            {
                "term": plan.query,
                "retmax": str(min(plan.max_results, 50)),
                "retmode": "json",
                "sort": "relevance",
            }
        )
        if plan.year_from:
            params["mindate"] = str(plan.year_from)
            params["maxdate"] = str(plan.year_to or 3000)
            params["datetype"] = "pdat"

        data = await self.get_json(f"{BASE}/esearch.fcgi", params=params)
        if not data:
            return []
        return list((data.get("esearchresult") or {}).get("idlist") or [])

    async def _efetch(self, pmids: list[str], plan: QueryPlan) -> list[RawHit]:
        params = self._common_params()
        params.update({"id": ",".join(pmids), "retmode": "xml"})

        text = await self.get_text(f"{BASE}/efetch.fcgi", params=params)
        if not text:
            return []

        try:
            root = ElementTree.fromstring(text)
        except ElementTree.ParseError as exc:
            logger.warning("pubmed: could not parse efetch XML (%s)", exc)
            return []

        # Preserve the relevance order esearch gave us; efetch does not
        # guarantee it returns records in the requested id order.
        order = {pmid: index for index, pmid in enumerate(pmids)}
        hits = []
        for article in root.findall(".//PubmedArticle"):
            hit = self._parse_article(article, plan, order)
            if hit:
                hits.append(hit)
        hits.sort(key=lambda h: order.get(h.pubmed_id or "", 999))
        return hits

    def _parse_article(self, article, plan: QueryPlan, order: dict) -> RawHit | None:
        pmid_el = article.find(".//MedlineCitation/PMID")
        pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else None

        title_el = article.find(".//Article/ArticleTitle")
        title = self._text_of(title_el)
        if not title:
            return None

        # Structured abstracts arrive as several labelled AbstractText nodes.
        chunks = []
        for node in article.findall(".//Article/Abstract/AbstractText"):
            body = self._text_of(node)
            if not body:
                continue
            label = node.get("Label")
            chunks.append(f"{label}: {body}" if label else body)
        abstract = " ".join(chunks) or None

        authors = []
        for author in article.findall(".//Article/AuthorList/Author"):
            last = self._text_of(author.find("LastName"))
            fore = self._text_of(author.find("ForeName"))
            collective = self._text_of(author.find("CollectiveName"))
            name = collective or " ".join(p for p in [fore, last] if p)
            if name:
                authors.append(name)

        year = None
        for path in (
            ".//Article/Journal/JournalIssue/PubDate/Year",
            ".//Article/ArticleDate/Year",
            ".//MedlineCitation/DateCompleted/Year",
        ):
            year_text = self._text_of(article.find(path))
            if year_text and year_text.isdigit():
                year = int(year_text)
                break
        if year is None:
            medline_date = self._text_of(
                article.find(".//Article/Journal/JournalIssue/PubDate/MedlineDate")
            )
            if medline_date and (match := re.search(r"(19|20)\d{2}", medline_date)):
                year = int(match.group(0))

        venue = self._text_of(article.find(".//Article/Journal/Title"))

        doi = None
        for id_node in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
            if id_node.get("IdType") == "doi":
                doi = self._text_of(id_node)
                break

        pub_types = [
            self._text_of(node) or ""
            for node in article.findall(".//Article/PublicationTypeList/PublicationType")
        ]

        return RawHit(
            source=self.name,
            source_id=pmid or title[:60],
            title=title,
            abstract=abstract,
            authors=authors,
            year=year,
            venue=venue,
            doi=doi,
            pubmed_id=pmid,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
            publication_types=[t for t in pub_types if t],
            is_preprint="Preprint" in pub_types,
            provenance=self.provenance_for(plan, order.get(pmid or "", 0)),
        )

    @staticmethod
    def _text_of(element) -> str | None:
        """Flatten an element's text, including inline markup children."""
        if element is None:
            return None
        text = "".join(element.itertext())
        text = re.sub(r"\s+", " ", text).strip()
        return text or None
