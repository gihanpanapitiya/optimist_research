"""Semantic Scholar source adapter.

The richest single source here: cross-domain metadata, citation counts, and
-- most importantly -- both directions of the citation graph, which is what
the snowballing stage runs on.

Works without a key at roughly 1 request/second. A free key raises that
considerably and is worth having for real runs.
"""

from __future__ import annotations

import logging
import os

from ..state import Document, QueryPlan, RawHit
from .base import SearchSource

logger = logging.getLogger(__name__)

BASE = "https://api.semanticscholar.org/graph/v1"

PREPRINT_VENUES = frozenset(
    {
        "arxiv",
        "arxiv.org",
        "biorxiv",
        "biorxiv.org",
        "medrxiv",
        "medrxiv.org",
        "ssrn",
        "ssrn electronic journal",
        "research square",
        "preprints.org",
        "chemrxiv",
        "osf preprints",
    }
)
"""Venue strings that denote a preprint server rather than peer review."""

_COMMON_FIELDS = [
    "paperId",
    "externalIds",
    "title",
    "abstract",
    "year",
    "venue",
    "citationCount",
    "referenceCount",
    "publicationTypes",
    "openAccessPdf",
    "url",
]

PAPER_FIELDS = ",".join([*_COMMON_FIELDS, "authors.name"])
"""Fields for /paper/search, which accepts the nested author subfield."""

CITATION_FIELDS = ",".join([*_COMMON_FIELDS, "authors"])
"""Fields for /paper/{id}/references and /citations.

These endpoints reject the nested form and 400 the whole request with
"Unrecognized or unsupported fields: [authors.name]", so they get plain
`authors` instead. The returned objects are shaped the same either way --
a list of ``{"name": ...}`` -- so normalization is unaffected.
"""


class SemanticScholarSource(SearchSource):
    name = "semantic_scholar"
    supports_backward = True
    supports_forward = True

    def __init__(self) -> None:
        self.api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
        # Set the pacing attributes before super().__init__ builds the limiter
        # from them: the unauthenticated pool is shared and much slower.
        if self.api_key:
            self.min_request_interval = 0.12
            self.max_concurrency = 4
        else:
            self.min_request_interval = 1.1
            self.max_concurrency = 1
        super().__init__()

    def is_available(self) -> bool:
        """Honour the three-way toggle in config. See Settings.semantic_scholar."""
        return self.settings.use_semantic_scholar

    def _headers(self) -> dict | None:
        return {"x-api-key": self.api_key} if self.api_key else None

    async def search(self, plan: QueryPlan) -> list[RawHit]:
        params = {
            "query": plan.query,
            "limit": min(plan.max_results, 100),
            "fields": PAPER_FIELDS,
        }
        if plan.year_from:
            params["year"] = f"{plan.year_from}-{plan.year_to or ''}"

        data = await self.get_json(
            f"{BASE}/paper/search", params=params, headers=self._headers()
        )
        if not data:
            return []

        hits = []
        for rank, paper in enumerate(data.get("data") or []):
            hit = self._to_hit(paper, self.provenance_for(plan, rank))
            if hit:
                hits.append(hit)
        return hits

    # -- snowballing -------------------------------------------------------

    async def references(self, doc: Document, limit: int = 40) -> list[RawHit]:
        return await self._citation_edge(doc, "references", "citedPaper", limit)

    async def citations(self, doc: Document, limit: int = 40) -> list[RawHit]:
        return await self._citation_edge(doc, "citations", "citingPaper", limit)

    async def _citation_edge(
        self, doc: Document, endpoint: str, payload_key: str, limit: int
    ) -> list[RawHit]:
        candidates = self._candidate_ids(doc)
        if not candidates:
            logger.debug("semantic_scholar: no usable id for %s", doc.doc_id)
            return []

        direction = "backward" if endpoint == "references" else "forward"
        plan = self.snowball_plan(doc, direction)

        # Try each identifier in turn: Semantic Scholar 404s on identifiers it
        # does not index, and which one works is not predictable from the
        # metadata. An arXiv-minted DOI (10.48550/arXiv.*) in particular is
        # not resolvable, while the bare arXiv id for the same paper is.
        data = None
        for paper_id in candidates:
            data = await self.get_json(
                f"{BASE}/paper/{paper_id}/{endpoint}",
                params={"fields": CITATION_FIELDS, "limit": min(limit, 100)},
                headers=self._headers(),
            )
            if data:
                break
        if not data:
            logger.debug(
                "semantic_scholar: no %s for %s (tried %s)",
                endpoint,
                doc.doc_id,
                ", ".join(candidates),
            )
            return []

        hits = []
        for rank, edge in enumerate(data.get("data") or []):
            paper = edge.get(payload_key) or {}
            hit = self._to_hit(
                paper,
                self.provenance_for(
                    plan,
                    rank,
                    via=f"{direction}_snowball",
                    seed_doc_id=doc.doc_id,
                ),
            )
            if hit:
                hits.append(hit)
        return hits

    @staticmethod
    def _candidate_ids(doc: Document) -> list[str]:
        """Identifiers to try, most reliable first.

        An arXiv DOI is deprioritized below the bare arXiv id: Semantic
        Scholar indexes the paper under the latter but not the former.
        """
        is_arxiv_doi = bool(doc.doi and doc.doi.lower().startswith("10.48550/"))
        candidates = []
        if doc.s2_paper_id:
            candidates.append(doc.s2_paper_id)
        if doc.doi and not is_arxiv_doi:
            candidates.append(f"DOI:{doc.doi}")
        if doc.arxiv_id:
            candidates.append(f"ARXIV:{doc.arxiv_id}")
        if is_arxiv_doi and doc.doi:
            candidates.append(f"DOI:{doc.doi}")
        if doc.pubmed_id:
            candidates.append(f"PMID:{doc.pubmed_id}")
        return candidates

    @staticmethod
    def _resolve_id(doc: Document) -> str | None:
        """The single best identifier, or None. See :meth:`_candidate_ids`."""
        candidates = SemanticScholarSource._candidate_ids(doc)
        return candidates[0] if candidates else None

    # -- normalization -----------------------------------------------------

    def _to_hit(self, paper: dict, provenance) -> RawHit | None:
        title = (paper.get("title") or "").strip()
        if not title:
            return None

        external = paper.get("externalIds") or {}
        pub_types = paper.get("publicationTypes") or []
        oa_pdf = paper.get("openAccessPdf") or {}
        venue = (paper.get("venue") or "").strip() or None

        arxiv_id = external.get("ArXiv")
        # A preprint server named as the venue is not peer review, so testing
        # "has a venue" alone would mark every arXiv paper as published.
        is_preprint = bool(arxiv_id) and (
            not venue or venue.lower().rstrip(".") in PREPRINT_VENUES
        )
        return RawHit(
            source=self.name,
            source_id=paper.get("paperId") or title[:60],
            title=title,
            abstract=paper.get("abstract"),
            authors=[a.get("name", "") for a in (paper.get("authors") or []) if a.get("name")],
            year=paper.get("year"),
            venue=venue,
            doi=external.get("DOI"),
            arxiv_id=arxiv_id,
            pubmed_id=external.get("PubMed"),
            s2_paper_id=paper.get("paperId"),
            url=paper.get("url"),
            pdf_url=oa_pdf.get("url") or (
                f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None
            ),
            citation_count=paper.get("citationCount"),
            reference_count=paper.get("referenceCount"),
            publication_types=[t for t in pub_types if t],
            is_preprint=is_preprint,
            provenance=provenance,
        )
