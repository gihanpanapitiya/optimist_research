"""Crossref source adapter.

Crossref is the DOI registration authority, so its value here is less about
discovery than about *canonical bibliographic records*: it reliably supplies
the DOI, venue and publication date that preprint-centric sources often lack.
Abstracts are present only sometimes, and arrive as JATS XML.

It also serves as a **backward snowball fallback**. Publishers deposit
reference lists with Crossref, and those lists are frequently populated for
records whose OpenAlex ``referenced_works`` is empty -- a published IEEE paper
measured 44 Crossref references against 37 in OpenAlex, while two NeurIPS
records had 44 and 91 in Crossref against zero in OpenAlex. Crossref supplies
the edge list; OpenAlex hydrates the DOIs into readable records.

The limit is inherent: this only works for Crossref-registered DOIs, so
published work benefits and arXiv-only preprints (whose DOIs are registered
with DataCite) do not.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from ..state import Document, QueryPlan, RawHit, normalize_doi
from .base import SearchSource
from .hydrate import hydrate_by_doi

logger = logging.getLogger(__name__)

BASE = "https://api.crossref.org/works"

SELECT_FIELDS = ",".join(
    [
        "DOI",
        "title",
        "abstract",
        "author",
        "issued",
        "container-title",
        "is-referenced-by-count",
        "references-count",
        "type",
        "URL",
        "link",
    ]
)
"""Fields requested via Crossref's `select`.

Every name here must be valid for the /works route: Crossref rejects the
*whole request* with a 400 if any one is not, so an invalid field silently
costs the entire source. `subtype` is notably not selectable here.
"""

NON_ARTICLE_TYPES = {
    "book",
    "book-chapter",
    "book-part",
    "component",
    "dataset",
    "dissertation",
    "report",
    "standard",
    "peer-review",
}


def strip_jats(text: str | None) -> str | None:
    """Flatten a JATS-XML abstract into plain text."""
    if not text:
        return None
    cleaned = re.sub(r"<[^>]+>", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Crossref abstracts frequently begin with a literal "Abstract" heading.
    cleaned = re.sub(r"^abstract[:\s]*", "", cleaned, flags=re.IGNORECASE)
    return cleaned or None


# Strings that name a venue rather than a paper. Publishers routinely put the
# cited paper's title in `journal-title` and the venue in `volume-title`, or
# the reverse, so a title candidate has to be sanity-checked before use.
VENUE_HINTS = (
    "advances in neural information processing",
    "proceedings of",
    "in proceedings",
    "conference on",
    "international conference",
    "international joint conference",
    "journal of",
    "transactions on",
    "arxiv preprint",
    "annual meeting",
    "workshop on",
    "symposium on",
    "lecture notes",
)


@dataclass
class ReferenceStub:
    """One entry from a Crossref reference list."""

    doi: str | None
    title: str | None
    author: str | None
    year: int | None
    venue: str | None

    def is_usable(self) -> bool:
        """Can anything downstream act on this?

        A DOI is enough -- hydration will supply the rest. Failing that we
        need a title, since screening and reading have nothing to work with
        otherwise.
        """
        return bool(self.doi or (self.title and len(self.title) >= 12))


class CrossrefSource(SearchSource):
    name = "crossref"
    min_request_interval = 0.12
    max_concurrency = 4
    supports_backward = True

    async def search(self, plan: QueryPlan) -> list[RawHit]:
        params = {
            "query.bibliographic": plan.query,
            "rows": min(plan.max_results, 50),
            "select": SELECT_FIELDS,
            "sort": "relevance",
            "order": "desc",
        }
        if self.settings.contact_email:
            params["mailto"] = self.settings.contact_email

        filters = []
        if plan.year_from:
            filters.append(f"from-pub-date:{plan.year_from}-01-01")
        if plan.year_to:
            filters.append(f"until-pub-date:{plan.year_to}-12-31")
        if filters:
            params["filter"] = ",".join(filters)

        data = await self.get_json(BASE, params=params)
        if data is None:
            # A rejected `select` fails the whole request. Retry without it so
            # a field-name problem costs us richer metadata, not the source.
            logger.info("crossref: retrying without field selection")
            params.pop("select", None)
            data = await self.get_json(BASE, params=params)
        if not data:
            return []

        items = (data.get("message") or {}).get("items") or []
        hits = []
        for rank, item in enumerate(items):
            hit = self._to_hit(item, plan, rank)
            if hit:
                hits.append(hit)
        return hits

    # -- backward snowballing ---------------------------------------------

    async def references(self, doc: Document, limit: int = 40) -> list[RawHit]:
        """Fetch the works ``doc`` cites, from its deposited reference list."""
        doi = normalize_doi(doc.doi)
        if not doi:
            return []
        if doi.startswith("10.48550/"):
            # An arXiv DOI is registered with DataCite, not Crossref, and
            # carries no reference list. Skip the request entirely.
            logger.debug("crossref: %s has an arXiv DOI; no reference list", doc.doc_id)
            return []

        # The single-work route rejects `select` outright ("This route does
        # not support select"), so the whole record comes back; the reference
        # list is the only part we read.
        params = {}
        if self.settings.contact_email:
            params["mailto"] = self.settings.contact_email
        data = await self.get_json(f"{BASE}/{doi}", params=params)
        if not data:
            return []

        entries = (data.get("message") or {}).get("reference") or []
        stubs = [s for s in (self._parse_reference(e) for e in entries) if s.is_usable()]
        if not stubs:
            logger.debug("crossref: no usable references deposited for %s", doc.doc_id)
            return []
        stubs = stubs[:limit]

        plan = self.snowball_plan(doc, "backward")

        def provenance(_key: str, rank: int = 0):
            return self.provenance_for(
                plan, rank, via="backward_snowball", seed_doc_id=doc.doc_id
            )

        # Hydrate the DOI-bearing entries into full records; the rest stay as
        # sparse candidates that screening can still judge on their titles.
        hydrated = await hydrate_by_doi(
            [s.doi for s in stubs if s.doi],
            lambda key: provenance(key),
            source_name=self.name,
        )

        hits: list[RawHit] = []
        for rank, stub in enumerate(stubs):
            key = normalize_doi(stub.doi) if stub.doi else None
            if key and key in hydrated:
                hit = hydrated[key].model_copy(deep=True)
                hit.provenance = provenance(key, rank)
                hits.append(hit)
            elif stub.title:
                hits.append(
                    RawHit(
                        source=self.name,
                        source_id=stub.doi or stub.title[:60],
                        title=stub.title,
                        authors=[stub.author] if stub.author else [],
                        year=stub.year,
                        venue=stub.venue,
                        doi=stub.doi,
                        url=f"https://doi.org/{stub.doi}" if stub.doi else None,
                        provenance=provenance(key or stub.title, rank),
                    )
                )
            # A DOI that hydration could not resolve and that carries no title
            # is dropped: there is nothing for screening to judge.

        logger.debug(
            "crossref: %s -> %d references (%d hydrated, %d sparse)",
            doc.doc_id,
            len(hits),
            sum(1 for h in hits if h.abstract),
            sum(1 for h in hits if not h.abstract),
        )
        return hits

    @staticmethod
    def _parse_reference(entry: dict) -> ReferenceStub:
        """Pull a usable record out of one reference entry.

        Crossref reference lists are publisher-supplied and inconsistent: the
        cited title turns up in `article-title`, `journal-title` or
        `volume-title` depending on the depositor, and the same fields carry
        venue names. Precedence plus a venue-name check sorts most of it out;
        anything still ambiguous falls back to the DOI, which is authoritative.
        """
        article = (entry.get("article-title") or "").strip()
        journal = (entry.get("journal-title") or "").strip()
        volume = (entry.get("volume-title") or "").strip()

        def looks_like_venue(text: str) -> bool:
            lowered = text.lower()
            return any(hint in lowered for hint in VENUE_HINTS)

        if article:
            title = article
            venue = volume or journal or None
        else:
            # No article-title: the depositor may have put the paper's title
            # in one of the other fields. Take a long, non-venue-looking
            # candidate; otherwise treat it as the venue.
            candidates = [t for t in (journal, volume) if t]
            title = next(
                (
                    t
                    for t in candidates
                    if not looks_like_venue(t) and len(t.split()) >= 4
                ),
                None,
            )
            venue = next((t for t in candidates if t != title), None)

        year = None
        raw_year = str(entry.get("year") or "")
        if match := re.search(r"(19|20)\d{2}", raw_year):
            year = int(match.group(0))

        return ReferenceStub(
            doi=(entry.get("DOI") or "").strip() or None,
            title=title,
            author=(entry.get("author") or "").strip() or None,
            year=year,
            venue=venue,
        )

    def _to_hit(self, item: dict, plan: QueryPlan, rank: int) -> RawHit | None:
        titles = item.get("title") or []
        title = (titles[0] if titles else "").strip()
        if not title:
            return None
        title = re.sub(r"\s+", " ", title)

        authors = []
        for author in item.get("author") or []:
            name = " ".join(
                part for part in [author.get("given"), author.get("family")] if part
            ).strip()
            if name:
                authors.append(name)

        year = None
        date_parts = (item.get("issued") or {}).get("date-parts") or []
        if date_parts and date_parts[0] and date_parts[0][0]:
            try:
                year = int(date_parts[0][0])
            except (TypeError, ValueError):
                year = None

        containers = item.get("container-title") or []
        venue = (containers[0] if containers else "").strip() or None

        pdf_url = None
        for link in item.get("link") or []:
            if link.get("content-type") == "application/pdf":
                pdf_url = link.get("URL")
                break

        work_type = (item.get("type") or "").lower()
        doi = item.get("DOI")

        return RawHit(
            source=self.name,
            source_id=doi or title[:60],
            title=title,
            abstract=strip_jats(item.get("abstract")),
            authors=authors,
            year=year,
            venue=venue,
            doi=doi,
            url=item.get("URL") or (f"https://doi.org/{doi}" if doi else None),
            pdf_url=pdf_url,
            citation_count=item.get("is-referenced-by-count"),
            reference_count=item.get("references-count"),
            publication_types=[work_type] if work_type else [],
            is_preprint=work_type == "posted-content",
            provenance=self.provenance_for(plan, rank),
        )
