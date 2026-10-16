"""OpenAlex source adapter.

Free, no key, very broad coverage, and good DOI hygiene -- which makes it the
most useful backstop for canonicalization. Also exposes both citation
directions, so it can snowball when Semantic Scholar is throttled or has no
record of a paper.

Supplying a contact email puts requests in the polite pool with far better
rate limits; do set ``OPTIMISTRESEARCH_CONTACT_EMAIL``.
"""

from __future__ import annotations

import logging
import re

from ..state import Document, QueryPlan, RawHit, normalize_doi
from ..store.canonicalize import surname, title_is_prefix, titles_match
from .base import SearchSource

logger = logging.getLogger(__name__)

BASE = "https://api.openalex.org"

SELECT_FIELDS = ",".join(
    [
        "id",
        "doi",
        "display_name",
        "publication_year",
        "type",
        "cited_by_count",
        "referenced_works_count",
        "authorships",
        "primary_location",
        "open_access",
        "abstract_inverted_index",
        "ids",
        "referenced_works",
    ]
)

REVIEW_TYPES = {"review", "book", "book-chapter", "editorial", "letter", "paratext"}

VERIFY_FIELDS = "id,display_name,authorships"
"""Minimum fields needed to confirm a looked-up record is the right paper."""


def reconstruct_abstract(inverted_index: dict | None) -> str | None:
    """Rebuild an abstract from OpenAlex's inverted index.

    OpenAlex ships abstracts as ``{token: [positions]}`` for licensing
    reasons; placing each token at its positions recovers the original text.
    """
    if not inverted_index:
        return None
    positions: list[tuple[int, str]] = []
    for token, indices in inverted_index.items():
        for index in indices or []:
            positions.append((index, token))
    if not positions:
        return None
    positions.sort(key=lambda pair: pair[0])
    return " ".join(token for _, token in positions)


class OpenAlexSource(SearchSource):
    name = "openalex"
    min_request_interval = 0.12
    max_concurrency = 4
    supports_backward = True
    supports_forward = True

    def _common_params(self) -> dict:
        params: dict = {"select": SELECT_FIELDS}
        if self.settings.contact_email:
            params["mailto"] = self.settings.contact_email
        return params

    async def search(self, plan: QueryPlan) -> list[RawHit]:
        params = self._common_params()
        params.update(
            {
                "search": plan.query,
                "per-page": min(plan.max_results, 50),
            }
        )
        filters = []
        if plan.year_from:
            filters.append(f"from_publication_date:{plan.year_from}-01-01")
        if plan.year_to:
            filters.append(f"to_publication_date:{plan.year_to}-12-31")
        if filters:
            params["filter"] = ",".join(filters)

        data = await self.get_json(f"{BASE}/works", params=params)
        if not data:
            return []

        hits = []
        for rank, work in enumerate(data.get("results") or []):
            hit = self._to_hit(work, self.provenance_for(plan, rank))
            if hit:
                hits.append(hit)
        return hits

    # -- snowballing -------------------------------------------------------

    async def references(self, doc: Document, limit: int = 40) -> list[RawHit]:
        """Fetch the works ``doc`` cites.

        OpenAlex returns referenced works as bare IDs, so this needs a second
        call. Those IDs are batched into one OR filter rather than fetched
        individually.
        """
        work_id = await self._resolve_work_id(doc)
        if not work_id:
            return []

        detail = await self.get_json(
            f"{BASE}/works/{work_id}",
            params={"select": "referenced_works", **self._polite()},
        )
        if not detail:
            return []
        ref_ids = [
            r.rsplit("/", 1)[-1] for r in (detail.get("referenced_works") or [])[:limit]
        ]
        if not ref_ids:
            return []

        plan = self.snowball_plan(doc, "backward")
        hits: list[RawHit] = []
        # OpenAlex caps an OR filter at 50 values per request.
        for start in range(0, len(ref_ids), 50):
            chunk = ref_ids[start : start + 50]
            params = self._common_params()
            params.update({"filter": f"openalex_id:{'|'.join(chunk)}", "per-page": 50})
            data = await self.get_json(f"{BASE}/works", params=params)
            if not data:
                continue
            for rank, work in enumerate(data.get("results") or []):
                hit = self._to_hit(
                    work,
                    self.provenance_for(
                        plan, start + rank, via="backward_snowball", seed_doc_id=doc.doc_id
                    ),
                )
                if hit:
                    hits.append(hit)
        return hits

    async def citations(self, doc: Document, limit: int = 40) -> list[RawHit]:
        """Fetch works citing ``doc`` via the cites filter."""
        work_id = await self._resolve_work_id(doc)
        if not work_id:
            return []

        params = self._common_params()
        params.update(
            {
                "filter": f"cites:{work_id}",
                "per-page": min(limit, 50),
                # Most-cited citing works first: the influential follow-ups.
                "sort": "cited_by_count:desc",
            }
        )
        data = await self.get_json(f"{BASE}/works", params=params)
        if not data:
            return []

        plan = self.snowball_plan(doc, "forward")
        hits = []
        for rank, work in enumerate(data.get("results") or []):
            hit = self._to_hit(
                work,
                self.provenance_for(
                    plan, rank, via="forward_snowball", seed_doc_id=doc.doc_id
                ),
            )
            if hit:
                hits.append(hit)
        return hits

    def _polite(self) -> dict:
        return {"mailto": self.settings.contact_email} if self.settings.contact_email else {}

    async def _resolve_work_id(self, doc: Document) -> str | None:
        """Get a bare OpenAlex work id for ``doc``, verified against its title.

        Verification is not paranoia. OpenAlex has some arXiv DOIs attached
        to the wrong record -- ``10.48550/arXiv.2201.11903`` (Chain-of-Thought
        Prompting) resolves to an unrelated 2022 paper, for example. Trusting
        that would seed snowballing from the wrong work and pull its
        references into the corpus as though they were the seed's, which is
        worse than not snowballing at all.
        """
        # OpenAlex's own identifier *is* its identity; nothing to verify.
        if doc.openalex_id:
            return doc.openalex_id.rsplit("/", 1)[-1]

        if doc.doi:
            record = await self.get_json(
                f"{BASE}/works/doi:{doc.doi}",
                params={"select": VERIFY_FIELDS, **self._polite()},
            )
            work_id = self._verify(doc, record, f"doi:{doc.doi}")
            if work_id:
                return work_id

        return await self._resolve_by_title(doc)

    async def _resolve_by_title(self, doc: Document) -> str | None:
        """Last-resort lookup by title search, verified the same way.

        Also the only route for a document with an arXiv id but no DOI, which
        the DOI lookup cannot reach at all.
        """
        # OpenAlex's title.search filter returns nothing for queries carrying
        # colons or commas, so strip punctuation before asking.
        clean = re.sub(r"[^A-Za-z0-9 ]+", " ", doc.title)
        clean = re.sub(r"\s+", " ", clean).strip()
        if len(clean) < 12:
            return None

        params = {
            "filter": f"title.search:{clean}",
            "select": f"{VERIFY_FIELDS},cited_by_count",
            "per-page": 5,
            **self._polite(),
        }
        data = await self.get_json(f"{BASE}/works", params=params)
        if not data:
            return None

        # OpenAlex sometimes holds several records for one paper; prefer the
        # most-cited verified match, which is the better-populated record.
        matches = [
            record
            for record in (data.get("results") or [])
            if self._verify(doc, record, "title search", quiet=True)
        ]
        if not matches:
            logger.info(
                "openalex: no verified title match for %s (%r)", doc.doc_id, doc.title[:60]
            )
            return None
        best = max(matches, key=lambda r: r.get("cited_by_count") or 0)
        return (best.get("id") or "").rsplit("/", 1)[-1] or None

    def _verify(
        self, doc: Document, record: dict | None, how: str, quiet: bool = False
    ) -> str | None:
        """Return the record's id only if it really is the same work."""
        if not record or not record.get("id"):
            return None
        title = record.get("display_name") or ""

        record_authors = [
            a["author"]["display_name"]
            for a in (record.get("authorships") or [])
            if a.get("author", {}).get("display_name")
        ]
        ours = {s for a in doc.authors if (s := surname(a))}
        theirs = {s for a in record_authors if (s := surname(a))}
        # Tri-state: None when either side lists no authors, so "unknown" is
        # not conflated with "no author in common".
        authors_agree = bool(ours & theirs) if (ours and theirs) else None

        # A full title match is accepted on its own. Author lists are *not*
        # allowed to veto it: sources render names inconsistently (initials
        # only, transliterations, collective names, dash variants), so an
        # empty surname intersection is weaker evidence of a different paper
        # than a matching full title is of the same one.
        if titles_match(doc.title, title, allow_truncation=False):
            return (record.get("id") or "").rsplit("/", 1)[-1] or None

        # Weaker title evidence -- a truncation or a prefix -- is only
        # trustworthy with an independent signal, so here authors must agree.
        if authors_agree and (
            titles_match(doc.title, title) or title_is_prefix(doc.title, title)
        ):
            return (record.get("id") or "").rsplit("/", 1)[-1] or None

        if not quiet:
            logger.warning(
                "openalex: %s for %s resolved to a different work (%r vs %r); "
                "ignoring it",
                how,
                doc.doc_id,
                doc.title[:60],
                title[:60],
            )
        return None

    async def fetch_by_dois(self, dois: list[str]) -> dict[str, dict]:
        """Fetch full OpenAlex records for ``dois``, batched.

        Exposed so another adapter can enrich bare DOIs it discovered -- see
        :mod:`optimistresearch.sources.hydrate`. Returns a mapping keyed by
        normalized DOI; DOIs OpenAlex does not know are simply absent.
        """
        found: dict[str, dict] = {}
        clean = [d for d in (normalize_doi(d) for d in dois) if d]
        # OpenAlex caps an OR filter at 50 values per request.
        for start in range(0, len(clean), 50):
            chunk = clean[start : start + 50]
            params = self._common_params()
            params.update({"filter": f"doi:{'|'.join(chunk)}", "per-page": 50})
            data = await self.get_json(f"{BASE}/works", params=params)
            if not data:
                continue
            for work in data.get("results") or []:
                key = normalize_doi(work.get("doi"))
                if key:
                    found[key] = work
        return found

    # -- normalization -----------------------------------------------------

    def _to_hit(self, work: dict, provenance) -> RawHit | None:
        return work_to_hit(work, provenance, self.name)


def work_to_hit(work: dict, provenance, source_name: str = "openalex") -> RawHit | None:
    """Normalize an OpenAlex work record into a :class:`RawHit`.

    ``source_name`` names the mechanism that *discovered* the document, which
    is not always OpenAlex: when another adapter finds a bare DOI and enriches
    it here, discovery credit belongs to that adapter.
    """
    title = (work.get("display_name") or "").strip()
    if not title:
        return None

    ids = work.get("ids") or {}
    location = work.get("primary_location") or {}
    source_info = location.get("source") or {}
    open_access = work.get("open_access") or {}
    work_type = (work.get("type") or "").lower()

    openalex_id = (work.get("id") or "").rsplit("/", 1)[-1] or None
    pmid = ids.get("pmid")
    if pmid:
        pmid = str(pmid).rsplit("/", 1)[-1]

    venue = (source_info.get("display_name") or "").strip() or None
    is_preprint = work_type == "preprint" or source_info.get("type") == "repository"

    return RawHit(
        source=source_name,
        source_id=openalex_id or title[:60],
        title=title,
        abstract=reconstruct_abstract(work.get("abstract_inverted_index")),
        authors=[
            a["author"]["display_name"]
            for a in (work.get("authorships") or [])
            if a.get("author", {}).get("display_name")
        ],
        year=work.get("publication_year"),
        venue=venue,
        doi=work.get("doi"),
        openalex_id=openalex_id,
        pubmed_id=pmid,
        url=location.get("landing_page_url") or work.get("id"),
        pdf_url=open_access.get("oa_url") or location.get("pdf_url"),
        citation_count=work.get("cited_by_count"),
        reference_count=work.get("referenced_works_count"),
        publication_types=[work_type] if work_type else [],
        is_preprint=is_preprint,
        provenance=provenance,
    )
