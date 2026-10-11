"""arXiv source adapter.

Uses the public Atom API. No key required, but arXiv asks for roughly one
request every three seconds, which is the tightest limit of any source here
and effectively makes arXiv the pacing constraint on a search round.

No citation graph is exposed, so arXiv cannot snowball; Semantic Scholar and
OpenAlex cover that.
"""

from __future__ import annotations

import logging
import re
from xml.etree import ElementTree

from ..state import QueryPlan, RawHit
from .base import SearchSource

logger = logging.getLogger(__name__)

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"

API_URL = "https://export.arxiv.org/api/query"

# Dropped from queries: each costs a conjunctive clause while matching almost
# every paper on arXiv, which is what tips a query over into zero results.
# The second group is the costly one -- research-generic vocabulary. On a real
# generated query, keeping "architectures", "traditional" and "methods" while
# dropping "agent" and "LLMs" produced results about the wrong subject.
STOPWORDS = frozenset(
    """
    a an and are as at be by for from has have in into is it its of on or
    that the their to with within using via towards toward how what why when
    can do does such other

    approach approaches architecture architectures component components
    design designs framework frameworks mechanism mechanisms method methods
    model models paradigm paradigms system systems technique techniques
    analysis evaluation implementation study studies research survey review
    based combining designed emerging novel new recent current existing
    traditional classical general various multiple different key primary
    advances developments improvements challenges issues problems
    """.split()
)


class ArxivSource(SearchSource):
    name = "arxiv"
    # arXiv's stated guidance is one request per three seconds; serialize.
    min_request_interval = 3.0
    max_concurrency = 1

    async def search(self, plan: QueryPlan) -> list[RawHit]:
        """Search arXiv, relaxing an over-constrained query rather than
        returning nothing.

        arXiv ANDs its clauses, so a verbose query collapses to zero results:
        measured on one real generated query, ANDing all eight terms returned
        0, the four most distinctive returned 82, and three returned 514.
        Prompting alone does not prevent this -- the model produced exactly
        such a query on a live run -- so the adapter drops the least
        distinctive clauses and retries.
        """
        clauses = self._clauses(plan)
        if not clauses:
            return []

        # Full query first, then progressively fewer clauses. Each attempt
        # costs a request at arXiv's 3s pacing, so the ladder is short and
        # only descends when the previous attempt found nothing at all.
        attempts = [len(clauses)]
        attempts += [n for n in (4, 3) if n < len(clauses)]

        for width in attempts:
            query = self._assemble(clauses[:width], plan)
            text = await self.get_text(
                API_URL,
                params={
                    "search_query": query,
                    "start": 0,
                    "max_results": min(plan.max_results, 50),
                    # Relevance ordering; recency is weighed against relevance
                    # by the ranking stage rather than overriding it here.
                    "sortBy": "relevance",
                    "sortOrder": "descending",
                },
            )
            if not text:
                return []
            hits = self._parse(text, plan)
            if hits:
                if width < len(clauses):
                    logger.info(
                        "arxiv: %r returned nothing; relaxed to %d clause(s) "
                        "and found %d",
                        plan.query,
                        width,
                        len(hits),
                    )
                return hits

        logger.info("arxiv: no results for %r even when relaxed", plan.query)
        return []

    # -- query construction ------------------------------------------------

    def _clauses(self, plan: QueryPlan) -> list[str]:
        """Field-prefixed clauses for a query, most distinctive first.

        Quoted phrases lead because they carry the most information per
        clause. Loose terms follow in the order written, which for a noun
        phrase puts the topical words before the generic tail; sorting by
        length was measurably worse, since it ranks "traditional" above
        "LLMs". Ordering matters because relaxation truncates this list.
        """
        phrases = re.findall(r'"([^"]+)"', plan.query)
        remainder = re.sub(r'"[^"]*"', " ", plan.query)
        terms = [
            t
            for t in re.split(r"[^A-Za-z0-9\-]+", remainder)
            if len(t) > 2 and t.lower() not in STOPWORDS
        ]

        clauses = [f'all:"{p}"' for p in phrases]
        clauses += [f"all:{t}" for t in terms]
        if not clauses and plan.query.strip():
            clauses = [f'all:"{plan.query.strip()}"']
        return clauses

    @staticmethod
    def _assemble(clauses: list[str], plan: QueryPlan) -> str:
        query = " AND ".join(clauses)
        if plan.year_from:
            year_to = plan.year_to or 2100
            query = (
                f"({query}) AND submittedDate:"
                f"[{plan.year_from}01010000 TO {year_to}12312359]"
            )
        return query

    def _build_query(self, plan: QueryPlan) -> str:
        """The unrelaxed query. Retained for tests and debugging."""
        return self._assemble(self._clauses(plan), plan)

    # -- parsing -----------------------------------------------------------

    def _parse(self, xml_text: str, plan: QueryPlan) -> list[RawHit]:
        try:
            root = ElementTree.fromstring(xml_text)
        except ElementTree.ParseError as exc:
            logger.warning("arxiv: could not parse Atom response (%s)", exc)
            return []

        hits: list[RawHit] = []
        for rank, entry in enumerate(root.findall(f"{ATOM}entry")):
            hit = self._parse_entry(entry, plan, rank)
            if hit is not None:
                hits.append(hit)
        return hits

    def _parse_entry(self, entry, plan: QueryPlan, rank: int) -> RawHit | None:
        title_el = entry.find(f"{ATOM}title")
        if title_el is None or not (title_el.text or "").strip():
            return None
        title = re.sub(r"\s+", " ", title_el.text).strip()

        summary_el = entry.find(f"{ATOM}summary")
        abstract = (
            re.sub(r"\s+", " ", summary_el.text).strip()
            if summary_el is not None and summary_el.text
            else None
        )

        authors = [
            name.text.strip()
            for author in entry.findall(f"{ATOM}author")
            if (name := author.find(f"{ATOM}name")) is not None and name.text
        ]

        id_el = entry.find(f"{ATOM}id")
        abs_url = id_el.text.strip() if id_el is not None and id_el.text else None
        arxiv_id = None
        if abs_url:
            arxiv_id = abs_url.rsplit("/", 1)[-1]

        published_el = entry.find(f"{ATOM}published")
        year = None
        if published_el is not None and published_el.text:
            try:
                year = int(published_el.text[:4])
            except ValueError:
                year = None

        pdf_url = None
        for link in entry.findall(f"{ATOM}link"):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href")
                break
        if pdf_url is None and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

        doi_el = entry.find(f"{ARXIV}doi")
        doi = doi_el.text.strip() if doi_el is not None and doi_el.text else None

        journal_el = entry.find(f"{ARXIV}journal_ref")
        venue = (
            journal_el.text.strip()
            if journal_el is not None and journal_el.text
            else "arXiv preprint"
        )

        category_el = entry.find(f"{ARXIV}primary_category")
        categories = [category_el.get("term")] if category_el is not None else []

        return RawHit(
            source=self.name,
            source_id=arxiv_id or abs_url or title[:60],
            title=title,
            abstract=abstract,
            authors=authors,
            year=year,
            venue=venue,
            doi=doi,
            arxiv_id=arxiv_id,
            url=abs_url,
            pdf_url=pdf_url,
            publication_types=[c for c in categories if c],
            # A journal_ref means it has appeared somewhere peer-reviewed.
            is_preprint=journal_el is None,
            provenance=self.provenance_for(plan, rank),
        )
