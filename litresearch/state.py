"""The research state: the single evolving record of the investigation.

This module is the spine of the system. The methodology's central claim is
that research is an iterative evidence-acquisition process over an explicit
state, so the state model -- not the graph topology -- is what actually
encodes the method:

    main question
        -> subquestions (hierarchical, mutable)
            -> search intents
                -> retrieval queries (per source)
                    -> documents (canonical, deduplicated, globally shared)
                        -> evidence (claim / support / location / source)
                            -> subquestion synthesis
                                -> global synthesis

Reducers at the bottom of this file let parallel graph branches write into
the same state without clobbering each other.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

SubquestionStatus = Literal[
    "unexplored",
    "in_progress",
    "answered",
    "deprioritized",
    "superseded",
]

IntentStatus = Literal["pending", "searched", "exhausted"]

DocumentStatus = Literal["candidate", "selected", "read", "skipped", "failed"]

UNKNOWN_SNOWBALL_DEPTH = 99
"""Placeholder depth for a hit that arrived through the citation graph.

``snowball_depth == 0`` means "reached by direct search". A raw snowball hit
does not know its round -- the snowball node assigns that -- so it must not
default to 0, or re-encountering a depth-2 document in a later round would
merge ``min(2, 0)`` and silently reclassify it as a search result.
"""

EvidenceType = Literal[
    "experimental",
    "theoretical",
    "survey",
    "benchmark",
    "position",
    "anecdotal",
]

SnowballDirection = Literal["backward", "forward"]


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class Provenance(BaseModel):
    """How a document came to be in the corpus.

    Retained per the methodology so retrieval coverage can be evaluated after
    the fact: which sources, intents and subquestions actually found things.
    """

    subquestion_id: str
    intent_id: str
    source: str
    query: str
    rank: int = 0
    via: Literal["search", "backward_snowball", "forward_snowball"] = "search"
    seed_doc_id: str | None = None
    discovered_at: str = Field(default_factory=_utcnow)

    def key(self) -> tuple:
        return (self.subquestion_id, self.intent_id, self.source, self.query, self.via)


# ---------------------------------------------------------------------------
# Search intent and query plan
# ---------------------------------------------------------------------------


class SearchIntent(BaseModel):
    """A retrieval goal: what evidence would close a knowledge gap.

    Distinct from a subquestion, which states what we need to *know*. The
    separation lets the agent rewrite queries without disturbing the research
    objective they serve.
    """

    id: str
    subquestion_id: str
    intent: str
    rationale: str = ""
    desired_evidence: list[str] = Field(default_factory=list)
    preferred_sources: list[str] = Field(default_factory=list)
    status: IntentStatus = "pending"
    created_in_iteration: int = 0
    # Set when the intent came from the critic rather than the initial planner.
    origin: Literal["planner", "critic", "discovery"] = "planner"
    documents_found: int = 0
    relevant_found: int = 0


class QueryPlan(BaseModel):
    """A concrete retrieval query aimed at one source."""

    intent_id: str
    subquestion_id: str
    source: str
    query: str
    year_from: int | None = None
    year_to: int | None = None
    rationale: str = ""
    max_results: int = 20


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


# Sources disagree on which dash they use. OpenAlex writes "Romera‐Paredes"
# with U+2010 HYPHEN; others use an ASCII hyphen. Folding them together is
# what lets the two spellings match.
_DASHES = dict.fromkeys(
    [0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2212, 0xFE63, 0xFF0D], "-"
)


def fold_text(text: str) -> str:
    """Strip accents and normalize dash variants, preserving letters.

    Without this, "Müller" and "Muller" -- or an ASCII hyphen and U+2010 --
    are different strings, which silently breaks both author matching and
    title dedup across sources that encode them differently.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    without_marks = "".join(c for c in decomposed if not unicodedata.combining(c))
    return without_marks.translate(_DASHES)


def normalize_title(title: str) -> str:
    """Aggressively normalize a title for matching.

    Lowercase, fold accents, strip punctuation and collapse whitespace.
    Deliberately lossy: this is a dedup key, not a display value.
    """
    text = fold_text(title).lower().strip()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    doi = doi.strip().lower()
    doi = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:", "", doi)
    return doi or None


def normalize_arxiv_id(value: str | None) -> str | None:
    """Reduce an arXiv identifier to its bare, version-less form."""
    if not value:
        return None
    value = value.strip().lower()
    value = re.sub(r"^(https?://)?arxiv\.org/(abs|pdf)/", "", value)
    value = re.sub(r"^arxiv:", "", value)
    value = re.sub(r"\.pdf$", "", value)
    value = re.sub(r"v\d+$", "", value)
    return value or None


class RawHit(BaseModel):
    """One search result, normalized across sources but not yet canonicalized."""

    source: str
    source_id: str = ""
    title: str
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    pubmed_id: str | None = None
    openalex_id: str | None = None
    s2_paper_id: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    citation_count: int | None = None
    reference_count: int | None = None
    publication_types: list[str] = Field(default_factory=list)
    is_preprint: bool = False
    provenance: Provenance

    def identity_keys(self) -> list[str]:
        """Persistent-identifier keys, strongest first."""
        keys = []
        if doi := normalize_doi(self.doi):
            keys.append(f"doi:{doi}")
        if arx := normalize_arxiv_id(self.arxiv_id):
            keys.append(f"arxiv:{arx}")
        if self.pubmed_id:
            keys.append(f"pmid:{self.pubmed_id.strip()}")
        if self.openalex_id:
            keys.append(f"openalex:{self.openalex_id.strip().rsplit('/', 1)[-1].lower()}")
        if self.s2_paper_id:
            keys.append(f"s2:{self.s2_paper_id.strip().lower()}")
        return keys


class Document(BaseModel):
    """A canonical document in the research corpus.

    One entry per real-world paper, however many sources surfaced it. The
    ``relevance`` map is what makes documents shareable across subquestions:
    a paper found while investigating RQ1 can be read for RQ3 without being
    retrieved again.
    """

    doc_id: str
    title: str
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    pubmed_id: str | None = None
    openalex_id: str | None = None
    s2_paper_id: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    citation_count: int | None = None
    reference_count: int | None = None
    publication_types: list[str] = Field(default_factory=list)
    is_preprint: bool = False

    sources: list[str] = Field(default_factory=list)
    discovered_by: list[Provenance] = Field(default_factory=list)

    relevance: dict[str, float] = Field(default_factory=dict)
    """subquestion_id -> screened relevance in [0, 1]."""

    screening_note: str = ""
    rank_score: float = 0.0
    status: DocumentStatus = "candidate"
    snowball_depth: int = 0
    read_for: list[str] = Field(default_factory=list)
    """Subquestion ids this document has already been read against."""

    fulltext_path: str | None = None
    fulltext_chars: int = 0
    notes: str = ""

    # ---- derived helpers -------------------------------------------------

    @property
    def citation_key(self) -> str:
        first_author = self.authors[0].split()[-1] if self.authors else "Anon"
        return f"{first_author}{self.year or 'n.d.'}"

    @property
    def is_primary_research(self) -> bool:
        """Heuristic: is this a primary research article rather than a survey?"""
        types = {t.lower() for t in self.publication_types}
        if types & {"review", "survey", "editorial", "letter", "news"}:
            return False
        haystack = f"{self.title} {self.abstract or ''}".lower()
        if re.search(r"\b(a |this )?(survey|review|overview|tutorial)\b", self.title.lower()):
            return False
        return bool(
            re.search(
                r"\b(we (propose|present|introduce|evaluate|show)"
                r"|our (method|approach|experiments))\b",
                haystack,
            )
        ) or not types

    def max_relevance(self) -> float:
        return max(self.relevance.values(), default=0.0)

    def best_subquestions(self, threshold: float) -> list[str]:
        """Subquestions this document is relevant enough to be read against."""
        return [sq for sq, score in self.relevance.items() if score >= threshold]

    def reference_label(self) -> str:
        authors = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors += " et al."
        parts = [authors or None, f"({self.year})" if self.year else None, self.title]
        bits = [b for b in parts if b]
        line = ". ".join(bits)
        if self.venue:
            line += f". {self.venue}"
        locator = self.url or (f"https://doi.org/{self.doi}" if self.doi else None)
        if locator:
            line += f". {locator}"
        return line

    def merge(self, other: Document) -> Document:
        """Fold another record of the same paper into this one.

        Field-level resolution prefers the more informative value, so a sparse
        web hit never degrades a rich Semantic Scholar record.
        """
        merged = self.model_copy(deep=True)

        def take_longer(a: str | None, b: str | None) -> str | None:
            if not a:
                return b
            if not b:
                return a
            return a if len(a) >= len(b) else b

        merged.title = take_longer(merged.title, other.title) or merged.title
        merged.abstract = take_longer(merged.abstract, other.abstract)
        merged.authors = merged.authors or other.authors
        merged.year = merged.year or other.year
        merged.venue = merged.venue or other.venue
        merged.doi = merged.doi or other.doi
        merged.arxiv_id = merged.arxiv_id or other.arxiv_id
        merged.pubmed_id = merged.pubmed_id or other.pubmed_id
        merged.openalex_id = merged.openalex_id or other.openalex_id
        merged.s2_paper_id = merged.s2_paper_id or other.s2_paper_id
        merged.url = merged.url or other.url
        merged.pdf_url = merged.pdf_url or other.pdf_url
        merged.citation_count = max(
            merged.citation_count or 0, other.citation_count or 0
        ) or None
        merged.reference_count = max(
            merged.reference_count or 0, other.reference_count or 0
        ) or None
        merged.publication_types = sorted(
            set(merged.publication_types) | set(other.publication_types)
        )
        merged.is_preprint = merged.is_preprint and other.is_preprint

        merged.sources = sorted(set(merged.sources) | set(other.sources))

        seen = {p.key() for p in merged.discovered_by}
        for prov in other.discovered_by:
            if prov.key() not in seen:
                merged.discovered_by.append(prov)
                seen.add(prov.key())

        for sq, score in other.relevance.items():
            merged.relevance[sq] = max(merged.relevance.get(sq, 0.0), score)

        merged.read_for = sorted(set(merged.read_for) | set(other.read_for))
        merged.rank_score = max(merged.rank_score, other.rank_score)
        merged.snowball_depth = min(merged.snowball_depth, other.snowball_depth)
        merged.fulltext_path = merged.fulltext_path or other.fulltext_path
        merged.fulltext_chars = max(merged.fulltext_chars, other.fulltext_chars)
        merged.screening_note = take_longer(merged.screening_note, other.screening_note) or ""

        # Status precedence, weakest first. "read" outranks everything so a
        # completed read never regresses. "failed" outranks "selected" so that
        # a document whose reading failed actually records the failure --
        # otherwise it stays "selected" and is re-queued every iteration.
        precedence = ["candidate", "selected", "skipped", "failed", "read"]
        merged.status = max(
            [merged.status, other.status], key=lambda s: precedence.index(s)
        )
        return merged

    @classmethod
    def from_hit(cls, hit: RawHit, doc_id: str) -> Document:
        return cls(
            doc_id=doc_id,
            title=hit.title.strip(),
            abstract=hit.abstract,
            authors=hit.authors,
            year=hit.year,
            venue=hit.venue,
            doi=normalize_doi(hit.doi),
            arxiv_id=normalize_arxiv_id(hit.arxiv_id),
            pubmed_id=hit.pubmed_id,
            openalex_id=hit.openalex_id,
            s2_paper_id=hit.s2_paper_id,
            url=hit.url,
            pdf_url=hit.pdf_url,
            citation_count=hit.citation_count,
            reference_count=hit.reference_count,
            publication_types=hit.publication_types,
            is_preprint=hit.is_preprint,
            sources=[hit.source],
            discovered_by=[hit.provenance],
            snowball_depth=(
                0 if hit.provenance.via == "search" else UNKNOWN_SNOWBALL_DEPTH
            ),
        )


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


class Evidence(BaseModel):
    """A single extracted claim with its support and provenance.

    The evidence store is what makes the final report traceable and lets the
    critic find weakly supported claims.
    """

    id: str = ""
    claim: str
    support: str
    doc_id: str
    subquestion_id: str
    location: str = ""
    evidence_type: EvidenceType = "theoretical"
    strength: float = 0.5
    """Extractor's confidence that the document really establishes the claim."""
    is_limitation: bool = False
    verbatim_quote: str | None = None

    def model_post_init(self, _context: Any) -> None:
        if not self.id:
            digest = hashlib.sha1(
                f"{self.doc_id}|{self.subquestion_id}|{self.claim}".encode()
            ).hexdigest()[:10]
            object.__setattr__(self, "id", f"ev_{digest}")


class Contradiction(BaseModel):
    """Conflicting findings across the corpus."""

    id: str = ""
    description: str
    subquestion_id: str = ""
    doc_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    status: Literal["open", "investigating", "resolved", "unresolvable"] = "open"
    resolution: str = ""

    def model_post_init(self, _context: Any) -> None:
        if not self.id:
            digest = hashlib.sha1(self.description.encode()).hexdigest()[:10]
            object.__setattr__(self, "id", f"con_{digest}")


class DocumentReading(BaseModel):
    """The structured record a reader agent produces for one document."""

    doc_id: str
    main_contribution: str = ""
    method: str = ""
    experimental_setup: str = ""
    major_findings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    new_subquestions: list[str] = Field(default_factory=list)
    snowball_candidates: list[str] = Field(default_factory=list)
    read_for: list[str] = Field(default_factory=list)
    used_full_text: bool = False


# ---------------------------------------------------------------------------
# Subquestions
# ---------------------------------------------------------------------------


class Subquestion(BaseModel):
    """A specific piece of knowledge required to answer the main question."""

    id: str
    question: str
    rationale: str = ""
    status: SubquestionStatus = "unexplored"
    priority: float = 0.5
    parent_id: str | None = None
    children: list[str] = Field(default_factory=list)

    intent_ids: list[str] = Field(default_factory=list)
    knowledge_gaps: list[str] = Field(default_factory=list)

    coverage: float = 0.0
    synthesis: str = ""
    documents_read: int = 0
    created_in_iteration: int = 0
    origin: Literal["planner", "critic", "discovery"] = "planner"

    @property
    def depth(self) -> int:
        return self.id.count(".")

    def is_active(self) -> bool:
        return self.status in ("unexplored", "in_progress")

    def merge(self, other: Subquestion) -> Subquestion:
        """Later writes win on scalars; collections union.

        Parallel branches only ever write disjoint scalar fields for a given
        subquestion, so last-write-wins on those is safe, while the list
        fields genuinely accumulate from several branches.
        """
        merged = other.model_copy(deep=True)
        merged.intent_ids = list(dict.fromkeys([*self.intent_ids, *other.intent_ids]))
        merged.knowledge_gaps = list(
            dict.fromkeys([*self.knowledge_gaps, *other.knowledge_gaps])
        )
        merged.children = list(dict.fromkeys([*self.children, *other.children]))
        merged.synthesis = other.synthesis or self.synthesis
        merged.documents_read = max(self.documents_read, other.documents_read)
        merged.created_in_iteration = min(
            self.created_in_iteration, other.created_in_iteration
        )
        return merged


# ---------------------------------------------------------------------------
# Critic output
# ---------------------------------------------------------------------------


class PlanEdit(BaseModel):
    """A single mutation to the research-question tree, proposed by the critic."""

    action: Literal["add", "split", "deprioritize", "mark_answered", "rephrase"]
    subquestion_id: str | None = None
    question: str | None = None
    rationale: str = ""
    children: list[str] = Field(default_factory=list)
    priority: float | None = None


class Critique(BaseModel):
    """The research critic's assessment and the actions it recommends."""

    status: Literal["continue", "sufficient"] = "continue"
    overall_coverage: float = 0.0
    reasoning: str = ""

    coverage_assessment: str = ""
    evidence_quality_assessment: str = ""
    source_diversity_assessment: str = ""
    recency_assessment: str = ""
    saturation_assessment: str = ""

    weak_subquestions: list[str] = Field(default_factory=list)
    subquestion_coverage: dict[str, float] = Field(default_factory=dict)
    knowledge_gaps: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    new_contradictions: list[Contradiction] = Field(default_factory=list)

    recommended_intents: list[SearchIntent] = Field(default_factory=list)
    plan_edits: list[PlanEdit] = Field(default_factory=list)
    snowball_requests: list[dict] = Field(default_factory=list)
    """Each: {"doc_id": ..., "direction": "forward"|"backward", "reason": ...}"""


class BudgetUsage(BaseModel):
    """Running tally against the configured budget.

    Every node writes a **delta**, never a running total, and the reducer
    sums. This is what makes fan-in correct: twenty parallel readers each
    write ``documents_read=1`` and the channel advances by twenty. Writing
    totals instead would require each branch to know what the others did.
    """

    search_operations: int = 0
    documents_read: int = 0
    snowball_rounds: int = 0
    research_iterations: int = 0
    documents_discovered: int = 0
    llm_calls: int = 0

    def merge(self, other: BudgetUsage) -> BudgetUsage:
        return BudgetUsage(
            search_operations=self.search_operations + other.search_operations,
            documents_read=self.documents_read + other.documents_read,
            snowball_rounds=self.snowball_rounds + other.snowball_rounds,
            research_iterations=self.research_iterations + other.research_iterations,
            documents_discovered=self.documents_discovered + other.documents_discovered,
            llm_calls=self.llm_calls + other.llm_calls,
        )


class SnowballRound(BaseModel):
    """Yield of one snowball round, used for the saturation criterion."""

    round_index: int
    direction: str
    candidates_examined: int
    newly_relevant: int

    @property
    def yield_rate(self) -> float:
        if self.candidates_examined == 0:
            return 0.0
        return self.newly_relevant / self.candidates_examined


# ---------------------------------------------------------------------------
# Reducers
#
# LangGraph merges concurrent writes to a channel with its reducer. Fan-out
# stages (document readers, per-subquestion synthesizers) all write into
# shared channels, so these have to be associative and tolerate being handed
# a baseline that another branch has also seen.
# ---------------------------------------------------------------------------


def merge_documents(
    current: dict[str, Document] | None, update: dict[str, Document] | None
) -> dict[str, Document]:
    if update is None:
        return {}
    merged = dict(current or {})
    for doc_id, doc in update.items():
        existing = merged.get(doc_id)
        merged[doc_id] = existing.merge(doc) if existing else doc
    return merged


def merge_subquestions(
    current: dict[str, Subquestion] | None, update: dict[str, Subquestion] | None
) -> dict[str, Subquestion]:
    if update is None:
        return {}
    merged = dict(current or {})
    for sq_id, sq in update.items():
        existing = merged.get(sq_id)
        merged[sq_id] = existing.merge(sq) if existing else sq
    return merged


def merge_intents(
    current: dict[str, SearchIntent] | None, update: dict[str, SearchIntent] | None
) -> dict[str, SearchIntent]:
    if update is None:
        return {}
    merged = dict(current or {})
    merged.update(update)
    return merged


def extend_unique_evidence(
    current: list[Evidence] | None, update: list[Evidence] | None
) -> list[Evidence]:
    """Append evidence, suppressing exact duplicates by content hash."""
    if update is None:
        return []
    merged = list(current or [])
    seen = {e.id for e in merged}
    for item in update:
        if item.id not in seen:
            merged.append(item)
            seen.add(item.id)
    return merged


def extend_unique_contradictions(
    current: list[Contradiction] | None, update: list[Contradiction] | None
) -> list[Contradiction]:
    if update is None:
        return []
    merged = list(current or [])
    seen = {c.id for c in merged}
    for item in update:
        if item.id not in seen:
            merged.append(item)
            seen.add(item.id)
    return merged


def extend_or_clear(current: list | None, update: list | None) -> list:
    """Accumulate a list; a ``None`` write clears the channel.

    Used for per-iteration scratch channels (pending hits, reading queue)
    which must be emptied at the start of each research cycle.
    """
    if update is None:
        return []
    return [*(current or []), *update]


def merge_dict_or_clear(current: dict | None, update: dict | None) -> dict:
    if update is None:
        return {}
    return {**(current or {}), **update}


def merge_budget(current: BudgetUsage | None, update: BudgetUsage | None) -> BudgetUsage:
    if update is None:
        return BudgetUsage()
    if current is None:
        return update
    return current.merge(update)


def take_last(current: Any, update: Any) -> Any:
    return update


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class ResearchState(TypedDict, total=False):
    """The complete graph state."""

    # --- the research objective ---
    main_question: str
    scope_notes: str
    plan_rationale: str

    # --- the (mutable) research plan ---
    subquestions: Annotated[dict[str, Subquestion], merge_subquestions]
    intents: Annotated[dict[str, SearchIntent], merge_intents]

    # --- retrieval ---
    #
    # These four are per-iteration work lists with exactly one writer each, so
    # they take replace-semantics: an iteration must start from a fresh query
    # set, not re-run the previous iteration's searches.
    query_plans: Annotated[list[QueryPlan], take_last]
    pending_hits: Annotated[list[RawHit], take_last]
    reading_queue: Annotated[list[dict], take_last]
    """Each: {"doc_id": ..., "subquestion_ids": [...]}"""
    snowball_seeds: Annotated[list[dict], take_last]

    documents: Annotated[dict[str, Document], merge_documents]
    identity_index: Annotated[dict[str, str], merge_dict_or_clear]
    """Persistent-identifier key -> doc_id, for O(1) dedup across iterations."""

    # Accumulates for the whole run: it is the record the saturation
    # criterion is computed from.
    snowball_history: Annotated[list[SnowballRound], extend_or_clear]

    # --- evidence and synthesis ---
    evidence: Annotated[list[Evidence], extend_unique_evidence]
    readings: Annotated[dict[str, DocumentReading], merge_dict_or_clear]
    contradictions: Annotated[list[Contradiction], extend_unique_contradictions]
    global_synthesis: str
    report_outline: Annotated[list[dict], take_last]
    final_report: str

    # --- reflection and control ---
    critique: Critique | None
    critique_history: Annotated[list[Critique], extend_or_clear]
    iteration: int
    budget: Annotated[BudgetUsage, merge_budget]
    terminated_because: str
    log: Annotated[list[str], extend_or_clear]


# ---------------------------------------------------------------------------
# Worker states for Send() fan-out
# ---------------------------------------------------------------------------


class ReaderState(TypedDict):
    """Input to one parallel document-reading branch."""

    main_question: str
    doc: Document
    subquestions: list[Subquestion]
    read_full_text: bool
    iteration: int


class SynthesisState(TypedDict):
    """Input to one parallel subquestion-synthesis branch."""

    main_question: str
    subquestion: Subquestion
    evidence: list[Evidence]
    documents: dict[str, Document]
    iteration: int


def new_state(main_question: str, scope_notes: str = "") -> ResearchState:
    """A fresh research state for a question."""
    return ResearchState(
        main_question=main_question.strip(),
        scope_notes=scope_notes,
        plan_rationale="",
        subquestions={},
        intents={},
        query_plans=[],
        pending_hits=[],
        documents={},
        identity_index={},
        reading_queue=[],
        snowball_history=[],
        snowball_seeds=[],
        evidence=[],
        readings={},
        contradictions=[],
        global_synthesis="",
        report_outline=[],
        final_report="",
        critique=None,
        critique_history=[],
        iteration=0,
        budget=BudgetUsage(),
        terminated_because="",
        log=[],
    )
