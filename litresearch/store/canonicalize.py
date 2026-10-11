"""Document deduplication and canonicalization.

The same paper routinely arrives from four sources at once -- arXiv as a
preprint, Crossref with the published DOI, Semantic Scholar with citation
counts, and a web hit pointing at its landing page. Collapsing these into one
canonical document *before* the expensive reading stage is what keeps the
budget honest, and the per-source provenance we keep while doing it is what
lets retrieval coverage be evaluated afterwards.

Matching hierarchy, strongest first:

    1. DOI
    2. arXiv / PubMed / OpenAlex / Semantic Scholar persistent id
    3. Normalized title + first-author surname
    4. Title similarity + corroborating signal (year, authors)

Tier 4 is the only fuzzy one and is deliberately conservative: a false merge
silently destroys evidence, which is far worse than carrying a duplicate.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from ..state import Document, RawHit, fold_text, normalize_title

logger = logging.getLogger(__name__)

TITLE_SIMILARITY_THRESHOLD = 0.93
"""Ratio above which two titles are considered the same work.

Set high on purpose. Papers in a series ("... Part II", "<Method> v2") have
titles in the 0.85-0.92 range and must not be merged.
"""

STOPWORDS = frozenset(
    """a an and are as at be by for from has in is of on or that the to with
    using via towards toward how what why when can do does""".split()
)


def surname(author: str) -> str:
    """Best-effort surname extraction for matching purposes."""
    cleaned = re.sub(r"[^A-Za-z\s\-']", " ", fold_text(author)).strip()
    if not cleaned:
        return ""
    parts = cleaned.split()
    # Handles both "Jane Doe" and "Doe, Jane" once the comma is stripped.
    return parts[-1].lower() if len(parts) > 1 else parts[0].lower()


def significant_tokens(title: str) -> set[str]:
    return {
        token
        for token in normalize_title(title).split()
        if len(token) > 2 and token not in STOPWORDS
    }


ROMAN_RE = re.compile(r"\b(?:ii|iii|iv|vi|vii|viii|ix|xi|xii)\b")


def numeric_signature(title: str) -> tuple[str, ...]:
    """Numbers in a title, which almost always distinguish separate works.

    "Llama 2" vs "Llama 3", "GPT-3" vs "GPT-4", "Part II" vs "Part III" differ
    by one or two characters and score far above any similarity threshold, yet
    are different papers. Numerals in titles are load-bearing, so a mismatch
    vetoes a fuzzy merge outright.
    """
    normalized = normalize_title(title)
    return tuple(sorted(re.findall(r"\d+", normalized) + ROMAN_RE.findall(normalized)))


def titles_match(left: str, right: str, *, allow_truncation: bool = True) -> bool:
    """Do two titles plausibly name the same work?

    Shared by the canonicalizer and by source adapters that need to *verify*
    an identifier lookup -- OpenAlex, for instance, has some arXiv DOIs
    attached to the wrong record, so a DOI hit has to be checked against the
    title before it is trusted.

    Tolerates case, punctuation and truncation, but a numeral mismatch is an
    outright veto.
    """
    a, b = normalize_title(left), normalize_title(right)
    if not a or not b:
        return False
    if a == b:
        return True
    if numeric_signature(left) != numeric_signature(right):
        return False
    if SequenceMatcher(None, a, b).ratio() >= TITLE_SIMILARITY_THRESHOLD:
        return True

    if not allow_truncation:
        # Caller wants only strong evidence -- see OpenAlexSource._verify,
        # which treats a truncation match as needing author corroboration.
        return False

    # One side is a truncation of the other -- common when a record carries a
    # short-form title or drops a subtitle. Three distinctive words is the
    # floor: no length threshold separates a specific stub ("llm planner")
    # from a generic one ("deep learning"), both two words, and accepting
    # two-word prefixes would match "Deep Learning" to any paper about it.
    # Callers with author lists can admit shorter prefixes themselves via
    # title_is_prefix(); see OpenAlexSource._verify.
    shorter, longer = sorted((a, b), key=len)
    if longer.startswith(shorter) and len(significant_tokens(shorter)) >= 3:
        return True

    # Otherwise require the shorter title's distinctive words to be wholly
    # contained in the longer one, which catches "Method: A Study Of X" vs
    # "Method" while still rejecting unrelated papers that share a topic word.
    short_tokens = significant_tokens(shorter)
    long_tokens = significant_tokens(longer)
    if len(short_tokens) >= 4 and short_tokens <= long_tokens:
        return True
    return False


def title_is_prefix(left: str, right: str) -> bool:
    """Is one normalized title a leading substring of the other?

    Weaker than :func:`titles_match` and never sufficient alone -- use it
    only with an independent corroborating signal such as a shared author.
    """
    a, b = normalize_title(left), normalize_title(right)
    if not a or not b:
        return False
    if numeric_signature(left) != numeric_signature(right):
        return False
    shorter, longer = sorted((a, b), key=len)
    return bool(shorter) and longer.startswith(shorter)


def title_author_key(title: str, authors: list[str]) -> str | None:
    """Tier-3 key: normalized title plus first-author surname.

    Returns ``None`` when no author is known. A title-only key with an empty
    author slot collides any two works sharing a title -- including a 2015
    survey and a 2024 one -- so author-less records are matched through the
    corroborated title path instead, where the year still gets a vote.
    """
    first = surname(authors[0]) if authors else ""
    if not first:
        return None
    return f"ta:{normalize_title(title)}|{first}"


@dataclass
class CanonicalizationResult:
    documents: dict[str, Document]
    """Only the documents created or modified by this ingest."""

    identity_index: dict[str, str]
    """New identity-key -> doc_id entries to merge into state."""

    new_doc_ids: list[str] = field(default_factory=list)
    merged_count: int = 0
    skipped_count: int = 0

    @property
    def new_count(self) -> int:
        return len(self.new_doc_ids)


class Canonicalizer:
    """Folds raw hits into a canonical corpus.

    Constructed around the current corpus, mutated by :meth:`ingest`, then
    discarded; the graph state remains the durable store.
    """

    def __init__(
        self,
        documents: dict[str, Document],
        identity_index: dict[str, str] | None = None,
    ) -> None:
        self._documents = {doc_id: doc for doc_id, doc in documents.items()}
        self._identity: dict[str, str] = dict(identity_index or {})
        self._touched: dict[str, Document] = {}
        self._new_identity: dict[str, str] = {}
        self._new_ids: list[str] = []
        self._merged = 0
        self._skipped = 0

        # Rebuild the derived indexes; cheap relative to any network call, and
        # it keeps the persisted state small (only the identity map is saved).
        self._title_index: dict[str, str] = {}
        self._token_index: dict[str, set[str]] = {}
        for doc in self._documents.values():
            self._index_document(doc)

        self._next_ordinal = len(self._documents) + 1

    # -- public API --------------------------------------------------------

    def ingest(self, hits: list[RawHit]) -> CanonicalizationResult:
        """Merge ``hits`` into the corpus, returning only what changed.

        The result describes only *this* call. The corpus and the identity
        index accumulate across calls; the change report does not.
        """
        self._touched = {}
        self._new_identity = {}
        self._new_ids = []
        self._merged = 0
        self._skipped = 0

        for hit in hits:
            if not hit.title or len(hit.title.strip()) < 8:
                # Junk result; a title this short is never a real paper.
                self._skipped += 1
                continue
            self._ingest_one(hit)

        return CanonicalizationResult(
            documents=self._touched,
            identity_index=self._new_identity,
            new_doc_ids=self._new_ids,
            merged_count=self._merged,
            skipped_count=self._skipped,
        )

    @property
    def corpus(self) -> dict[str, Document]:
        return self._documents

    # -- internals ---------------------------------------------------------

    def _ingest_one(self, hit: RawHit) -> None:
        match_id = self._find_match(hit)

        if match_id is None:
            doc_id = self._mint_id()
            doc = Document.from_hit(hit, doc_id)
            self._documents[doc_id] = doc
            self._touched[doc_id] = doc
            self._new_ids.append(doc_id)
            self._index_document(doc, register_new=True)
            return

        existing = self._documents[match_id]
        incoming = Document.from_hit(hit, match_id)
        merged = existing.merge(incoming)
        self._documents[match_id] = merged
        self._touched[match_id] = merged
        self._merged += 1
        # The merge may have supplied identifiers the existing record lacked,
        # so re-index: that is how a later arrival teaches us the DOI of a
        # paper we first saw as a bare preprint.
        self._index_document(merged, register_new=True)

    def _find_match(self, hit: RawHit) -> str | None:
        # Tiers 1 and 2: persistent identifiers.
        for key in hit.identity_keys():
            if (doc_id := self._identity.get(key)) is not None:
                return doc_id

        # Tier 3: exact normalized title + first author.
        ta_key = title_author_key(hit.title, hit.authors)
        if ta_key is not None and (doc_id := self._identity.get(ta_key)) is not None:
            return doc_id

        # Exact title with nothing contradicting it. Web hits frequently carry
        # no author list at all, so requiring authors here would duplicate
        # them; an exact title is strong enough that only a conflicting year or
        # a disjoint author set vetoes the match.
        norm = normalize_title(hit.title)
        if (doc_id := self._title_index.get(norm)) is not None:
            if self._corroborated(hit, self._documents[doc_id], year_tolerance=2):
                return doc_id

        # Tier 4: fuzzy title match, blocked on shared significant tokens so
        # this stays linear in candidates rather than in corpus size.
        return self._fuzzy_match(hit, norm)

    def _fuzzy_match(self, hit: RawHit, normalized: str) -> str | None:
        tokens = significant_tokens(hit.title)
        if len(tokens) < 3:
            # Too little signal to risk a fuzzy merge.
            return None

        candidate_ids: set[str] = set()
        for token in tokens:
            candidate_ids |= self._token_index.get(token, set())

        best_id: str | None = None
        best_ratio = 0.0
        for doc_id in candidate_ids:
            candidate = self._documents[doc_id]
            cand_tokens = significant_tokens(candidate.title)
            overlap = len(tokens & cand_tokens) / max(len(tokens | cand_tokens), 1)
            if overlap < 0.6:
                continue
            ratio = SequenceMatcher(
                None, normalized, normalize_title(candidate.title)
            ).ratio()
            if ratio > best_ratio:
                best_ratio, best_id = ratio, doc_id

        if best_id is None or best_ratio < TITLE_SIMILARITY_THRESHOLD:
            return None

        candidate = self._documents[best_id]

        # Numerals veto: a fuzzy title match that disagrees on its numbers is
        # a different work, however high the similarity ratio.
        if numeric_signature(hit.title) != numeric_signature(candidate.title):
            return None

        # Require a second, independent signal before merging on a fuzzy
        # title: a compatible year or an overlapping author set.
        if not self._corroborated(hit, candidate, year_tolerance=1, require_signal=True):
            return None

        logger.debug(
            "fuzzy-merged %r into %s (%r) at ratio %.3f",
            hit.title[:60],
            best_id,
            candidate.title[:60],
            best_ratio,
        )
        return best_id

    def _corroborated(
        self,
        hit: RawHit,
        candidate: Document,
        *,
        year_tolerance: int,
        require_signal: bool = False,
    ) -> bool:
        """Is there a second signal consistent with these being one work?

        ``year_tolerance`` is deliberately loose: a preprint and its published
        version routinely carry different years. ``require_signal`` demands
        positive corroboration rather than mere absence of conflict, which is
        what a fuzzy title match needs and an exact one does not.
        """
        if hit.year and candidate.year and abs(hit.year - candidate.year) > year_tolerance:
            return False
        if hit.authors and candidate.authors:
            return self._authors_compatible(hit.authors, candidate.authors)
        if require_signal:
            # Title similarity is all we have; insist on a year on both sides.
            return bool(hit.year and candidate.year)
        return True

    @staticmethod
    def _authors_compatible(left: list[str], right: list[str]) -> bool:
        """Do two author lists plausibly describe the same paper?

        Compares surname sets. Sources disagree constantly on initials,
        ordering and completeness, so any meaningful overlap counts; an empty
        intersection is treated as evidence against a match.
        """
        left_names = {s for a in left if (s := surname(a))}
        right_names = {s for a in right if (s := surname(a))}
        if not left_names or not right_names:
            return True
        overlap = left_names & right_names
        return len(overlap) / min(len(left_names), len(right_names)) >= 0.5

    def _index_document(self, doc: Document, register_new: bool = False) -> None:
        keys = []
        if doc.doi:
            keys.append(f"doi:{doc.doi}")
        if doc.arxiv_id:
            keys.append(f"arxiv:{doc.arxiv_id}")
        if doc.pubmed_id:
            keys.append(f"pmid:{doc.pubmed_id}")
        if doc.openalex_id:
            keys.append(f"openalex:{doc.openalex_id.lower()}")
        if doc.s2_paper_id:
            keys.append(f"s2:{doc.s2_paper_id.lower()}")
        if (ta_key := title_author_key(doc.title, doc.authors)) is not None:
            keys.append(ta_key)

        for key in keys:
            if self._identity.get(key) != doc.doc_id:
                self._identity[key] = doc.doc_id
                if register_new:
                    self._new_identity[key] = doc.doc_id

        self._title_index.setdefault(normalize_title(doc.title), doc.doc_id)
        for token in significant_tokens(doc.title):
            self._token_index.setdefault(token, set()).add(doc.doc_id)

    def _mint_id(self) -> str:
        doc_id = f"doc_{self._next_ordinal:04d}"
        while doc_id in self._documents:
            self._next_ordinal += 1
            doc_id = f"doc_{self._next_ordinal:04d}"
        self._next_ordinal += 1
        return doc_id
