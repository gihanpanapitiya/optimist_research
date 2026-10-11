"""Evidence store views.

The evidence list in state is append-only and flat. This module provides the
grouped views the synthesizers and the critic need, plus the diversity and
support metrics the critic uses to judge whether a claim is actually
established or merely asserted by one paper.
"""

from __future__ import annotations

import datetime
from collections import Counter, defaultdict
from dataclasses import dataclass

from ..state import Document, Evidence

STRONG_EVIDENCE_TYPES = frozenset({"experimental", "benchmark"})


@dataclass
class ClaimCluster:
    """One claim and every piece of evidence bearing on it."""

    claim: str
    subquestion_id: str
    evidence: list[Evidence]

    @property
    def doc_ids(self) -> list[str]:
        return sorted({e.doc_id for e in self.evidence})

    @property
    def support_count(self) -> int:
        return len(self.doc_ids)

    @property
    def mean_strength(self) -> float:
        if not self.evidence:
            return 0.0
        return sum(e.strength for e in self.evidence) / len(self.evidence)

    @property
    def has_empirical_support(self) -> bool:
        return any(e.evidence_type in STRONG_EVIDENCE_TYPES for e in self.evidence)

    @property
    def is_weakly_supported(self) -> bool:
        """A claim resting on a single source or on thin confidence.

        This is the signal the critic acts on: it is what distinguishes
        "the literature shows X" from "one paper claimed X".
        """
        return self.support_count < 2 or self.mean_strength < 0.5


def group_by_subquestion(evidence: list[Evidence]) -> dict[str, list[Evidence]]:
    grouped: dict[str, list[Evidence]] = defaultdict(list)
    for item in evidence:
        grouped[item.subquestion_id].append(item)
    return dict(grouped)


def cluster_claims(
    evidence: list[Evidence], subquestion_id: str | None = None
) -> list[ClaimCluster]:
    """Group evidence into claim clusters, keyed on normalized claim text.

    Exact-text clustering is a deliberate floor, not a ceiling: the
    subquestion synthesizer is where semantically equivalent claims get
    reconciled, because that needs judgement rather than string equality.
    """
    buckets: dict[tuple[str, str], list[Evidence]] = defaultdict(list)
    for item in evidence:
        if subquestion_id is not None and item.subquestion_id != subquestion_id:
            continue
        key = (item.subquestion_id, " ".join(item.claim.lower().split()))
        buckets[key].append(item)

    clusters = [
        ClaimCluster(claim=items[0].claim, subquestion_id=sq_id, evidence=items)
        for (sq_id, _), items in buckets.items()
    ]
    # Best-supported first: what the synthesizer should lead with.
    clusters.sort(key=lambda c: (c.support_count, c.mean_strength), reverse=True)
    return clusters


def weakly_supported_claims(
    evidence: list[Evidence], subquestion_id: str | None = None
) -> list[ClaimCluster]:
    return [c for c in cluster_claims(evidence, subquestion_id) if c.is_weakly_supported]


@dataclass
class DiversityReport:
    """Whether the corpus leans on too few papers, groups or years."""

    document_count: int
    distinct_first_authors: int
    distinct_venues: int
    top_author_share: float
    top_document_share: float
    preprint_share: float
    web_share: float
    median_year: int | None
    recent_share: float
    """Fraction published within the last three years."""

    @property
    def is_concentrated(self) -> bool:
        return (
            self.top_document_share > 0.35
            or self.top_author_share > 0.30
            or self.distinct_first_authors < 4
        )

    def summary(self) -> str:
        parts = [
            f"{self.document_count} documents cited",
            f"{self.distinct_first_authors} distinct first authors",
            f"{self.distinct_venues} distinct venues",
            f"top document supplies {self.top_document_share:.0%} of evidence",
            f"top author group supplies {self.top_author_share:.0%}",
            f"{self.preprint_share:.0%} preprints, {self.web_share:.0%} grey literature",
        ]
        if self.median_year:
            parts.append(f"median year {self.median_year}")
        parts.append(f"{self.recent_share:.0%} from the last three years")
        return "; ".join(parts)


def diversity_report(
    evidence: list[Evidence], documents: dict[str, Document]
) -> DiversityReport:
    """Measure how concentrated the evidence base is.

    Computed over documents that actually contributed evidence, not over
    everything retrieved -- a corpus of 200 papers where eight supply all the
    evidence is a concentrated corpus.
    """
    cited_ids = [e.doc_id for e in evidence]
    unique_ids = sorted(set(cited_ids))
    cited_docs = [documents[d] for d in unique_ids if d in documents]

    if not cited_docs:
        return DiversityReport(0, 0, 0, 0.0, 0.0, 0.0, 0.0, None, 0.0)

    doc_counts = Counter(cited_ids)
    total_evidence = max(len(cited_ids), 1)

    first_authors = Counter(
        doc.authors[0].split()[-1].lower() for doc in cited_docs if doc.authors
    )
    author_evidence = Counter()
    for item in evidence:
        doc = documents.get(item.doc_id)
        if doc and doc.authors:
            author_evidence[doc.authors[0].split()[-1].lower()] += 1

    venues = {doc.venue.lower() for doc in cited_docs if doc.venue}
    years = sorted(doc.year for doc in cited_docs if doc.year)

    current_year = datetime.date.today().year
    dated_docs = [doc for doc in cited_docs if doc.year]
    recent_share = (
        sum(1 for doc in dated_docs if current_year - doc.year <= 3) / len(dated_docs)
        if dated_docs
        else 0.0
    )

    return DiversityReport(
        document_count=len(cited_docs),
        distinct_first_authors=len(first_authors),
        distinct_venues=len(venues),
        top_author_share=(
            author_evidence.most_common(1)[0][1] / total_evidence if author_evidence else 0.0
        ),
        top_document_share=doc_counts.most_common(1)[0][1] / total_evidence,
        preprint_share=sum(1 for d in cited_docs if d.is_preprint) / len(cited_docs),
        web_share=sum(1 for d in cited_docs if "web" in d.sources) / len(cited_docs),
        median_year=years[len(years) // 2] if years else None,
        recent_share=recent_share,
    )
