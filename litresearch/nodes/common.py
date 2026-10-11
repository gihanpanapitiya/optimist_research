"""Helpers shared across graph nodes.

Mostly state-to-prompt formatting. Keeping it here means every node presents
the research state to its model in the same shape, which matters because the
planner, the critic and the synthesizers all reason over overlapping slices
of it.
"""

from __future__ import annotations

import datetime
import logging
import re
from collections.abc import Collection

from ..state import (
    Document,
    Evidence,
    ResearchState,
    SearchIntent,
    Subquestion,
)
from ..store.evidence import cluster_claims

logger = logging.getLogger(__name__)

SOURCE_DESCRIPTIONS = {
    "arxiv": (
        "arXiv preprints. Strong for very recent CS/ML/physics work, often months "
        "ahead of publication. Keyword matching, not semantic: use precise technical "
        "terms and quoted phrases. No citation graph."
    ),
    "semantic_scholar": (
        "Semantic Scholar. Cross-domain, semantically ranked, with citation counts "
        "and the richest citation graph. Natural-language queries work well. A "
        "strong general-purpose choice."
    ),
    "openalex": (
        "OpenAlex. Very broad cross-domain scholarly index with reliable DOIs and "
        "both citation directions. Handles natural-language queries. A strong "
        "general-purpose choice, and the one to reach for when you want breadth "
        "or the published version of a preprint."
    ),
    "crossref": (
        "Crossref. The DOI registry. Best for authoritative bibliographic records of "
        "published work; abstracts are often missing. Use for peer-reviewed venues."
    ),
    "pubmed": (
        "PubMed. Biomedical and life sciences. Supports MeSH terms and boolean "
        "operators. Only useful for clinical, biological or health-related intents."
    ),
    "web": (
        "General web search. Finds grey literature: lab blog posts, technical "
        "reports, framework documentation, very recent unindexed work. Lower "
        "evidentiary weight than peer-reviewed sources."
    ),
}


def current_year() -> int:
    return datetime.date.today().year


def log_line(message: str) -> str:
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    logger.info(message)
    return f"[{stamp}] {message}"


def active_subquestions(state: ResearchState) -> list[Subquestion]:
    """Subquestions still worth working on, highest priority first."""
    subquestions = (state.get("subquestions") or {}).values()
    active = [sq for sq in subquestions if sq.is_active()]
    return sorted(active, key=lambda sq: (-sq.priority, sq.id))


def sorted_subquestions(state: ResearchState) -> list[Subquestion]:
    def sort_key(sq: Subquestion) -> list:
        return [int(p) for p in re.findall(r"\d+", sq.id)] or [999]

    return sorted((state.get("subquestions") or {}).values(), key=sort_key)


def format_subquestion_tree(state: ResearchState, include_coverage: bool = True) -> str:
    """The research-question tree as an indented list."""
    lines = []
    for sq in sorted_subquestions(state):
        indent = "  " * sq.depth
        bits = [f"{indent}- {sq.id}: {sq.question}"]
        meta = [f"status={sq.status}", f"priority={sq.priority:.2f}"]
        if include_coverage:
            meta.append(f"coverage={sq.coverage:.2f}")
            meta.append(f"docs_read={sq.documents_read}")
        bits.append(f" [{', '.join(meta)}]")
        lines.append("".join(bits))
    return "\n".join(lines) or "(no subquestions yet)"


def format_source_menu(source_names: list[str]) -> str:
    return "\n".join(
        f"- {name}: {SOURCE_DESCRIPTIONS.get(name, 'A literature source.')}"
        for name in source_names
    )


def format_document_brief(doc: Document, index: int | None = None) -> str:
    """Compact document representation for screening prompts."""
    prefix = f"[{index}] " if index is not None else ""
    header = f"{prefix}{doc.title}"
    meta = []
    if doc.authors:
        authors = ", ".join(doc.authors[:3])
        if len(doc.authors) > 3:
            authors += " et al."
        meta.append(authors)
    if doc.year:
        meta.append(str(doc.year))
    if doc.venue:
        meta.append(doc.venue)
    if doc.citation_count is not None:
        meta.append(f"{doc.citation_count} citations")
    meta.append(f"via {', '.join(doc.sources)}")
    if doc.is_preprint:
        meta.append("preprint")

    abstract = (doc.abstract or "").strip()
    if len(abstract) > 1200:
        abstract = abstract[:1200].rsplit(" ", 1)[0] + " […]"
    if not abstract:
        abstract = "(no abstract available)"

    return f"{header}\n  {' · '.join(meta)}\n  {abstract}"


def format_document_for_reading(doc: Document, full_text: str | None = None) -> str:
    """Full document presentation for the evidence extractor."""
    lines = [f"Title: {doc.title}"]
    if doc.authors:
        lines.append(f"Authors: {', '.join(doc.authors[:15])}")
    lines.append(f"Year: {doc.year or 'unknown'}")
    if doc.venue:
        lines.append(f"Venue: {doc.venue}")
    if doc.citation_count is not None:
        lines.append(f"Citations: {doc.citation_count}")
    lines.append(f"Type: {'preprint' if doc.is_preprint else 'published'}")
    if doc.publication_types:
        lines.append(f"Categories: {', '.join(doc.publication_types)}")
    if doc.doi:
        lines.append(f"DOI: {doc.doi}")

    if full_text:
        lines.append("")
        lines.append("--- FULL TEXT ---")
        lines.append(full_text)
    else:
        lines.append("")
        lines.append("--- ABSTRACT ---")
        lines.append(doc.abstract or "(no abstract available)")
    return "\n".join(lines)


def format_evidence_for_synthesis(
    evidence: list[Evidence],
    documents: dict[str, Document],
    subquestion_id: str | None = None,
) -> str:
    """Evidence grouped into claim clusters, with per-document attribution.

    Presenting clusters rather than a flat list is what lets the synthesizer
    see which claims are corroborated and which rest on a single paper.
    """
    clusters = cluster_claims(evidence, subquestion_id)
    if not clusters:
        return "(no evidence collected)"

    blocks = []
    for cluster in clusters:
        refs = []
        for doc_id in cluster.doc_ids:
            doc = documents.get(doc_id)
            if doc:
                refs.append(f"{doc_id} ({doc.citation_key}, {doc.year or 'n.d.'})")
            else:
                refs.append(doc_id)
        header = (
            f"CLAIM: {cluster.claim}\n"
            f"  supported by {cluster.support_count} document(s): {', '.join(refs)}\n"
            f"  mean strength {cluster.mean_strength:.2f}, "
            f"empirical: {'yes' if cluster.has_empirical_support else 'no'}"
        )
        details = []
        for item in cluster.evidence:
            where = f", {item.location}" if item.location else ""
            flag = " [LIMITATION]" if item.is_limitation else ""
            details.append(
                f"    - {item.doc_id}{where} ({item.evidence_type}){flag}: {item.support}"
            )
        blocks.append(header + "\n" + "\n".join(details))
    return "\n\n".join(blocks)


def format_intents(intents: dict[str, SearchIntent], subquestion_id: str) -> str:
    relevant = [i for i in intents.values() if i.subquestion_id == subquestion_id]
    if not relevant:
        return "(none)"
    return "\n".join(
        f"- [{i.status}] {i.intent} (found {i.documents_found}, "
        f"relevant {i.relevant_found})"
        for i in relevant
    )


def issued_queries(state: ResearchState) -> set[tuple[str, str]]:
    """Every ``(source, query)`` pair already issued in this run.

    Reconstructed from document provenance, which is the only durable record
    of what was searched -- the per-iteration query list is cleared each
    round. Used both to tell the model what not to repeat and to drop
    repeats it produces anyway.
    """
    issued: set[tuple[str, str]] = set()
    for doc in (state.get("documents") or {}).values():
        for prov in doc.discovered_by:
            if prov.via == "search":
                issued.add((prov.source, prov.query.lower().strip()))
    return issued


def format_search_history(state: ResearchState, limit: int = 40) -> str:
    """Queries already issued, so later rounds do not repeat them."""
    queries: list[tuple[str, str]] = []
    seen = set()
    for doc in (state.get("documents") or {}).values():
        for prov in doc.discovered_by:
            key = (prov.source, prov.query)
            if key not in seen and prov.via == "search":
                seen.add(key)
                queries.append(key)
    if not queries:
        return "(no searches yet)"
    shown = queries[:limit]
    lines = [f"- {source}: {query}" for source, query in shown]
    if len(queries) > limit:
        lines.append(f"- … and {len(queries) - limit} more")
    return "\n".join(lines)


def resolve_subquestion_id(raw: str, valid_ids: Collection[str]) -> str | None:
    """Match a model-returned subquestion id against the ids we actually have.

    Models do not reliably return the bare id. Prompts present subquestions
    as ``- RQ1: How do agents generate plans?`` and some models echo the whole
    line back as the ``subquestion_id``. Discarding those would be quietly
    catastrophic: screening scores and extracted evidence are attached by id,
    so every item would be dropped and nothing would ever be read.

    Returns the canonical id, or ``None`` when the value does not
    unambiguously name a known subquestion.
    """
    if not raw:
        return None
    text = raw.strip()
    if text in valid_ids:
        return text

    lowered = text.lower()
    exact = {v.lower(): v for v in valid_ids}
    if lowered in exact:
        return exact[lowered]

    # The longest known id the text begins with, so "RQ2.1: ..." resolves to
    # RQ2.1 rather than RQ2. A boundary check stops "RQ10" matching "RQ1".
    def is_prefix(candidate: str) -> bool:
        if not lowered.startswith(candidate.lower()):
            return False
        rest = text[len(candidate) :]
        return not rest or not rest[0].isalnum()

    matches = sorted(
        (v for v in valid_ids if is_prefix(v)), key=len, reverse=True
    )
    if matches:
        return matches[0]
    return None


def truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " […]"
