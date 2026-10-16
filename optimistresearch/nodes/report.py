"""Final research report generation.

The report is written from the evidence store and the subquestion syntheses,
following the outline that global synthesis derived from the findings. It is
generated section by section rather than in one call: each section gets the
evidence that actually belongs to it, which keeps citations accurate and
keeps a long report from degrading toward the end.

References are assembled from the document registry rather than by the
model, so every citation resolves to a real retrieved document and nothing
is fabricated.
"""

from __future__ import annotations

import asyncio
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import get_settings
from ..llm import get_model
from ..state import (
    BudgetUsage,
    Contradiction,
    Critique,
    Document,
    Evidence,
    ResearchState,
    Subquestion,
)
from ..store.evidence import cluster_claims, diversity_report, group_by_subquestion
from .common import (
    format_evidence_for_synthesis,
    log_line,
    sorted_subquestions,
    truncate,
)
from .critic import termination_reason

logger = logging.getLogger(__name__)

SECTION_SYSTEM = """You are writing one section of a literature research report.

Write in the register of a careful survey article: precise, substantive, and \
organized by idea rather than by paper. No bullet-point dumps where prose \
would serve, and no filler.

Citation rules, which are absolute:
- Cite documents inline using their ids exactly as given, like [doc_0012]. \
Several ids in one bracket when several sources support a point: \
[doc_0012; doc_0031].
- Every factual claim about the subject matter carries a citation.
- Cite ONLY document ids present in the evidence you were given. Never invent \
an id, a title, an author or a result.
- Do not cite a document for something its evidence does not say.

Represent the evidence honestly. The evidence you are given records how many \
documents support each claim and how strong that support is; carry that into \
the prose. "Several studies report" requires several studies. One paper means \
"X reports" or "one study found". Where the literature disagrees, present the \
disagreement rather than resolving it by fiat. Where evidence is thin, say so \
plainly -- a section that admits a gap is more useful than one that papers \
over it.

Write only this section's content. Do not write the heading -- it is added \
for you -- and do not write a preamble, a recap of the report so far, or a \
transition into the next section."""

SECTION_HUMAN = """Report on the research question:
{main_question}

SECTION TO WRITE
Heading: {heading}
What this section must establish: {purpose}

{context}

EVIDENCE AVAILABLE FOR THIS SECTION

{evidence}

{syntheses}
Write the section body in markdown. Use ### for any sub-headings you need \
(the section heading itself is ##)."""

SUMMARY_SYSTEM = """You are writing the executive summary of a literature \
research report.

You have the full report. Write a summary that answers the research question \
directly, in prose, for a reader who will not read further.

Lead with the answer, not with what the report contains. "This review \
examines ..." wastes the most valuable paragraph; state what the literature \
shows.

Cover: what the evidence establishes, where it is contested or thin, and what \
remains open. Carry citations as [doc_XXXX] for the main claims, using only \
ids that appear in the report.

Four to seven paragraphs. No bullet lists, no headings."""


async def write_report(state: ResearchState) -> dict:
    """Generate the final report."""
    documents: dict[str, Document] = state.get("documents") or {}
    evidence: list[Evidence] = state.get("evidence") or []
    outline: list[dict] = state.get("report_outline") or []

    if not evidence:
        report = _empty_report(state)
        return {
            "final_report": report,
            "terminated_because": termination_reason(state),
            "log": [log_line("no evidence collected; wrote a diagnostic report")],
        }

    if not outline:
        # Fall back to one section per synthesized subquestion.
        outline = [
            {"heading": sq.question, "purpose": "", "subquestion_ids": [sq.id]}
            for sq in sorted_subquestions(state)
            if sq.synthesis
        ]

    sections = await _write_sections(state, outline)

    body = "\n\n".join(
        f"## {index}. {section['heading']}\n\n{section['body']}"
        for index, section in enumerate(sections, start=4)
        if section["body"]
    )

    # A model can emit an id that resolves to no retrieved document. Removing
    # those before the references are built keeps the report honest: a reader
    # must never see a citation that points nowhere.
    body = _strip_unresolved_citations(body, set(documents))

    scope = _scope_section(state)
    contradictions = _contradictions_section(state)
    limitations = _limitations_section(state)
    references, used_ids = _references_section(state, body)

    draft = "\n\n".join(part for part in [body, contradictions, limitations] if part)
    summary = _strip_unresolved_citations(
        await _write_summary(state, draft), set(documents)
    )

    title = _title(state["main_question"])
    report = "\n\n".join(
        part
        for part in [
            f"# {title}",
            f"**Research question.** {state['main_question']}",
            "## 1. Executive summary",
            summary,
            "## 2. Research scope and methodology",
            scope,
            "## 3. How to read this report",
            _reading_guide(state),
            draft,
            references,
        ]
        if part
    )

    return {
        "final_report": report,
        "terminated_because": termination_reason(state),
        "budget": BudgetUsage(llm_calls=len(sections) + 1),
        "log": [
            log_line(
                f"wrote report: {len(sections)} section(s), "
                f"{len(used_ids)} documents cited of {len(documents)} retrieved"
            )
        ],
    }


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


async def _write_sections(state: ResearchState, outline: list[dict]) -> list[dict]:
    """Write each outline section concurrently."""
    settings = get_settings()
    semaphore = asyncio.Semaphore(settings.max_concurrent_readers)

    async def one(section: dict, index: int) -> dict:
        async with semaphore:
            body = await _write_section(state, section, outline, index)
        return {"heading": section["heading"], "body": body}

    return list(
        await asyncio.gather(
            *(one(section, index) for index, section in enumerate(outline))
        )
    )


async def _write_section(
    state: ResearchState, section: dict, outline: list[dict], index: int
) -> str:
    documents: dict[str, Document] = state.get("documents") or {}
    subquestions: dict[str, Subquestion] = state.get("subquestions") or {}
    by_sq = group_by_subquestion(state.get("evidence") or [])

    sq_ids = [sq_id for sq_id in section.get("subquestion_ids") or [] if sq_id in subquestions]

    section_evidence: list[Evidence] = []
    for sq_id in sq_ids:
        section_evidence.extend(by_sq.get(sq_id, []))

    if not section_evidence:
        # A framing section with no evidence of its own: give it the global
        # synthesis so it can be written from the overall picture.
        evidence_text = "(no evidence assigned to this section)"
        syntheses_text = (
            "GLOBAL SYNTHESIS\n" + truncate(state.get("global_synthesis") or "", 6000)
        )
    else:
        evidence_text = format_evidence_for_synthesis(section_evidence, documents)
        blocks = []
        for sq_id in sq_ids:
            sq = subquestions.get(sq_id)
            if sq and sq.synthesis:
                blocks.append(f"### {sq.id}: {sq.question}\n{truncate(sq.synthesis, 4000)}")
        syntheses_text = (
            "SUBQUESTION SYNTHESES FEEDING THIS SECTION\n\n" + "\n\n".join(blocks)
            if blocks
            else ""
        )

    # Neighbouring headings, so sections do not duplicate one another.
    others = "\n".join(
        f"- {'>>> ' if i == index else ''}{s['heading']}" for i, s in enumerate(outline)
    )
    context = (
        f"The report's full outline (this section marked >>>):\n{others}\n\n"
        f"Do not cover material that belongs to another section."
    )

    messages = [
        SystemMessage(SECTION_SYSTEM),
        HumanMessage(
            SECTION_HUMAN.format(
                main_question=state["main_question"],
                heading=section["heading"],
                purpose=section.get("purpose") or "(not specified)",
                context=context,
                evidence=evidence_text,
                syntheses=syntheses_text,
            )
        ),
    ]

    try:
        response = await get_model("writer").ainvoke(messages)
    except Exception as exc:  # noqa: BLE001
        logger.warning("section %r failed to generate: %s", section["heading"], exc)
        # Degrade to the raw syntheses rather than dropping the section.
        return "\n\n".join(
            subquestions[sq_id].synthesis for sq_id in sq_ids if subquestions.get(sq_id)
        )

    return _text_of(response).strip()


async def _write_summary(state: ResearchState, draft: str) -> str:
    messages = [
        SystemMessage(SUMMARY_SYSTEM),
        HumanMessage(
            f"Research question:\n{state['main_question']}\n\n"
            f"THE REPORT\n\n{truncate(draft, 40000)}\n\n"
            f"Write the executive summary."
        ),
    ]
    try:
        response = await get_model("writer").ainvoke(messages)
    except Exception as exc:  # noqa: BLE001
        logger.warning("executive summary failed: %s", exc)
        return truncate(state.get("global_synthesis") or "", 4000)
    return _text_of(response).strip()


def _text_of(response) -> str:
    """Extract text from a chat response across content formats."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


# ---------------------------------------------------------------------------
# Deterministic sections
# ---------------------------------------------------------------------------


def _scope_section(state: ResearchState) -> str:
    settings = get_settings()
    documents: dict[str, Document] = state.get("documents") or {}
    evidence: list[Evidence] = state.get("evidence") or []
    usage = state.get("budget") or BudgetUsage()
    subquestions = sorted_subquestions(state)

    source_counts: dict[str, int] = {}
    for doc in documents.values():
        for source in doc.sources:
            source_counts[source] = source_counts.get(source, 0) + 1

    read_count = sum(1 for d in documents.values() if d.status == "read")
    snowballed = sum(1 for d in documents.values() if d.snowball_depth > 0)

    lines = [
        "This report was produced by an iterative literature research agent. "
        "The research question was decomposed into subquestions, each of which "
        "generated search intents that were routed to the sources below and "
        "converted into source-specific queries. Retrieved documents were "
        "deduplicated into a canonical corpus, screened for relevance, and the "
        "most promising were read to extract structured evidence. Findings were "
        "synthesized per subquestion and then globally, after which a critic "
        "assessed coverage and either requested further research or terminated "
        "the run.",
        "",
        "**Research process.**",
        "",
        f"- Research iterations: {usage.research_iterations}",
        f"- Search operations: {usage.search_operations}",
        f"- Snowball rounds: {usage.snowball_rounds}",
        f"- Documents retrieved: {len(documents)}",
        f"- Documents read in depth: {read_count}",
        f"- Documents found via citation snowballing: {snowballed}",
        f"- Evidence items extracted: {len(evidence)}",
        "- Reading mode: "
        + ("full text" if settings.read_full_text else "title, abstract and metadata"),
        "",
        "**Sources queried.**",
        "",
    ]
    for source, count in sorted(source_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {source}: {count} documents")

    lines += ["", "**Subquestions investigated.**", ""]
    for sq in subquestions:
        marker = {
            "answered": "answered",
            "deprioritized": "deprioritized",
            "superseded": "split into sub-questions",
            "in_progress": "partially answered",
            "unexplored": "not investigated",
        }.get(sq.status, sq.status)
        origin = "" if sq.origin == "planner" else ", added by the critic during research"
        lines.append(
            f"- **{sq.id}** {sq.question} — {marker}, coverage {sq.coverage:.0%}, "
            f"{sq.documents_read} documents read{origin}"
        )

    lines += [
        "",
        "**Evidence base.**",
        "",
        diversity_report(evidence, documents).summary() + ".",
        "",
        "**Why the research stopped.**",
        "",
        state.get("terminated_because") or termination_reason(state),
    ]
    return "\n".join(lines)


def _reading_guide(state: ResearchState) -> str:
    """Tell the reader how strongly to take the report's claims."""
    evidence: list[Evidence] = state.get("evidence") or []
    documents: dict[str, Document] = state.get("documents") or {}
    clusters = cluster_claims(evidence)
    weak = [c for c in clusters if c.is_weakly_supported]
    report = diversity_report(evidence, documents)

    lines = [
        # Deliberately not written as a real id: anything matching the
        # doc_NNNN pattern here would be picked up as a citation.
        "Citations are given as bracketed document ids of the form "
        "[doc_NNNN]; the references section maps each to its bibliographic "
        "record. Every factual claim about the subject matter is traceable to "
        "a specific retrieved document.",
        "",
    ]

    if clusters:
        pct = len(weak) / len(clusters)
        lines.append(
            f"Of {len(clusters)} distinct claims extracted from the literature, "
            f"{len(weak)} ({pct:.0%}) rest on a single source or on support the "
            f"extractor judged weak. Those claims are attributed to their "
            f"individual sources in the text rather than presented as settled."
        )
        lines.append("")

    caveats = []
    if report.is_concentrated:
        caveats.append(
            f"the evidence base is concentrated ({report.summary()}), so these "
            f"findings may reflect the emphases of a small number of groups"
        )
    if report.preprint_share > 0.5:
        caveats.append(
            f"{report.preprint_share:.0%} of cited documents are preprints that "
            f"have not completed peer review"
        )
    if report.web_share > 0.2:
        caveats.append(
            f"{report.web_share:.0%} of cited documents are grey literature from "
            f"web search rather than peer-reviewed venues"
        )
    if not get_settings().read_full_text:
        caveats.append(
            "evidence was extracted from titles, abstracts and metadata rather "
            "than full texts, so experimental details could not be verified"
        )

    if caveats:
        lines.append(
            "Readers should weigh the following when interpreting this report: "
            + "; ".join(caveats)
            + "."
        )
    return "\n".join(lines)


def _contradictions_section(state: ResearchState) -> str:
    contradictions: list[Contradiction] = state.get("contradictions") or []
    critique: Critique | None = state.get("critique")
    open_items = [c for c in contradictions if c.status in ("open", "investigating")]

    gaps = critique.knowledge_gaps if critique else []
    if not open_items and not gaps:
        return ""

    lines = ["## Contradictions and open questions", ""]
    if open_items:
        lines += ["**Conflicting findings in the literature.**", ""]
        documents: dict[str, Document] = state.get("documents") or {}
        for item in open_items:
            refs = ", ".join(f"[{d}]" for d in item.doc_ids if d in documents)
            lines.append(f"- {item.description}{(' ' + refs) if refs else ''}")
        lines.append("")
    if gaps:
        lines += ["**Questions the retrieved literature does not settle.**", ""]
        lines += [f"- {gap}" for gap in gaps]
    return "\n".join(lines)


def _limitations_section(state: ResearchState) -> str:
    """Limitations of this review, distinct from limitations of the literature."""
    critique: Critique | None = state.get("critique")
    settings = get_settings()
    usage = state.get("budget") or BudgetUsage()
    documents: dict[str, Document] = state.get("documents") or {}

    lines = ["## Limitations of this review", ""]
    items = []

    unread = sum(1 for d in documents.values() if d.status in ("candidate", "selected"))
    if unread:
        items.append(
            f"{unread} retrieved documents screened as potentially relevant were "
            f"not read, the reading budget having been spent on higher-ranked ones."
        )
    if usage.search_operations >= settings.budget.max_search_operations:
        items.append(
            "The search-operation budget was exhausted, so some search intents "
            "were never issued."
        )
    if critique and critique.status == "continue":
        items.append(
            "The research critic judged the investigation incomplete when the run "
            "ended; the gaps it identified are listed above."
        )
    if critique and critique.unsupported_claims:
        items.append(
            "The critic flagged these statements as insufficiently supported by "
            "the retrieved evidence: "
            + "; ".join(critique.unsupported_claims[:5])
            + "."
        )
    failed = sum(1 for d in documents.values() if d.status == "failed")
    if failed:
        items.append(
            f"{failed} documents could not be read (no retrievable abstract or "
            f"full text) and contributed no evidence."
        )
    items.append(
        "Coverage is bounded by what the queried sources index and by the query "
        "formulations used; work using unanticipated terminology may have been missed."
    )

    lines += [f"- {item}" for item in items]
    return "\n".join(lines)


def _strip_unresolved_citations(text: str, known_ids: set[str]) -> str:
    """Remove citations to document ids that are not in the corpus.

    Bracketed groups like ``[doc_0012; doc_0031]`` are rewritten to keep only
    the ids that resolve, and dropped entirely when none do. The surrounding
    sentence is left alone -- losing a citation is better than losing a claim,
    and the critic's unsupported-claims list is where that gets reported.
    """
    if not text:
        return text

    unknown: set[str] = set()

    def rewrite(match: re.Match) -> str:
        ids = re.findall(r"doc_\d{4}", match.group(0))
        if not ids:
            return match.group(0)
        resolved = [doc_id for doc_id in ids if doc_id in known_ids]
        unknown.update(set(ids) - set(resolved))
        if not resolved:
            return ""
        return "[" + "; ".join(dict.fromkeys(resolved)) + "]"

    # Only touch brackets whose entire content is document ids and separators,
    # so ordinary bracketed prose and markdown links are untouched.
    cleaned = re.sub(r"\[\s*doc_\d{4}(?:\s*[;,]\s*doc_\d{4})*\s*\]", rewrite, text)

    # Any remaining bare mention of an unknown id (outside brackets).
    for doc_id in re.findall(r"doc_\d{4}", cleaned):
        if doc_id not in known_ids:
            unknown.add(doc_id)
            cleaned = cleaned.replace(f"[{doc_id}]", "").replace(doc_id, "")

    if unknown:
        logger.warning(
            "removed citations to %d unknown document id(s): %s",
            len(unknown),
            ", ".join(sorted(unknown)),
        )
    # Tidy the spacing the removals leave behind.
    cleaned = re.sub(r" +([.,;:])", r"\1", cleaned)
    return re.sub(r"[ \t]{2,}", " ", cleaned)


def _references_section(state: ResearchState, body: str) -> tuple[str, set[str]]:
    """Build references from documents actually cited in the body.

    Driven by the registry, not the model: a citation that resolves to no
    retrieved document is dropped, and nothing is invented.
    """
    documents: dict[str, Document] = state.get("documents") or {}
    cited = set(re.findall(r"doc_\d{4}", body))
    resolved = {doc_id for doc_id in cited if doc_id in documents}

    missing = cited - resolved
    if missing:
        logger.warning("report cited unknown document ids: %s", ", ".join(sorted(missing)))

    if not resolved:
        return "", resolved

    ordered = sorted(
        (documents[doc_id] for doc_id in resolved),
        key=lambda d: (d.authors[0].split()[-1].lower() if d.authors else "zz", -(d.year or 0)),
    )

    lines = ["## References", ""]
    for doc in ordered:
        marks = []
        if doc.is_preprint:
            marks.append("preprint")
        if "web" in doc.sources and len(doc.sources) == 1:
            marks.append("web source")
        suffix = f" [{', '.join(marks)}]" if marks else ""
        lines.append(f"- **{doc.doc_id}** — {doc.reference_label()}{suffix}")

    return "\n".join(lines), resolved


def _empty_report(state: ResearchState) -> str:
    """Diagnostic report for a run that collected no evidence."""
    documents = state.get("documents") or {}
    lines = [
        f"# Literature research: {state['main_question']}",
        "",
        "## No evidence was collected",
        "",
        "This research run completed without extracting any evidence. "
        f"{len(documents)} document(s) were retrieved.",
        "",
        "Possible causes:",
        "",
        "- The research question's terminology does not match the indexed literature.",
        "- The enabled sources do not cover this subject area.",
        "- Screening rejected all candidates as irrelevant.",
        "- Document reading failed (check the log for extraction errors).",
        "",
        "## Research log",
        "",
    ]
    lines += [f"- {entry}" for entry in (state.get("log") or [])]
    if documents:
        lines += ["", "## Documents retrieved but not used", ""]
        for doc in list(documents.values())[:50]:
            lines.append(
                f"- {doc.doc_id} ({doc.status}, relevance {doc.max_relevance():.2f}) "
                f"{doc.title}"
            )
    return "\n".join(lines)


def _title(question: str) -> str:
    cleaned = question.strip().rstrip("?").strip()
    if not cleaned:
        return "Literature Review"
    return f"Literature Review: {cleaned[0].upper()}{cleaned[1:]}"
