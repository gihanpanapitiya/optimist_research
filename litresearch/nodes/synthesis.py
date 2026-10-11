"""Subquestion-level and global synthesis.

Subquestion synthesis fans out with ``Send``, one branch per subquestion that
has evidence. Each consolidates its evidence into an answer, reports what the
evidence does *not* establish, and scores its own coverage -- which is what
the critic later acts on.

Global synthesis then reads those answers and constructs the cross-cutting
narrative plus the report's structure. Crucially, it derives that structure
from what the evidence showed rather than echoing the initial subquestion
list: the right organization of a literature review is usually only apparent
after the literature has been read.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send

from ..llm import ainvoke_structured
from ..schemas import GlobalSynthesisOutput, SubquestionSynthesisOutput
from ..state import (
    BudgetUsage,
    Contradiction,
    Document,
    Evidence,
    ResearchState,
    Subquestion,
    SynthesisState,
)
from ..store.evidence import diversity_report, group_by_subquestion
from .common import (
    format_evidence_for_synthesis,
    log_line,
    sorted_subquestions,
    truncate,
)

logger = logging.getLogger(__name__)

SQ_SYSTEM = """You synthesize the evidence collected for one research \
subquestion into an answer.

You are given evidence grouped into CLAIM CLUSTERS. Each cluster shows a \
claim, which documents support it, how many, the mean strength of that \
support, and whether any of it is empirical.

Write a synthesis, not a summary. The difference matters:

  A summary walks through papers one at a time: "Smith et al. propose X. \
Jones et al. propose Y."
  A synthesis is organized by SUBSTANCE: it states what is known, groups the \
approaches that share an idea, sets the competing ones against each other, \
and says what the evidence actually shows.

Never write a paragraph per paper. Organize by finding, mechanism or \
disagreement, citing documents as [doc_0001] inline. Cite every claim; a \
sentence with no citation should be a statement about the shape of the \
literature, not a factual claim about the subject.

Represent the strength of the evidence honestly, and make it visible in the \
prose. Distinguish:

- what multiple independent papers show, with empirical support;
- what one paper reports and nobody has replicated;
- what is asserted or argued but not demonstrated.

A claim supported by one document is "X reports" or "a single study finds", \
never "the literature shows" or "research demonstrates". Where the evidence \
is contested, say so and give both sides rather than averaging them into a \
false consensus.

Report genuine contradictions. Do not smooth over conflicting findings: a \
real disagreement in the literature is a finding in itself.

Then judge your own coverage. Score how well this evidence answers the \
SUBQUESTION AS ASKED -- not how many papers were read. Twelve papers that all \
address one facet of a three-facet question is low coverage. Four papers that \
settle a narrow question is high coverage. Set answered=true only if more \
searching would not materially improve the answer.

In knowledge_gaps, name what the evidence does not establish and should. Be \
specific enough to search for: "no studies evaluating replanning beyond \
20-step horizons" is actionable, "more research is needed" is not."""

SQ_HUMAN = """Main research question:
{main_question}

SUBQUESTION TO SYNTHESIZE
{sq_id}: {question}
Why it matters: {rationale}

Documents read for this subquestion: {doc_count}

EVIDENCE (grouped into claim clusters):

{evidence}

{contradictions}
Write the synthesis for {sq_id}."""

GLOBAL_SYSTEM = """You are the global synthesizer of a literature research \
system. You receive the per-subquestion syntheses and construct the overall \
research narrative.

Your question is: what does the ENTIRE body of evidence tell us about the \
original research question?

Two things are expected of you.

FIRST, synthesize across subquestions. The subquestions were a device for \
acquiring evidence; the value you add is seeing what emerges when their \
findings are put together. Look for:
- themes that cut across several subquestions and that no single one owns;
- tensions between what different subquestions' evidence implies;
- a trajectory: where this area has come from and where it is heading;
- findings that only become visible when evidence is combined.

Do not concatenate the subquestion syntheses. If your narrative reads as \
"regarding RQ1 ... regarding RQ2 ...", you have not synthesized anything.

SECOND, design the report's structure. Organize it around WHAT THE EVIDENCE \
SHOWED, not around the subquestions used to gather it. The initial research \
plan guided information acquisition; it does not determine the final report \
structure, and the right structure is usually only apparent now.

So: if the evidence clustered around three competing paradigms, the report is \
organized by paradigm, whatever the subquestions were. If it revealed a \
historical progression, organize chronologically. If one subquestion turned \
out to contain the whole story, give it several sections and fold the others \
into context.

Each section names the subquestions whose evidence feeds it -- several \
sections may draw on one subquestion, and one section may draw on several. \
Framing sections may name none.

Be honest about the state of the literature. If the evidence is thin, \
contested, or concentrated in a few groups, the report should say so. A review \
that overstates consensus is worse than one that reports uncertainty."""

GLOBAL_HUMAN = """Main research question:
{main_question}

{interpretation}
RESEARCH SCOPE
- Subquestions investigated: {sq_count}
- Documents retrieved: {doc_total}; read in depth: {doc_read}
- Evidence items extracted: {evidence_count}
- Evidence base: {diversity}

SUBQUESTION SYNTHESES

{syntheses}

{contradictions}
Construct the global synthesis and the report outline."""


# ---------------------------------------------------------------------------
# Subquestion-level synthesis
# ---------------------------------------------------------------------------


def dispatch_synthesizers(state: ResearchState) -> list[Send] | str:
    """Fan out one synthesis branch per subquestion that has evidence."""
    evidence_by_sq = group_by_subquestion(state.get("evidence") or [])
    subquestions: dict[str, Subquestion] = state.get("subquestions") or {}
    documents: dict[str, Document] = state.get("documents") or {}

    sends: list[Send] = []
    for sq_id, items in evidence_by_sq.items():
        sq = subquestions.get(sq_id)
        if sq is None or sq.status == "deprioritized":
            continue
        # Only the documents this subquestion's evidence refers to, so a
        # branch's payload stays proportional to its own work.
        cited = {
            item.doc_id: documents[item.doc_id]
            for item in items
            if item.doc_id in documents
        }
        sends.append(
            Send(
                "synthesize_subquestion",
                SynthesisState(
                    main_question=state["main_question"],
                    subquestion=sq,
                    evidence=items,
                    documents=cited,
                    iteration=state.get("iteration", 1),
                ),
            )
        )

    return sends or "global_synthesis"


async def synthesize_subquestion(state: SynthesisState) -> dict:
    """Consolidate one subquestion's evidence into an answer."""
    sq: Subquestion = state["subquestion"]
    evidence: list[Evidence] = state["evidence"]
    documents: dict[str, Document] = state["documents"]

    contradiction_text = ""
    limitation_items = [e for e in evidence if e.is_limitation]
    if limitation_items:
        contradiction_text = (
            "REPORTED LIMITATIONS:\n"
            + "\n".join(f"- {e.doc_id}: {e.support}" for e in limitation_items)
            + "\n\n"
        )

    messages = [
        SystemMessage(SQ_SYSTEM),
        HumanMessage(
            SQ_HUMAN.format(
                main_question=state["main_question"],
                sq_id=sq.id,
                question=sq.question,
                rationale=sq.rationale or "(not stated)",
                doc_count=len({e.doc_id for e in evidence}),
                evidence=format_evidence_for_synthesis(evidence, documents, sq.id),
                contradictions=contradiction_text,
            )
        ),
    ]

    output = await ainvoke_structured("synthesizer", SubquestionSynthesisOutput, messages)

    if output is None:
        logger.warning("synthesis failed for %s", sq.id)
        return {
            "budget": BudgetUsage(llm_calls=1),
            "log": [log_line(f"{sq.id}: synthesis failed")],
        }

    updated = sq.model_copy(deep=True)
    updated.synthesis = _assemble_synthesis(output)
    updated.coverage = max(0.0, min(1.0, output.coverage))
    updated.knowledge_gaps = list(
        dict.fromkeys(
            [
                # Drop the terminology hints the readers parked here; they
                # have served their purpose in query generation.
                *[g for g in sq.knowledge_gaps if not g.startswith("terminology:")],
                *output.knowledge_gaps,
            ]
        )
    )
    # Status is deliberately NOT touched here. Only the critic may change it:
    # a synthesizer sees its own subquestion in isolation and cannot weigh it
    # against the others. It also runs after the critic may have marked the
    # subquestion superseded or deprioritized, and a write here would clobber
    # that -- the self-assessment travels in `coverage` instead.

    return {
        "subquestions": {sq.id: updated},
        "budget": BudgetUsage(llm_calls=1),
        "log": [
            log_line(
                f"{sq.id}: synthesized {len(evidence)} evidence items, "
                f"coverage {updated.coverage:.2f}"
                + (f", {len(output.knowledge_gaps)} gap(s)" if output.knowledge_gaps else "")
            )
        ],
    }


def _assemble_synthesis(output: SubquestionSynthesisOutput) -> str:
    """Compose the narrative with its structured findings appended."""
    parts = [output.synthesis.strip()]

    sections = [
        ("Consensus", output.consensus_points),
        ("Competing approaches", output.competing_approaches),
        ("Empirical findings", output.empirical_findings),
        ("Limitations", output.limitations),
        ("Unresolved questions", output.unresolved_questions),
    ]
    for heading, items in sections:
        if items:
            parts.append(
                f"**{heading}.**\n" + "\n".join(f"- {item}" for item in items)
            )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Global synthesis
# ---------------------------------------------------------------------------


async def global_synthesis(state: ResearchState) -> dict:
    """Build the cross-cutting narrative and the report outline."""
    subquestions = sorted_subquestions(state)
    documents: dict[str, Document] = state.get("documents") or {}
    evidence: list[Evidence] = state.get("evidence") or []
    contradictions: list[Contradiction] = state.get("contradictions") or []

    synthesized = [sq for sq in subquestions if sq.synthesis]
    if not synthesized:
        return {
            "global_synthesis": "",
            "log": [log_line("no subquestion syntheses available; skipping global synthesis")],
        }

    blocks = []
    for sq in synthesized:
        blocks.append(
            f"### {sq.id}: {sq.question}\n"
            f"(coverage {sq.coverage:.2f}, {sq.documents_read} documents read)\n\n"
            f"{truncate(sq.synthesis, 6000)}"
        )

    contradiction_text = ""
    open_contradictions = [c for c in contradictions if c.status == "open"]
    if open_contradictions:
        contradiction_text = (
            "RECORDED CONTRADICTIONS:\n"
            + "\n".join(f"- {c.description}" for c in open_contradictions[:20])
            + "\n\n"
        )

    report = diversity_report(evidence, documents)
    interpretation = (
        f"Planner's interpretation of the question:\n{state.get('plan_rationale', '')}\n\n"
        if state.get("plan_rationale")
        else ""
    )

    messages = [
        SystemMessage(GLOBAL_SYSTEM),
        HumanMessage(
            GLOBAL_HUMAN.format(
                main_question=state["main_question"],
                interpretation=interpretation,
                sq_count=len(synthesized),
                doc_total=len(documents),
                doc_read=sum(1 for d in documents.values() if d.status == "read"),
                evidence_count=len(evidence),
                diversity=report.summary(),
                syntheses="\n\n".join(blocks),
                contradictions=contradiction_text,
            )
        ),
    ]

    output = await ainvoke_structured("synthesizer", GlobalSynthesisOutput, messages)

    if output is None:
        # Fall back to concatenating the subquestion syntheses: a coarse
        # report still beats no report.
        logger.warning("global synthesis failed; falling back to concatenation")
        fallback = "\n\n".join(
            f"## {sq.id}: {sq.question}\n\n{sq.synthesis}" for sq in synthesized
        )
        return {
            "global_synthesis": fallback,
            "report_outline": [
                {
                    "heading": sq.question,
                    "purpose": "",
                    "subquestion_ids": [sq.id],
                }
                for sq in synthesized
            ],
            "budget": BudgetUsage(llm_calls=1),
            "log": [log_line("global synthesis failed; using concatenated fallback")],
        }

    narrative_parts = [output.narrative.strip()]
    if output.major_themes:
        narrative_parts.append(
            "**Major themes.**\n" + "\n".join(f"- {t}" for t in output.major_themes)
        )
    if output.cross_cutting_findings:
        narrative_parts.append(
            "**Cross-cutting findings.**\n"
            + "\n".join(f"- {f}" for f in output.cross_cutting_findings)
        )
    if output.open_questions:
        narrative_parts.append(
            "**Open questions.**\n" + "\n".join(f"- {q}" for q in output.open_questions)
        )
    if output.emerging_directions:
        narrative_parts.append(
            "**Emerging directions.**\n"
            + "\n".join(f"- {d}" for d in output.emerging_directions)
        )

    outline = [
        {
            "heading": section.heading,
            "purpose": section.purpose,
            "subquestion_ids": section.subquestion_ids,
        }
        for section in output.outline
    ]

    return {
        "global_synthesis": "\n\n".join(narrative_parts),
        "report_outline": outline,
        "budget": BudgetUsage(llm_calls=1),
        "log": [
            log_line(
                f"global synthesis complete: {len(output.major_themes)} theme(s), "
                f"{len(outline)}-section outline"
            )
        ],
    }
