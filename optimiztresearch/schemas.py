"""Structured-output schemas for the LLM nodes.

These are the wire format between models and the research state, kept
separate from :mod:`optimiztresearch.state` for two reasons:

* State models carry derived fields and ids the model should not invent.
* Portability. Providers differ in what JSON Schema they accept, so these
  schemas avoid open-ended mappings (``dict[str, float]`` becomes a list of
  pairs) and deeply nested unions, both of which are common failure points
  for strict schema validation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


class PlannedSubquestion(BaseModel):
    id: str = Field(
        description="Identifier like 'RQ1', or 'RQ4.2' for a child of RQ4."
    )
    question: str = Field(
        description="The subquestion, phrased as you would ask a knowledgeable person."
    )
    rationale: str = Field(
        description="Why answering the main question requires establishing this."
    )
    priority: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How central this is to the main question.",
    )
    parent_id: str | None = Field(
        default=None, description="Parent subquestion id, or null for a top-level one."
    )


class PlanOutput(BaseModel):
    """Initial decomposition of the main research question."""

    interpretation: str = Field(
        description="What the main question is actually asking, including scope "
        "judgements you are making and any ambiguity you resolved."
    )
    rationale: str = Field(
        description="Why this particular decomposition gives comprehensive coverage."
    )
    subquestions: list[PlannedSubquestion] = Field(
        description="The minimum set of subquestions that covers the main question."
    )


# ---------------------------------------------------------------------------
# Search intents
# ---------------------------------------------------------------------------


class PlannedIntent(BaseModel):
    intent: str = Field(
        description="A retrieval goal describing the documents that would close "
        "this knowledge gap. Starts with a verb, e.g. 'Find empirical studies "
        "evaluating ...'. Describes documents, not search keywords."
    )
    rationale: str = Field(description="Which part of the subquestion this serves.")
    desired_evidence: list[str] = Field(
        default_factory=list,
        description="The specific kinds of findings that would satisfy this intent.",
    )
    preferred_sources: list[str] = Field(
        default_factory=list,
        description="Subset of the offered sources best suited to this intent.",
    )


class IntentOutput(BaseModel):
    intents: list[PlannedIntent]


# ---------------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------------


class GeneratedQuery(BaseModel):
    source: str = Field(description="Which source this query is written for.")
    query: str = Field(
        description="The literal query string. Use the idiom of the target "
        "source: quoted phrases and field operators where supported, natural "
        "language where the source ranks semantically."
    )
    year_from: int | None = Field(
        default=None,
        description="Earliest publication year, only when recency genuinely matters.",
    )
    rationale: str = Field(default="", description="What this phrasing is meant to surface.")


class QueryPlanOutput(BaseModel):
    """Source routing plus per-source query generation for one intent."""

    reasoning: str = Field(
        default="",
        description="Why these sources, and why these phrasings differ between them.",
    )
    queries: list[GeneratedQuery] = Field(
        description="Queries across the selected sources. Several phrasings for "
        "one source is fine when the terminology is genuinely contested."
    )


# ---------------------------------------------------------------------------
# Screening and ranking
# ---------------------------------------------------------------------------


class SubquestionScore(BaseModel):
    subquestion_id: str
    relevance: float = Field(ge=0.0, le=1.0)


class ScreenedDocument(BaseModel):
    index: int = Field(description="The document's index in the provided list.")
    scores: list[SubquestionScore] = Field(
        default_factory=list,
        description="Relevance to each subquestion it plausibly bears on. Omit "
        "subquestions it is irrelevant to rather than scoring them zero.",
    )
    note: str = Field(
        default="",
        description="One line: what this document would contribute, or why it is off-target.",
    )


class ScreeningOutput(BaseModel):
    documents: list[ScreenedDocument]


# ---------------------------------------------------------------------------
# Document reading / evidence extraction
# ---------------------------------------------------------------------------


class ExtractedEvidence(BaseModel):
    claim: str = Field(
        description="A single, self-contained, checkable assertion the document "
        "establishes or argues. Not a topic label."
    )
    support: str = Field(
        description="What in the document supports the claim: the result, "
        "measurement, argument or construction, with specifics."
    )
    subquestion_id: str = Field(description="Which subquestion this bears on.")
    location: str = Field(
        default="",
        description="Where in the document, e.g. 'Section 4.2', 'Table 3', 'Abstract'.",
    )
    evidence_type: Literal[
        "experimental", "theoretical", "survey", "benchmark", "position", "anecdotal"
    ] = "theoretical"
    strength: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Your confidence that the document really establishes this "
        "claim. Low for claims asserted without support.",
    )
    is_limitation: bool = Field(
        default=False, description="True if this records a limitation or negative result."
    )
    verbatim_quote: str | None = Field(
        default=None, description="A short exact quote, when one makes the claim concrete."
    )


class ObservedContradiction(BaseModel):
    description: str = Field(
        description="The conflict, naming what disagrees with what and on which point."
    )
    subquestion_id: str = ""
    conflicting_doc_hint: str = Field(
        default="",
        description="Title or author-year of the other work involved, if identifiable.",
    )


class ReadingOutput(BaseModel):
    """Structured evidence extracted from one document."""

    relevant: bool = Field(
        description="False if, on reading, this document does not actually bear "
        "on any of the assigned subquestions."
    )
    main_contribution: str = ""
    method: str = ""
    experimental_setup: str = Field(
        default="", description="Datasets, benchmarks, baselines, metrics. Empty if none."
    )
    major_findings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(
        default_factory=list,
        description="Limitations, including ones the authors do not acknowledge.",
    )
    evidence: list[ExtractedEvidence] = Field(
        default_factory=list,
        description="The structured evidence. This is the primary output; "
        "prose summary is secondary.",
    )
    contradictions: list[ObservedContradiction] = Field(default_factory=list)
    new_subquestions: list[str] = Field(
        default_factory=list,
        description="Research directions this document reveals that the current "
        "plan does not cover. Only genuinely important omissions.",
    )
    snowball_candidates: list[str] = Field(
        default_factory=list,
        description="Specific cited works worth retrieving, by title.",
    )
    terminology_notes: list[str] = Field(
        default_factory=list,
        description="Alternative terms this field uses for the concepts at issue, "
        "which later queries should try.",
    )


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


class SubquestionSynthesisOutput(BaseModel):
    """Consolidated answer to one subquestion."""

    synthesis: str = Field(
        description="A synthesis in markdown that answers the subquestion from "
        "the evidence. Organize by substance, cite documents as [doc_XXXX], and "
        "attribute every claim. Do not summarize paper by paper."
    )
    consensus_points: list[str] = Field(
        default_factory=list, description="What the literature agrees on."
    )
    competing_approaches: list[str] = Field(
        default_factory=list, description="Genuine methodological disagreements."
    )
    empirical_findings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    knowledge_gaps: list[str] = Field(
        default_factory=list,
        description="What the retrieved evidence does not establish and should.",
    )
    coverage: float = Field(
        ge=0.0,
        le=1.0,
        description="How well the current evidence answers this subquestion. Judge "
        "against the question, not against how many papers were read.",
    )
    answered: bool = Field(
        description="True if further search would not materially improve this answer."
    )


class ReportSection(BaseModel):
    heading: str
    purpose: str = Field(description="What this section establishes.")
    subquestion_ids: list[str] = Field(
        default_factory=list,
        description="Subquestions whose evidence feeds this section. May be "
        "several, or none for framing sections.",
    )


class GlobalSynthesisOutput(BaseModel):
    """The cross-cutting narrative and the report's structure."""

    narrative: str = Field(
        description="What the whole body of evidence says about the main "
        "question, in markdown. Synthesize across subquestions; draw out themes "
        "that no single subquestion owns."
    )
    major_themes: list[str] = Field(default_factory=list)
    cross_cutting_findings: list[str] = Field(
        default_factory=list,
        description="Findings visible only when evidence from several "
        "subquestions is combined.",
    )
    outline: list[ReportSection] = Field(
        description="The report's structure, derived from what the evidence "
        "actually showed rather than from the initial subquestion list."
    )
    open_questions: list[str] = Field(default_factory=list)
    emerging_directions: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Reflection
# ---------------------------------------------------------------------------


class RecommendedIntent(BaseModel):
    subquestion_id: str
    intent: str = Field(description="A retrieval goal, phrased as a search intent.")
    rationale: str = ""


class ProposedPlanEdit(BaseModel):
    action: Literal["add", "split", "deprioritize", "mark_answered", "rephrase"]
    subquestion_id: str | None = Field(
        default=None, description="Target subquestion. Null when action is 'add'."
    )
    question: str | None = Field(
        default=None, description="New or rephrased question text."
    )
    children: list[str] = Field(
        default_factory=list,
        description="For 'split': the child questions to create.",
    )
    priority: float | None = None
    rationale: str = ""


class SnowballRequest(BaseModel):
    doc_id: str = Field(description="The seed document, as doc_XXXX.")
    direction: Literal["backward", "forward"]
    reason: str = Field(
        description="Backward for foundations and precursors; forward for "
        "later developments, improvements and critiques."
    )


class CoverageScore(BaseModel):
    subquestion_id: str
    coverage: float = Field(ge=0.0, le=1.0)


class CritiqueOutput(BaseModel):
    """The research critic's assessment of whether to continue."""

    status: Literal["continue", "sufficient"] = Field(
        description="'sufficient' only when further search would not materially "
        "change the answer to the main question."
    )
    reasoning: str = Field(description="The decisive considerations behind the verdict.")
    overall_coverage: float = Field(ge=0.0, le=1.0)
    subquestion_coverage: list[CoverageScore] = Field(default_factory=list)

    coverage_assessment: str = Field(
        default="", description="Are all important subquestions addressed?"
    )
    evidence_quality_assessment: str = Field(
        default="", description="Are major claims adequately sourced?"
    )
    source_diversity_assessment: str = Field(
        default="",
        description="Is the synthesis over-dependent on few papers or groups?",
    )
    recency_assessment: str = Field(default="", description="Is relevant recent work missing?")
    saturation_assessment: str = Field(
        default="",
        description="Are additional searches still yielding new relevant evidence?",
    )

    weak_subquestions: list[str] = Field(
        default_factory=list, description="Ids with insufficient or weak evidence."
    )
    knowledge_gaps: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims in the synthesis that the evidence does not carry.",
    )
    contradictions: list[ObservedContradiction] = Field(default_factory=list)

    recommended_intents: list[RecommendedIntent] = Field(
        default_factory=list,
        description="New retrieval goals that would close the gaps you identified. "
        "Empty when status is 'sufficient'.",
    )
    plan_edits: list[ProposedPlanEdit] = Field(
        default_factory=list,
        description="Changes to the research-question tree justified by what was found.",
    )
    snowball_requests: list[SnowballRequest] = Field(default_factory=list)
