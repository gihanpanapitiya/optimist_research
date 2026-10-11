"""Persistent research corpus on disk.

Mirrors the layout in the methodology, so a run leaves behind an artifact a
human can actually read and audit:

    research/<slug>/
        research_plan.md
        research_state.json
        titles.md
        sources/doc_0001/{metadata.json,notes.md,paper.pdf}
        subtopics/<subquestion>.md
        evidence/{claims.md,contradictions.md}
        final_report.md

``research_state.json`` is the complete state and is what a resumed run
loads; everything else is a human-readable projection of it.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT
from ..state import (
    BudgetUsage,
    Contradiction,
    Critique,
    Document,
    DocumentReading,
    Evidence,
    ResearchState,
    SearchIntent,
    SnowballRound,
    Subquestion,
)
from .evidence import cluster_claims, group_by_subquestion

logger = logging.getLogger(__name__)


def slugify(text: str, max_length: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_length].rstrip("-") or "research"


class ResearchCorpus:
    """Filesystem-backed store for one research run."""

    def __init__(self, root: str | Path, question: str = "") -> None:
        # Resolved immediately so every message names an unambiguous
        # destination. A relative path is interpreted against the working
        # directory, which in a notebook is often not the project.
        base = Path(root).expanduser().resolve()
        # A bare output dir gets a per-question subdirectory so that successive
        # runs do not overwrite one another.
        self.root = base / slugify(question) if question else base
        self.sources_dir = self.root / "sources"
        self.subtopics_dir = self.root / "subtopics"
        self.evidence_dir = self.root / "evidence"

    def ensure(self) -> None:
        """Create the corpus directories, explaining a failure usefully.

        The bare pathlib error for this is a wall of traceback ending in
        "Read-only file system: 'research'", which does not reveal that the
        relative path resolved somewhere unexpected.
        """
        for directory in (self.root, self.sources_dir, self.subtopics_dir, self.evidence_dir):
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise OSError(
                    f"Cannot create the research output directory {directory} "
                    f"({exc.strerror}). The output path was resolved against the "
                    f"current working directory, {Path.cwd()}. Pass an absolute "
                    f"path instead, e.g. "
                    f'run_research(..., output_dir="{PROJECT_ROOT / "research"}"), '
                    f"or set LITRESEARCH_OUTPUT_DIR."
                ) from exc

    # -- state round-trip --------------------------------------------------

    @property
    def state_path(self) -> Path:
        return self.root / "research_state.json"

    def save_state(self, state: ResearchState) -> Path:
        self.ensure()
        payload = _state_to_jsonable(state)
        self.state_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return self.state_path

    def load_state(self) -> ResearchState | None:
        if not self.state_path.exists():
            return None
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("could not read %s (%s)", self.state_path, exc)
            return None
        return _state_from_jsonable(payload)

    # -- human-readable projections ----------------------------------------

    def write_all(self, state: ResearchState) -> None:
        """Write every projection. Called at the end of each iteration."""
        self.ensure()
        self.save_state(state)
        self.write_research_plan(state)
        self.write_documents(state)
        self.write_titles_index(state)
        self.write_subtopics(state)
        self.write_evidence(state)
        if state.get("final_report"):
            self.write_report(state["final_report"])

    def write_research_plan(self, state: ResearchState) -> Path:
        subquestions: dict[str, Subquestion] = state.get("subquestions") or {}
        intents: dict[str, SearchIntent] = state.get("intents") or {}
        budget: BudgetUsage = state.get("budget") or BudgetUsage()

        lines = [
            "# Research Plan",
            "",
            f"**Main question:** {state.get('main_question', '')}",
            "",
            f"- Iterations completed: {budget.research_iterations}",
            f"- Search operations: {budget.search_operations}",
            f"- Documents discovered: {len(state.get('documents') or {})}",
            f"- Documents read: {budget.documents_read}",
            f"- Evidence items: {len(state.get('evidence') or [])}",
            "",
        ]
        if state.get("plan_rationale"):
            lines += ["## Planning rationale", "", state["plan_rationale"], ""]

        lines += ["## Research questions", ""]
        for sq in _ordered_subquestions(subquestions):
            indent = "  " * sq.depth
            lines.append(
                f"{indent}- **{sq.id}** ({sq.status}, priority {sq.priority:.2f}, "
                f"coverage {sq.coverage:.2f}) {sq.question}"
            )
            if sq.rationale:
                lines.append(f"{indent}  - _why:_ {sq.rationale}")
            # Resolve by the intent's own subquestion_id as well as through
            # the back-link, so an intent is never silently dropped from the
            # plan because one side of the pair was not updated.
            linked = [intents[i] for i in sq.intent_ids if i in intents]
            owned = [i for i in intents.values() if i.subquestion_id == sq.id]
            for intent in {i.id: i for i in [*linked, *owned]}.values():
                lines.append(
                    f"{indent}  - _intent {intent.id}_ ({intent.status}, "
                    f"{intent.relevant_found} relevant): {intent.intent}"
                )
            for gap in sq.knowledge_gaps:
                lines.append(f"{indent}  - _gap:_ {gap}")

        path = self.root / "research_plan.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def write_documents(self, state: ResearchState) -> None:
        documents: dict[str, Document] = state.get("documents") or {}
        readings: dict[str, DocumentReading] = state.get("readings") or {}
        evidence_by_doc: dict[str, list[Evidence]] = {}
        for item in state.get("evidence") or []:
            evidence_by_doc.setdefault(item.doc_id, []).append(item)

        for doc_id, doc in documents.items():
            # Only materialize directories for documents we actually read;
            # otherwise a 300-candidate run creates 300 near-empty folders.
            if doc.status != "read" and doc_id not in readings:
                continue
            doc_dir = self.sources_dir / doc_id
            doc_dir.mkdir(parents=True, exist_ok=True)
            (doc_dir / "metadata.json").write_text(
                doc.model_dump_json(indent=2), encoding="utf-8"
            )
            (doc_dir / "notes.md").write_text(
                _document_notes(doc, readings.get(doc_id), evidence_by_doc.get(doc_id, [])),
                encoding="utf-8",
            )

    def write_titles_index(self, state: ResearchState) -> Path:
        documents: dict[str, Document] = state.get("documents") or {}
        ordered = sorted(
            documents.values(),
            key=lambda d: (-d.rank_score, -(d.year or 0), d.title.lower()),
        )
        lines = [
            "# Retrieved documents",
            "",
            f"{len(ordered)} canonical documents "
            f"({sum(1 for d in ordered if d.status == 'read')} read).",
            "",
            "| id | year | title | sources | rel | status |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for doc in ordered:
            title = doc.title.replace("|", "\\|")
            lines.append(
                f"| {doc.doc_id} | {doc.year or '?'} | {title} | "
                f"{', '.join(doc.sources)} | {doc.max_relevance():.2f} | {doc.status} |"
            )
        path = self.root / "titles.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def write_subtopics(self, state: ResearchState) -> None:
        subquestions: dict[str, Subquestion] = state.get("subquestions") or {}
        by_sq = group_by_subquestion(state.get("evidence") or [])
        documents: dict[str, Document] = state.get("documents") or {}

        for sq in subquestions.values():
            if not sq.synthesis and sq.id not in by_sq:
                continue
            lines = [
                f"# {sq.id}: {sq.question}",
                "",
                f"Status: {sq.status} · coverage {sq.coverage:.2f} · "
                f"{sq.documents_read} documents read",
                "",
            ]
            if sq.synthesis:
                lines += ["## Synthesis", "", sq.synthesis, ""]
            if sq.knowledge_gaps:
                lines += ["## Open gaps", ""]
                lines += [f"- {gap}" for gap in sq.knowledge_gaps]
                lines.append("")

            clusters = cluster_claims(by_sq.get(sq.id, []), sq.id)
            if clusters:
                lines += ["## Evidence", ""]
                for cluster in clusters:
                    cites = ", ".join(
                        documents[d].citation_key for d in cluster.doc_ids if d in documents
                    )
                    marker = " ⚠ weakly supported" if cluster.is_weakly_supported else ""
                    lines.append(f"- **{cluster.claim}** [{cites}]{marker}")
                    for item in cluster.evidence:
                        where = f" ({item.location})" if item.location else ""
                        lines.append(
                            f"  - {item.doc_id}{where}, {item.evidence_type}, "
                            f"strength {item.strength:.2f}: {item.support}"
                        )
                lines.append("")

            path = self.subtopics_dir / f"{slugify(sq.id + '-' + sq.question, 70)}.md"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def write_evidence(self, state: ResearchState) -> None:
        evidence: list[Evidence] = state.get("evidence") or []
        documents: dict[str, Document] = state.get("documents") or {}
        subquestions: dict[str, Subquestion] = state.get("subquestions") or {}

        lines = ["# Evidence store", "", f"{len(evidence)} evidence items.", ""]
        for sq in _ordered_subquestions(subquestions):
            clusters = cluster_claims(evidence, sq.id)
            if not clusters:
                continue
            lines += [f"## {sq.id}: {sq.question}", ""]
            for cluster in clusters:
                lines.append(
                    f"### {cluster.claim}"
                )
                lines.append("")
                lines.append(
                    f"Support: {cluster.support_count} document(s), "
                    f"mean strength {cluster.mean_strength:.2f}, "
                    f"empirical: {'yes' if cluster.has_empirical_support else 'no'}"
                )
                lines.append("")
                for item in cluster.evidence:
                    doc = documents.get(item.doc_id)
                    label = doc.reference_label() if doc else item.doc_id
                    where = f" — {item.location}" if item.location else ""
                    lines.append(f"- [{item.evidence_type}] {item.support}{where}")
                    lines.append(f"  - Source: {label}")
                    if item.verbatim_quote:
                        lines.append(f'  - Quote: "{item.verbatim_quote}"')
                lines.append("")
        (self.evidence_dir / "claims.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

        contradictions: list[Contradiction] = state.get("contradictions") or []
        clines = ["# Contradictions", ""]
        if not contradictions:
            clines.append("None recorded.")
        for con in contradictions:
            clines.append(f"## {con.id} ({con.status})")
            clines.append("")
            clines.append(con.description)
            if con.doc_ids:
                refs = ", ".join(
                    documents[d].citation_key for d in con.doc_ids if d in documents
                )
                clines.append("")
                clines.append(f"Involved documents: {refs}")
            if con.resolution:
                clines.append("")
                clines.append(f"Resolution: {con.resolution}")
            clines.append("")
        (self.evidence_dir / "contradictions.md").write_text(
            "\n".join(clines) + "\n", encoding="utf-8"
        )

    def write_report(self, report: str) -> Path:
        self.ensure()
        path = self.root / "final_report.md"
        path.write_text(report, encoding="utf-8")
        return path

    def pdf_path(self, doc_id: str) -> Path:
        directory = self.sources_dir / doc_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory / "paper.pdf"


# ---------------------------------------------------------------------------
# State serialization
#
# The state holds pydantic models inside plain dicts and lists, so round-trip
# has to be explicit about which fields hold which model.
# ---------------------------------------------------------------------------

_MODEL_DICT_FIELDS = {
    "subquestions": Subquestion,
    "intents": SearchIntent,
    "readings": DocumentReading,
    "documents": Document,
}

_MODEL_LIST_FIELDS = {
    "evidence": Evidence,
    "contradictions": Contradiction,
    "critique_history": Critique,
    # Run-long, not scratch: the snowball round cap and the saturation
    # criterion are computed from it, so dropping it on save handed a
    # resumed run a fresh snowball budget.
    "snowball_history": SnowballRound,
}


def _state_to_jsonable(state: ResearchState) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in state.items():
        if key in _MODEL_DICT_FIELDS:
            payload[key] = {k: v.model_dump(mode="json") for k, v in (value or {}).items()}
        elif key in _MODEL_LIST_FIELDS:
            payload[key] = [v.model_dump(mode="json") for v in (value or [])]
        elif key == "budget":
            payload[key] = (value or BudgetUsage()).model_dump(mode="json")
        elif key == "critique":
            payload[key] = value.model_dump(mode="json") if value else None
        elif key in ("pending_hits", "query_plans"):
            # Per-iteration scratch; not worth persisting.
            continue
        else:
            payload[key] = value
    return payload


def _state_from_jsonable(payload: dict[str, Any]) -> ResearchState:
    state: dict[str, Any] = {}
    for key, value in payload.items():
        if key in _MODEL_DICT_FIELDS:
            model = _MODEL_DICT_FIELDS[key]
            state[key] = {k: model.model_validate(v) for k, v in (value or {}).items()}
        elif key in _MODEL_LIST_FIELDS:
            model = _MODEL_LIST_FIELDS[key]
            state[key] = [model.model_validate(v) for v in (value or [])]
        elif key == "budget":
            state[key] = BudgetUsage.model_validate(value or {})
        elif key == "critique":
            state[key] = Critique.model_validate(value) if value else None
        else:
            state[key] = value
    state.setdefault("pending_hits", [])
    state.setdefault("query_plans", [])
    state.setdefault("snowball_history", [])
    state.setdefault("snowball_expanded", [])
    return state  # type: ignore[return-value]


def _ordered_subquestions(subquestions: dict[str, Subquestion]) -> list[Subquestion]:
    """Sort by id so that RQ2 precedes RQ2.1 precedes RQ10."""

    def sort_key(sq: Subquestion) -> list:
        parts = re.findall(r"\d+", sq.id)
        return [int(p) for p in parts] or [999]

    return sorted(subquestions.values(), key=sort_key)


def _document_notes(
    doc: Document, reading: DocumentReading | None, evidence: list[Evidence]
) -> str:
    lines = [f"# {doc.title}", ""]
    if doc.authors:
        lines.append(f"**Authors:** {', '.join(doc.authors[:12])}")
    lines.append(f"**Year:** {doc.year or 'unknown'}  ")
    lines.append(f"**Venue:** {doc.venue or 'unknown'}  ")
    if doc.doi:
        lines.append(f"**DOI:** https://doi.org/{doc.doi}  ")
    if doc.url:
        lines.append(f"**URL:** {doc.url}  ")
    citations = doc.citation_count if doc.citation_count is not None else "?"
    lines.append(f"**Citations:** {citations}  ")
    lines.append(f"**Found via:** {', '.join(doc.sources)}")
    lines.append("")

    lines += ["## How this document was discovered", ""]
    for prov in doc.discovered_by:
        lines.append(
            f"- {prov.subquestion_id}/{prov.intent_id} via {prov.source} "
            f"({prov.via}, rank {prov.rank}): `{prov.query}`"
        )
    lines.append("")

    if doc.relevance:
        lines += ["## Screened relevance", ""]
        for sq_id, score in sorted(doc.relevance.items(), key=lambda kv: -kv[1]):
            lines.append(f"- {sq_id}: {score:.2f}")
        if doc.screening_note:
            lines += ["", f"_{doc.screening_note}_"]
        lines.append("")

    if reading:
        lines += ["## Reading", ""]
        if reading.main_contribution:
            lines += ["### Main contribution", "", reading.main_contribution, ""]
        if reading.method:
            lines += ["### Method", "", reading.method, ""]
        if reading.experimental_setup:
            lines += ["### Experimental setup", "", reading.experimental_setup, ""]
        if reading.major_findings:
            lines += ["### Major findings", ""]
            lines += [f"- {f}" for f in reading.major_findings]
            lines.append("")
        if reading.limitations:
            lines += ["### Limitations", ""]
            lines += [f"- {limit}" for limit in reading.limitations]
            lines.append("")
        lines.append(
            f"_Read from {'full text' if reading.used_full_text else 'abstract and metadata'}._"
        )
        lines.append("")

    if evidence:
        lines += ["## Extracted evidence", ""]
        for item in evidence:
            where = f" ({item.location})" if item.location else ""
            lines.append(
                f"- **{item.subquestion_id}** — {item.claim}{where} "
                f"[{item.evidence_type}, strength {item.strength:.2f}]"
            )
            lines.append(f"  - {item.support}")
        lines.append("")

    if doc.abstract:
        lines += ["## Abstract", "", doc.abstract, ""]

    return "\n".join(lines) + "\n"
