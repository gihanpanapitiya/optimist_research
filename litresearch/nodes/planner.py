"""Research planning: decompose the main question into subquestions.

Runs once at the start. Thereafter the plan is mutated by the critic
(:mod:`litresearch.nodes.critic`), because the methodology treats the initial
tree as a starting hypothesis rather than an authority.
"""

from __future__ import annotations

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import get_settings
from ..llm import ainvoke_structured
from ..schemas import PlanOutput
from ..sources.registry import active_sources
from ..state import BudgetUsage, ResearchState, Subquestion
from .common import SOURCE_DESCRIPTIONS, current_year, log_line

logger = logging.getLogger(__name__)

SYSTEM = """You are the research planner of a deep literature research system.

Given a research question, you decide what must be established in order to \
answer it, and express that as a set of research subquestions.

A subquestion states WHAT WE NEED TO KNOW. It is a knowledge gap, phrased the \
way you would ask a knowledgeable person. It says nothing about where the \
answer lives, what documents would contain it, or what search terms would \
find it -- later stages handle that.

What makes a good decomposition:

- Each subquestion is a distinct piece of knowledge the main question depends \
on. If two subquestions would be answered by the same evidence, merge them.
- Together they cover the main question. A reader who had all the answers \
should not be left asking "but what about ...?" about anything central.
- Each is specific enough that you could recognise a document as relevant or \
irrelevant to it.
- They follow the structure of the subject matter, not a generic template. \
Resist the urge to append a boilerplate "what are the limitations?" or "how is \
it evaluated?" unless that genuinely is a distinct knowledge requirement here.
- Use hierarchy (RQ4 with children RQ4.1, RQ4.2) only when a subquestion is \
genuinely compound and its parts need separate evidence. Do not nest for \
tidiness.

Minimality is a real constraint. You are producing the MINIMUM set of \
meaningful subquestions that gives comprehensive coverage -- not the maximum \
set you can defend. Every extra subquestion spends search and reading budget \
that a more central one could have used. Prefer {max_subquestions} or fewer; \
fewer is better when the question is narrow.

Priority reflects how central a subquestion is to the main question: 0.9+ for \
load-bearing, around 0.5 for useful context, below 0.3 for peripheral.

State your interpretation of the question explicitly, including any scope \
decision you had to make. If the question is ambiguous, resolve it the way a \
careful researcher would and say so rather than hedging across readings."""

HUMAN = """Research question:

{question}
{scope}
Today's date: {today}. Available literature sources:
{sources}

Decompose this question for literature research."""


async def plan_research(state: ResearchState) -> dict:
    """Produce the initial research-question tree."""
    settings = get_settings()
    question = state["main_question"]

    try:
        source_names = list(active_sources().keys())
    except RuntimeError as exc:
        # No usable source: fail loudly here rather than returning an empty
        # report after burning planning tokens.
        raise RuntimeError(f"Cannot start research: {exc}") from exc

    source_menu = "\n".join(
        f"- {name}: {SOURCE_DESCRIPTIONS.get(name, '')}" for name in source_names
    )
    scope = (
        f"\nScope notes from the requester:\n{state['scope_notes']}\n"
        if state.get("scope_notes")
        else ""
    )

    messages = [
        SystemMessage(SYSTEM.format(max_subquestions=settings.budget.max_subquestions)),
        HumanMessage(
            HUMAN.format(
                question=question,
                scope=scope,
                today=f"{current_year()}",
                sources=source_menu,
            )
        ),
    ]

    plan = await ainvoke_structured("planner", PlanOutput, messages)
    if plan is None or not plan.subquestions:
        # Degrade to treating the main question as a single subquestion: a
        # shallow run still beats no run.
        logger.warning("planner produced no decomposition; falling back to single RQ")
        fallback = Subquestion(
            id="RQ1",
            question=question,
            rationale="Planner failed; investigating the main question directly.",
            priority=1.0,
        )
        return {
            "subquestions": {"RQ1": fallback},
            "plan_rationale": "Planning failed; using the main question directly.",
            "iteration": 1,
            "budget": BudgetUsage(research_iterations=1, llm_calls=1),
            "log": [log_line("planning failed, using single-subquestion fallback")],
        }

    subquestions = _materialize(plan, settings.budget.max_subquestions)

    rationale = plan.rationale
    if plan.interpretation:
        rationale = f"**Interpretation.** {plan.interpretation}\n\n{rationale}"

    return {
        "subquestions": subquestions,
        "plan_rationale": rationale,
        "iteration": 1,
        "budget": BudgetUsage(research_iterations=1, llm_calls=1),
        "log": [
            log_line(
                f"planned {len(subquestions)} subquestions: "
                f"{', '.join(sorted(subquestions))}"
            )
        ],
    }


def _materialize(plan: PlanOutput, max_subquestions: int) -> dict[str, Subquestion]:
    """Turn planner output into state objects, normalizing ids and links.

    Models occasionally emit duplicate ids, dangling parents or more
    subquestions than requested, none of which should corrupt the state.
    """
    # Keep the highest-priority subquestions if the planner overshot, but
    # never drop a parent whose child survives.
    ordered = sorted(plan.subquestions, key=lambda sq: -sq.priority)
    kept: list = []
    for item in ordered:
        if len(kept) < max_subquestions:
            kept.append(item)
    kept_ids = {item.id for item in kept}
    for item in plan.subquestions:
        if item.parent_id in kept_ids and item.id not in kept_ids:
            kept.append(item)
            kept_ids.add(item.id)

    subquestions: dict[str, Subquestion] = {}
    used_ids: set[str] = set()
    for index, item in enumerate(sorted(kept, key=lambda sq: _id_sort_key(sq.id)), start=1):
        sq_id = _normalize_id(item.id, index, used_ids)
        used_ids.add(sq_id)
        subquestions[sq_id] = Subquestion(
            id=sq_id,
            question=item.question.strip(),
            rationale=item.rationale.strip(),
            priority=max(0.0, min(1.0, item.priority)),
            parent_id=item.parent_id,
            status="unexplored",
            created_in_iteration=1,
            origin="planner",
        )

    # Drop parent links that point nowhere, then wire up children both ways.
    for sq in subquestions.values():
        if sq.parent_id and sq.parent_id not in subquestions:
            sq.parent_id = None
    for sq in subquestions.values():
        if sq.parent_id:
            parent = subquestions[sq.parent_id]
            if sq.id not in parent.children:
                parent.children.append(sq.id)

    return subquestions


def _normalize_id(raw: str, index: int, used: set[str]) -> str:
    """Coerce an id into RQ<n>[.<n>...] form and guarantee uniqueness."""
    cleaned = raw.strip().upper().replace(" ", "")
    if not re.fullmatch(r"RQ\d+(\.\d+)*", cleaned):
        digits = re.findall(r"\d+", cleaned)
        cleaned = "RQ" + ".".join(digits) if digits else f"RQ{index}"
    while cleaned in used:
        cleaned = f"{cleaned}b"
    return cleaned


def _id_sort_key(raw: str) -> list[int]:
    return [int(p) for p in re.findall(r"\d+", raw)] or [999]
