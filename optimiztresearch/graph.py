"""The research graph.

Topology, with ``⇉`` marking a ``Send`` fan-out:

    START
      │
      ▼
    plan ──────────────────────────────┐
      │                                │ (the loop re-enters here)
      ▼                                │
    generate_intents ◄─────────────────┤
      │                                │
      ▼                                │
    generate_queries                   │
      │                                │
      ▼                                │
    search_all                         │
      │                                │
      ▼                                │
    canonicalize_documents             │
      │                                │
      ▼                                │
    screen_and_rank ◄──────┐           │
      │                    │           │
      ▼                    │           │
    plan_snowball          │           │
      │                    │           │
      ├── seeds ──► expand_snowball ───┘ (back to screening)
      │                                  … until saturated or out of budget
      ▼
    ⇉ read_document ⇉           (one branch per selected document)
      │
      ▼
    synthesize_dispatch
      │
      ▼
    ⇉ synthesize_subquestion ⇉  (one branch per subquestion with evidence)
      │
      ▼
    global_synthesis
      │
      ▼
    critique_research
      │
      ├── continue ──► apply_critique ──► generate_intents   (next iteration)
      │
      └── sufficient / budget spent ──► write_report ──► END

Two loops, with different jobs. The inner one (snowball ⟲ screening) expands
the citation neighbourhood of what has already been found and stops on a
saturation criterion. The outer one (critique ⟲ planning) is the adaptive
research loop: it can rewrite the research plan itself, which is what makes
the decomposition dynamic rather than fixed.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from .nodes import (
    apply_critique,
    canonicalize_documents,
    critique_research,
    dispatch_readers,
    dispatch_synthesizers,
    expand_snowball,
    generate_intents,
    generate_queries,
    global_synthesis,
    plan_research,
    plan_snowball,
    read_document,
    screen_and_rank,
    search_all,
    should_continue,
    should_snowball,
    synthesize_subquestion,
    write_report,
)
from .state import ResearchState

logger = logging.getLogger(__name__)


async def synthesize_dispatch(state: ResearchState) -> dict:
    """Join point between the reader fan-out and the synthesis fan-out.

    LangGraph needs a concrete node for the readers to converge on before the
    next fan-out begins; this is also the natural place to mark subquestions
    whose evidence has now been gathered.
    """
    read_total = sum(
        1 for doc in (state.get("documents") or {}).values() if doc.status == "read"
    )
    return {
        "log": [
            f"reading complete: {read_total} documents read in total, "
            f"{len(state.get('evidence') or [])} evidence items collected"
        ]
    }


def build_graph(checkpointer=None):
    """Compile the research graph.

    Pass a checkpointer (e.g. ``langgraph.checkpoint.memory.MemorySaver``) to
    make runs resumable and to allow inspection of intermediate state.
    """
    builder = StateGraph(ResearchState)

    # --- planning and retrieval ---
    builder.add_node("plan", plan_research)
    builder.add_node("generate_intents", generate_intents)
    builder.add_node("generate_queries", generate_queries)
    builder.add_node("search_all", search_all)
    builder.add_node("canonicalize_documents", canonicalize_documents)
    builder.add_node("screen_and_rank", screen_and_rank)

    # --- snowballing ---
    builder.add_node("plan_snowball", plan_snowball)
    builder.add_node("expand_snowball", expand_snowball)

    # --- reading and synthesis ---
    builder.add_node("read_document", read_document)
    builder.add_node("synthesize_dispatch", synthesize_dispatch)
    builder.add_node("synthesize_subquestion", synthesize_subquestion)
    builder.add_node("global_synthesis", global_synthesis)

    # --- reflection ---
    builder.add_node("critique_research", critique_research)
    builder.add_node("apply_critique", apply_critique)
    builder.add_node("write_report", write_report)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "generate_intents")
    builder.add_edge("generate_intents", "generate_queries")
    builder.add_edge("generate_queries", "search_all")
    builder.add_edge("search_all", "canonicalize_documents")
    builder.add_edge("canonicalize_documents", "screen_and_rank")
    builder.add_edge("screen_and_rank", "plan_snowball")

    # Snowball rounds loop back through screening, so citation-discovered
    # documents are screened and ranked on the same terms as searched ones.
    builder.add_conditional_edges(
        "plan_snowball",
        should_snowball,
        ["expand_snowball", "prepare_reading"],
    )
    builder.add_edge("expand_snowball", "screen_and_rank")

    # Reader fan-out. `dispatch_readers` is a routing function returning
    # Send objects, and LangGraph needs a concrete node to hang that
    # conditional edge on -- hence this no-op join, which is also where the
    # snowball loop exits to.
    builder.add_node("prepare_reading", _prepare_reading)
    builder.add_conditional_edges(
        "prepare_reading",
        dispatch_readers,
        ["read_document", "synthesize_dispatch"],
    )
    builder.add_edge("read_document", "synthesize_dispatch")

    # Synthesis fan-out.
    builder.add_conditional_edges(
        "synthesize_dispatch",
        dispatch_synthesizers,
        ["synthesize_subquestion", "global_synthesis"],
    )
    builder.add_edge("synthesize_subquestion", "global_synthesis")
    builder.add_edge("global_synthesis", "critique_research")

    # The adaptive research loop.
    builder.add_conditional_edges(
        "critique_research",
        should_continue,
        ["apply_critique", "write_report"],
    )
    builder.add_edge("apply_critique", "generate_intents")
    builder.add_edge("write_report", END)

    return builder.compile(checkpointer=checkpointer)


async def _prepare_reading(_state: ResearchState) -> dict:
    """Join node that hosts the reader fan-out edge. Writes nothing."""
    return {}
