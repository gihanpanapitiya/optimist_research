"""End-to-end graph tests with a fake model and fake sources.

These exercise the machinery, not model quality: that state flows through
every stage, fan-out converges, the adaptive loop iterates and terminates,
budgets are honoured, and the corpus lands on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

from forestresearch.config import get_settings
from forestresearch.graph import build_graph
from forestresearch.runner import run_research, summarize_run
from forestresearch.schemas import (
    CoverageScore,
    CritiqueOutput,
    ProposedPlanEdit,
    RecommendedIntent,
)
from forestresearch.state import new_state

QUESTION = "How do AI agents plan and recover from failures?"


async def run(question: str = QUESTION) -> dict:
    graph = build_graph()
    return await graph.ainvoke(new_state(question), {"recursion_limit": 100})


class TestHappyPath:
    async def test_single_iteration_run_completes(self, wired):
        final = await run()

        assert final["final_report"]
        assert final["terminated_because"]
        assert sorted(final["subquestions"]) == ["RQ1", "RQ2"]
        assert final["intents"]
        assert final["documents"]
        assert final["evidence"]
        assert final["global_synthesis"]
        assert final["critique"].status == "sufficient"

    async def test_every_stage_ran(self, wired):
        await run()
        kinds = set(wired.calls)
        assert kinds == {
            "PlanOutput",
            "IntentOutput",
            "QueryPlanOutput",
            "ScreeningOutput",
            "ReadingOutput",
            "SubquestionSynthesisOutput",
            "GlobalSynthesisOutput",
            "CritiqueOutput",
            "prose",
        }

    async def test_evidence_is_attributed_to_real_documents(self, wired):
        final = await run()
        doc_ids = set(final["documents"])
        for item in final["evidence"]:
            assert item.doc_id in doc_ids
            assert item.subquestion_id in final["subquestions"]
            assert item.claim and item.support

    async def test_readers_fanned_out_in_parallel(self, wired):
        final = await run()
        read = [d for d in final["documents"].values() if d.status == "read"]
        assert len(read) > 1
        # One reading record per document read, from separate branches.
        assert set(final["readings"]) == {d.doc_id for d in read}

    async def test_budget_counts_each_reader(self, wired):
        """The fan-in must sum reader deltas, not collapse them."""
        final = await run()
        read = sum(1 for d in final["documents"].values() if d.status == "read")
        assert final["budget"].documents_read == read
        assert final["budget"].search_operations > 0
        assert final["budget"].llm_calls >= len(wired.calls) - 1

    async def test_report_cites_only_retrieved_documents(self, wired):
        final = await run()
        report = final["final_report"]
        assert "## References" in report
        import re

        for doc_id in set(re.findall(r"doc_\d{4}", report)):
            assert doc_id in final["documents"]

    async def test_subquestions_get_syntheses_and_coverage(self, wired):
        final = await run()
        for sq in final["subquestions"].values():
            assert sq.synthesis
            assert sq.coverage > 0
            assert sq.documents_read > 0


class TestAdaptiveLoop:
    async def test_critic_can_request_another_iteration(self, wired):
        wired.critique_queue = [
            CritiqueOutput(
                status="continue",
                reasoning="RQ2 is thin.",
                overall_coverage=0.4,
                subquestion_coverage=[CoverageScore(subquestion_id="RQ2", coverage=0.3)],
                weak_subquestions=["RQ2"],
                knowledge_gaps=["No evidence on long-horizon replanning."],
                recommended_intents=[
                    RecommendedIntent(
                        subquestion_id="RQ2",
                        intent="Find studies of replanning over long horizons.",
                        rationale="Closes the gap.",
                    )
                ],
            )
        ]
        final = await run()

        assert final["budget"].research_iterations == 2
        assert final["iteration"] == 2
        assert len(final["critique_history"]) == 2
        # The critic's intent was queued and actually searched.
        critic_intents = [i for i in final["intents"].values() if i.origin == "critic"]
        assert critic_intents
        assert all(i.status != "pending" for i in critic_intents)
        assert final["critique"].status == "sufficient"

    async def test_critic_can_add_a_subquestion(self, wired):
        wired.critique_queue = [
            CritiqueOutput(
                status="continue",
                reasoning="World models are missing.",
                overall_coverage=0.5,
                plan_edits=[
                    ProposedPlanEdit(
                        action="add",
                        question="How is world-model-based planning used by agents?",
                        rationale="Revealed by the literature.",
                        priority=0.8,
                    )
                ],
                recommended_intents=[
                    RecommendedIntent(
                        subquestion_id="RQ3",
                        intent="Find work on world-model-based planning.",
                    )
                ],
            )
        ]
        final = await run()

        assert "RQ3" in final["subquestions"]
        added = final["subquestions"]["RQ3"]
        assert added.origin == "critic"
        assert added.created_in_iteration == 2
        # A newly added subquestion must get the existing corpus re-screened
        # against it, not just new searches.
        assert any("RQ3" in d.relevance for d in final["documents"].values())

    async def test_critic_can_split_a_subquestion(self, wired):
        wired.critique_queue = [
            CritiqueOutput(
                status="continue",
                reasoning="RQ2 is compound.",
                overall_coverage=0.5,
                plan_edits=[
                    ProposedPlanEdit(
                        action="split",
                        subquestion_id="RQ2",
                        children=[
                            "How are planning failures detected?",
                            "How are alternative plans generated?",
                        ],
                        rationale="Separate evidence needed.",
                    )
                ],
            )
        ]
        final = await run()

        assert final["subquestions"]["RQ2"].status == "superseded"
        assert "RQ2.1" in final["subquestions"]
        assert "RQ2.2" in final["subquestions"]
        assert final["subquestions"]["RQ2.1"].parent_id == "RQ2"

    async def test_critic_can_deprioritize(self, wired):
        wired.critique_queue = [
            CritiqueOutput(
                status="continue",
                reasoning="RQ2 is a dead end.",
                overall_coverage=0.6,
                plan_edits=[
                    ProposedPlanEdit(
                        action="deprioritize", subquestion_id="RQ2", rationale="Marginal."
                    )
                ],
                recommended_intents=[
                    RecommendedIntent(subquestion_id="RQ1", intent="Find more on RQ1.")
                ],
            )
        ]
        final = await run()
        assert final["subquestions"]["RQ2"].status == "deprioritized"

    async def test_continue_with_no_actions_terminates(self, wired):
        """A critic that wants more but names nothing must not spin the loop."""
        wired.critique_queue = [
            CritiqueOutput(
                status="continue", reasoning="Vaguely unsatisfied.", overall_coverage=0.5
            )
        ]
        final = await run()
        assert final["budget"].research_iterations == 1
        assert "no actionable next steps" in final["terminated_because"]

    async def test_iteration_limit_stops_a_persistent_critic(self, wired, monkeypatch):
        monkeypatch.setenv("FORESTRESEARCH_BUDGET__MAX_RESEARCH_ITERATIONS", "2")
        get_settings(refresh=True)
        wired.critique_queue = [
            CritiqueOutput(
                status="continue",
                reasoning="Never satisfied.",
                overall_coverage=0.3,
                weak_subquestions=["RQ1"],
                recommended_intents=[
                    RecommendedIntent(subquestion_id="RQ1", intent="Find yet more.")
                ],
            )
            for _ in range(10)
        ]
        final = await run()
        assert final["budget"].research_iterations == 2
        assert "iteration limit" in final["terminated_because"]


class TestFailureHandling:
    async def test_planner_failure_falls_back_to_one_subquestion(self, wired):
        wired.failures = {"PlanOutput"}
        final = await run()
        assert list(final["subquestions"]) == ["RQ1"]
        assert final["subquestions"]["RQ1"].question == QUESTION
        assert final["final_report"]

    async def test_a_failed_reader_does_not_abort_the_run(self, wired):
        wired.failures = {"ReadingOutput"}
        final = await run()
        assert final["evidence"] == []
        # Every document that was dispatched to a reader is marked failed.
        assert any(d.status == "failed" for d in final["documents"].values())
        assert all(d.status != "read" for d in final["documents"].values())
        # No evidence means the diagnostic report, not a crash.
        assert "No evidence was collected" in final["final_report"]

    async def test_screening_failure_degrades_without_crashing(self, wired):
        wired.failures = {"ScreeningOutput"}
        final = await run()
        # Nothing clears the relevance threshold, so nothing is read.
        assert final["budget"].documents_read == 0
        assert final["final_report"]

    async def test_critic_failure_terminates_cleanly(self, wired):
        wired.failures = {"CritiqueOutput"}
        final = await run()
        assert final["critique"].status == "sufficient"
        assert "Critic call failed" in final["critique"].reasoning
        assert final["final_report"]

    async def test_intent_failure_falls_back_to_the_subquestion(self, wired):
        wired.failures = {"IntentOutput"}
        final = await run()
        assert final["intents"]
        assert any("Find research that addresses" in i.intent for i in final["intents"].values())
        assert final["documents"]


class TestDeduplicationInGraph:
    async def test_same_paper_from_two_intents_is_one_document(self, wired, fake_source):
        """Both subquestions search the same fake corpus; it must not double."""
        final = await run()
        searched = [d for d in final["documents"].values() if d.snowball_depth == 0]
        assert len(searched) == fake_source.hit_count
        # Each document records being discovered by more than one intent.
        multi = [d for d in final["documents"].values() if len(d.discovered_by) > 1]
        assert multi

    async def test_documents_are_shared_across_subquestions(self, wired):
        final = await run()
        shared = [d for d in final["documents"].values() if len(d.read_for) > 1]
        assert shared, "a document relevant to both subquestions should be read for both"


class TestPersistence:
    async def test_corpus_is_written_to_disk(self, wired, tmp_path):
        state = await run_research(QUESTION)
        root = Path(summarize_run(state)["corpus_root"])

        assert (root / "final_report.md").exists()
        assert (root / "research_plan.md").exists()
        assert (root / "research_state.json").exists()
        assert (root / "titles.md").exists()
        assert (root / "evidence" / "claims.md").exists()
        assert (root / "evidence" / "contradictions.md").exists()
        assert list((root / "subtopics").glob("*.md"))
        assert list((root / "sources").glob("doc_*/metadata.json"))
        assert list((root / "sources").glob("doc_*/notes.md"))

    async def test_state_round_trips(self, wired, tmp_path):
        from forestresearch.store.corpus import ResearchCorpus

        state = await run_research(QUESTION)
        corpus = ResearchCorpus(get_settings().output_dir, QUESTION)
        reloaded = corpus.load_state()

        assert reloaded is not None
        assert reloaded["main_question"] == state["main_question"]
        assert set(reloaded["documents"]) == set(state["documents"])
        assert len(reloaded["evidence"]) == len(state["evidence"])
        assert reloaded["budget"].documents_read == state["budget"].documents_read
        # Models must come back as models, not dicts.
        assert reloaded["evidence"][0].claim == state["evidence"][0].claim
        assert reloaded["subquestions"]["RQ1"].question

    async def test_saved_state_is_valid_json(self, wired):
        from forestresearch.store.corpus import ResearchCorpus

        await run_research(QUESTION)
        corpus = ResearchCorpus(get_settings().output_dir, QUESTION)
        payload = json.loads(corpus.state_path.read_text())
        assert payload["main_question"] == QUESTION
        assert "pending_hits" not in payload  # scratch channels are not persisted

    async def test_summary_reports_real_numbers(self, wired):
        state = await run_research(QUESTION)
        summary = summarize_run(state)
        assert summary["documents_read"] > 0
        assert summary["evidence_items"] > 0
        assert summary["iterations"] == 1
        assert 0 < summary["mean_coverage"] <= 1
