"""Tests for the research state and its LangGraph reducers.

The reducers are what make parallel fan-out safe, so they get tested for the
property that matters: applying concurrent updates in any order must give the
same result, and nothing a branch wrote may be lost.
"""

from __future__ import annotations

from optimiztresearch.state import (
    BudgetUsage,
    Contradiction,
    Document,
    Evidence,
    Provenance,
    SearchIntent,
    SnowballRound,
    Subquestion,
    extend_unique_contradictions,
    extend_unique_evidence,
    merge_budget,
    merge_documents,
    merge_subquestions,
    new_state,
    take_last,
)


def doc(doc_id: str, **kwargs) -> Document:
    defaults = dict(title=f"Title {doc_id}", sources=["arxiv"])
    defaults.update(kwargs)
    return Document(doc_id=doc_id, **defaults)


def ev(claim: str, doc_id: str = "doc_0001", sq: str = "RQ1", **kwargs) -> Evidence:
    return Evidence(claim=claim, support="support", doc_id=doc_id, subquestion_id=sq, **kwargs)


class TestBudgetReducer:
    def test_parallel_readers_each_increment(self):
        """Twenty readers writing documents_read=1 must advance by twenty."""
        total = BudgetUsage()
        for _ in range(20):
            total = merge_budget(total, BudgetUsage(documents_read=1, llm_calls=1))
        assert total.documents_read == 20
        assert total.llm_calls == 20

    def test_deltas_accumulate_across_stages(self):
        state = merge_budget(None, BudgetUsage(research_iterations=1))
        state = merge_budget(state, BudgetUsage(search_operations=12))
        state = merge_budget(state, BudgetUsage(documents_discovered=30))
        state = merge_budget(state, BudgetUsage(snowball_rounds=1))
        assert state.research_iterations == 1
        assert state.search_operations == 12
        assert state.documents_discovered == 30
        assert state.snowball_rounds == 1

    def test_merge_is_order_independent(self):
        a = BudgetUsage(documents_read=3, llm_calls=2)
        b = BudgetUsage(documents_read=5, search_operations=4)
        assert merge_budget(a, b).model_dump() == merge_budget(b, a).model_dump()


class TestDocumentReducer:
    def test_disjoint_writes_both_land(self):
        merged = merge_documents({"doc_0001": doc("doc_0001")}, {"doc_0002": doc("doc_0002")})
        assert sorted(merged) == ["doc_0001", "doc_0002"]

    def test_concurrent_relevance_writes_union(self):
        """Two screening batches scoring the same document must not clobber."""
        base = {"doc_0001": doc("doc_0001", relevance={"RQ1": 0.9})}
        update = {"doc_0001": doc("doc_0001", relevance={"RQ2": 0.7})}
        merged = merge_documents(base, update)
        assert merged["doc_0001"].relevance == {"RQ1": 0.9, "RQ2": 0.7}

    def test_higher_relevance_wins(self):
        base = {"doc_0001": doc("doc_0001", relevance={"RQ1": 0.4})}
        update = {"doc_0001": doc("doc_0001", relevance={"RQ1": 0.9})}
        assert merge_documents(base, update)["doc_0001"].relevance["RQ1"] == 0.9

    def test_read_status_never_regresses(self):
        """A late candidate write must not undo a completed read."""
        base = {"doc_0001": doc("doc_0001", status="read", read_for=["RQ1"])}
        update = {"doc_0001": doc("doc_0001", status="candidate")}
        merged = merge_documents(base, update)
        assert merged["doc_0001"].status == "read"
        assert merged["doc_0001"].read_for == ["RQ1"]

    def test_read_for_accumulates(self):
        base = {"doc_0001": doc("doc_0001", status="read", read_for=["RQ1"])}
        update = {"doc_0001": doc("doc_0001", status="read", read_for=["RQ3"])}
        assert merge_documents(base, update)["doc_0001"].read_for == ["RQ1", "RQ3"]

    def test_provenance_unions_without_duplicates(self):
        prov_a = Provenance(subquestion_id="RQ1", intent_id="RQ1-i1", source="arxiv", query="q")
        prov_b = Provenance(subquestion_id="RQ2", intent_id="RQ2-i1", source="openalex", query="r")
        base = {"doc_0001": doc("doc_0001", discovered_by=[prov_a])}
        update = {"doc_0001": doc("doc_0001", discovered_by=[prov_a, prov_b])}
        merged = merge_documents(base, update)
        assert len(merged["doc_0001"].discovered_by) == 2


class TestSubquestionReducer:
    def test_reader_increments_do_not_lose_synthesis(self):
        base = {"RQ1": Subquestion(id="RQ1", question="q", synthesis="A synthesis.")}
        update = {"RQ1": Subquestion(id="RQ1", question="q", documents_read=3)}
        merged = merge_subquestions(base, update)
        assert merged["RQ1"].synthesis == "A synthesis."
        assert merged["RQ1"].documents_read == 3

    def test_collections_union(self):
        base = {
            "RQ1": Subquestion(
                id="RQ1", question="q", intent_ids=["RQ1-i1"], knowledge_gaps=["a"]
            )
        }
        update = {
            "RQ1": Subquestion(
                id="RQ1", question="q", intent_ids=["RQ1-i2"], knowledge_gaps=["b"]
            )
        }
        merged = merge_subquestions(base, update)
        assert merged["RQ1"].intent_ids == ["RQ1-i1", "RQ1-i2"]
        assert merged["RQ1"].knowledge_gaps == ["a", "b"]


class TestEvidenceReducer:
    def test_identical_evidence_is_deduplicated(self):
        item = ev("A claim")
        merged = extend_unique_evidence([item], [item])
        assert len(merged) == 1

    def test_same_claim_from_two_documents_is_kept(self):
        """Corroboration is the point; these must not collapse."""
        merged = extend_unique_evidence(
            [ev("A claim", doc_id="doc_0001")], [ev("A claim", doc_id="doc_0002")]
        )
        assert len(merged) == 2

    def test_same_claim_for_two_subquestions_is_kept(self):
        merged = extend_unique_evidence([ev("A claim", sq="RQ1")], [ev("A claim", sq="RQ2")])
        assert len(merged) == 2

    def test_none_clears_the_channel(self):
        assert extend_unique_evidence([ev("A claim")], None) == []

    def test_contradictions_dedupe_by_description(self):
        c = Contradiction(description="X and Y disagree")
        assert len(extend_unique_contradictions([c], [c])) == 1


class TestTakeLast:
    def test_replaces_rather_than_appends(self):
        """Work lists must not carry over between iterations."""
        assert take_last([1, 2, 3], [4]) == [4]
        assert take_last([1, 2], []) == []


class TestSnowballYield:
    def test_yield_rate(self):
        full = SnowballRound(
            round_index=1, direction="backward", candidates_examined=40, newly_relevant=4
        )
        empty = SnowballRound(
            round_index=2, direction="forward", candidates_examined=0, newly_relevant=0
        )
        assert full.yield_rate == 0.1
        assert empty.yield_rate == 0.0


class TestDocumentHelpers:
    def test_citation_key_uses_surname_and_year(self):
        assert doc("doc_0001", authors=["Jane Van Doe"], year=2024).citation_key == "Doe2024"

    def test_survey_is_not_primary_research(self):
        assert not doc("doc_0001", title="A Survey of Planning Methods").is_primary_research

    def test_empirical_paper_is_primary_research(self):
        d = doc(
            "doc_0001",
            title="Tree Search for Agent Planning",
            abstract="We propose a method and our experiments show gains.",
            publication_types=["JournalArticle"],
        )
        assert d.is_primary_research

    def test_best_subquestions_applies_threshold(self):
        d = doc("doc_0001", relevance={"RQ1": 0.9, "RQ2": 0.3, "RQ3": 0.5})
        assert d.best_subquestions(0.45) == ["RQ1", "RQ3"]

    def test_reference_label_includes_locator(self):
        label = doc("doc_0001", authors=["A B", "C D", "E F", "G H"], year=2023,
                    title="T", venue="V", doi="10.1/x").reference_label()
        assert "et al." in label and "10.1/x" in label and "(2023)" in label


class TestEvidenceIdentity:
    def test_id_is_derived_and_stable(self):
        a = ev("Claim text")
        b = ev("Claim text")
        assert a.id == b.id and a.id.startswith("ev_")

    def test_id_varies_with_document(self):
        assert ev("Claim", doc_id="doc_0001").id != ev("Claim", doc_id="doc_0002").id


class TestNewState:
    def test_all_channels_initialized(self):
        state = new_state("  A question?  ", "scope")
        assert state["main_question"] == "A question?"
        assert state["scope_notes"] == "scope"
        assert state["iteration"] == 0
        assert state["documents"] == {}
        assert isinstance(state["budget"], BudgetUsage)


class TestIntentIdentity:
    def test_intent_defaults(self):
        intent = SearchIntent(id="RQ1-i1", subquestion_id="RQ1", intent="Find things.")
        assert intent.status == "pending"
        assert intent.relevant_found == 0


class TestRegressions:
    """Guards for bugs found while building this out."""

    def test_failed_read_outranks_selected(self):
        """A failed read must record, or the document is re-queued forever."""
        base = {"doc_0001": doc("doc_0001", status="selected")}
        update = {"doc_0001": doc("doc_0001", status="failed")}
        assert merge_documents(base, update)["doc_0001"].status == "failed"

    def test_read_still_outranks_failed(self):
        """Read for RQ1, failed for RQ2: "read" is the truthful summary."""
        base = {"doc_0001": doc("doc_0001", status="read")}
        update = {"doc_0001": doc("doc_0001", status="failed")}
        assert merge_documents(base, update)["doc_0001"].status == "read"

    def test_snowball_hit_does_not_default_to_depth_zero(self):
        """Depth 0 means "found by direct search" and must not be claimed."""
        from optimiztresearch.state import UNKNOWN_SNOWBALL_DEPTH, Document, RawHit

        hit = RawHit(
            source="semantic_scholar",
            title="A cited work on planning",
            provenance=Provenance(
                subquestion_id="RQ1",
                intent_id="RQ1-i1",
                source="semantic_scholar",
                query="backward snowball",
                via="backward_snowball",
            ),
        )
        assert Document.from_hit(hit, "doc_0002").snowball_depth == UNKNOWN_SNOWBALL_DEPTH

        searched = RawHit(
            source="arxiv",
            title="A searched work on planning",
            provenance=Provenance(
                subquestion_id="RQ1", intent_id="RQ1-i1", source="arxiv", query="q"
            ),
        )
        assert Document.from_hit(searched, "doc_0003").snowball_depth == 0

    def test_rediscovering_a_snowball_document_keeps_its_depth(self):
        from optimiztresearch.state import UNKNOWN_SNOWBALL_DEPTH

        established = doc("doc_0001", snowball_depth=2)
        rediscovered = doc("doc_0001", snowball_depth=UNKNOWN_SNOWBALL_DEPTH)
        assert merge_documents({"doc_0001": established}, {"doc_0001": rediscovered})[
            "doc_0001"
        ].snowball_depth == 2

    def test_a_later_search_can_claim_depth_zero(self):
        established = doc("doc_0001", snowball_depth=2)
        searched = doc("doc_0001", snowball_depth=0)
        assert merge_documents({"doc_0001": established}, {"doc_0001": searched})[
            "doc_0001"
        ].snowball_depth == 0
