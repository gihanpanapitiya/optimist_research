"""Tests for ranking and reading selection.

Selection is where the reading budget is actually allocated, so the
properties under test are allocation properties: no subquestion is starved,
no document-subquestion pair is read twice, and caps are respected.
"""

from __future__ import annotations

from forestresearch.config import get_settings
from forestresearch.nodes.screen import (
    _citation_score,
    _intent_yields,
    _novelty,
    _rank,
    _recency_score,
    _select,
    _source_quality,
)
from forestresearch.state import BudgetUsage, Document, Provenance, SearchIntent, Subquestion
from forestresearch.store.canonicalize import significant_tokens


def doc(doc_id: str, **kwargs) -> Document:
    defaults = dict(title=f"Paper {doc_id} On Planning", sources=["arxiv"], abstract="An abstract.")
    defaults.update(kwargs)
    return Document(doc_id=doc_id, **defaults)


def sq(sq_id: str, priority: float = 0.8) -> Subquestion:
    return Subquestion(id=sq_id, question=f"Question {sq_id}?", priority=priority)


class TestComponentScores:
    def test_recency_favours_new_work_but_not_absolutely(self):
        now = 2026
        assert _recency_score(doc("d", year=2026), now) == 1.0
        assert _recency_score(doc("d", year=2010), now) == 0.0
        assert 0 < _recency_score(doc("d", year=2020), now) < 1
        # Unknown year is neutral, not zero.
        assert _recency_score(doc("d"), now) == 0.4

    def test_citations_are_normalized_by_age(self):
        """A new paper with few citations must not be buried by an old one."""
        now = 2026
        recent = _citation_score(doc("d", year=2025, citation_count=30), now)
        old = _citation_score(doc("d", year=2010, citation_count=60), now)
        assert recent > old

    def test_unknown_citation_count_is_neutral(self):
        assert _citation_score(doc("d", year=2024), 2026) == 0.3

    def test_source_quality_discounts_grey_literature(self):
        assert _source_quality(doc("d", sources=["web"])) < _source_quality(
            doc("d", sources=["arxiv"], is_preprint=True)
        )
        assert _source_quality(doc("d", sources=["crossref"], doi="10.1/x", venue="V")) == 1.0

    def test_multi_source_web_hit_is_not_penalized_as_grey(self):
        """A paper also found on arXiv is not grey literature."""
        assert _source_quality(doc("d", sources=["web", "arxiv"])) > _source_quality(
            doc("d", sources=["web"])
        )

    def test_novelty_penalizes_near_duplicates_of_read_work(self):
        already_read = [significant_tokens("Tree Search Methods For Agent Plan Generation")]
        similar = _novelty(
            doc("d", title="Tree Search Methods For Agent Plan Generation"), already_read
        )
        different = _novelty(
            doc("d", title="Biomedical Ontology Alignment Using Graphs"), already_read
        )
        assert similar < different

    def test_novelty_is_maximal_when_nothing_read(self):
        assert _novelty(doc("d"), []) == 1.0


class TestRanking:
    def test_relevance_dominates(self):
        settings = get_settings(refresh=True)
        corpus = {
            "doc_0001": doc("doc_0001", relevance={"RQ1": 0.95}, year=2015, citation_count=0),
            "doc_0002": doc("doc_0002", relevance={"RQ1": 0.20}, year=2026, citation_count=5000),
        }
        ranked = _rank(corpus, settings, [])
        assert ranked["doc_0001"].rank_score > ranked["doc_0002"].rank_score

    def test_missing_abstract_is_penalized_in_abstract_mode(self):
        settings = get_settings(refresh=True)
        corpus = {
            "a": doc("a", relevance={"RQ1": 0.8}, abstract="Some abstract."),
            "b": doc("b", relevance={"RQ1": 0.8}, abstract=None),
        }
        ranked = _rank(corpus, settings, [])
        assert ranked["b"].rank_score < ranked["a"].rank_score


class TestSelection:
    def test_no_subquestion_is_starved(self):
        """Round-robin allocation, not a global top-N cut."""
        settings = get_settings(refresh=True)
        settings.budget.max_documents_per_iteration = 4
        settings.budget.max_documents_per_subquestion = 10

        corpus = {}
        # Eight documents that all strongly favour RQ1; two that favour RQ2.
        for i in range(1, 9):
            corpus[f"doc_{i:04d}"] = doc(f"doc_{i:04d}", relevance={"RQ1": 0.95}, rank_score=0.9)
        for i in range(9, 11):
            corpus[f"doc_{i:04d}"] = doc(f"doc_{i:04d}", relevance={"RQ2": 0.6}, rank_score=0.2)

        queue, selected = _select(corpus, [sq("RQ1", 0.9), sq("RQ2", 0.5)], settings, BudgetUsage())
        covered = {s for item in queue for s in item["subquestion_ids"]}
        assert covered == {"RQ1", "RQ2"}

    def test_below_threshold_documents_are_never_read(self):
        settings = get_settings(refresh=True)
        corpus = {"doc_0001": doc("doc_0001", relevance={"RQ1": 0.1})}
        queue, selected = _select(corpus, [sq("RQ1")], settings, BudgetUsage())
        assert queue == [] and selected == set()

    def test_already_read_pairs_are_not_requeued(self):
        settings = get_settings(refresh=True)
        corpus = {
            "doc_0001": doc("doc_0001", relevance={"RQ1": 0.9}, status="read", read_for=["RQ1"])
        }
        queue, _ = _select(corpus, [sq("RQ1")], settings, BudgetUsage())
        assert queue == []

    def test_document_relevant_to_a_new_subquestion_is_reread(self):
        """Corpus sharing: an already-read paper serves a later subquestion."""
        settings = get_settings(refresh=True)
        corpus = {
            "doc_0001": doc(
                "doc_0001",
                relevance={"RQ1": 0.9, "RQ2": 0.8},
                status="read",
                read_for=["RQ1"],
            )
        }
        queue, _ = _select(corpus, [sq("RQ1"), sq("RQ2")], settings, BudgetUsage())
        assert queue == [{"doc_id": "doc_0001", "subquestion_ids": ["RQ2"]}]

    def test_iteration_cap_is_respected(self):
        settings = get_settings(refresh=True)
        settings.budget.max_documents_per_iteration = 3
        corpus = {
            f"doc_{i:04d}": doc(f"doc_{i:04d}", relevance={"RQ1": 0.9}) for i in range(1, 11)
        }
        _, selected = _select(corpus, [sq("RQ1")], settings, BudgetUsage())
        assert len(selected) == 3

    def test_remaining_read_budget_caps_selection(self):
        settings = get_settings(refresh=True)
        settings.budget.max_documents_read = 12
        settings.budget.max_documents_per_iteration = 20
        corpus = {
            f"doc_{i:04d}": doc(f"doc_{i:04d}", relevance={"RQ1": 0.9}) for i in range(1, 11)
        }
        _, selected = _select(corpus, [sq("RQ1")], settings, BudgetUsage(documents_read=10))
        assert len(selected) == 2

    def test_exhausted_budget_selects_nothing(self):
        settings = get_settings(refresh=True)
        settings.budget.max_documents_read = 5
        corpus = {"doc_0001": doc("doc_0001", relevance={"RQ1": 0.9})}
        queue, selected = _select(corpus, [sq("RQ1")], settings, BudgetUsage(documents_read=5))
        assert queue == [] and selected == set()

    def test_per_subquestion_cap_is_respected(self):
        settings = get_settings(refresh=True)
        settings.budget.max_documents_per_subquestion = 2
        settings.budget.max_documents_per_iteration = 20
        corpus = {
            f"doc_{i:04d}": doc(f"doc_{i:04d}", relevance={"RQ1": 0.9}) for i in range(1, 11)
        }
        queue, _ = _select(corpus, [sq("RQ1")], settings, BudgetUsage())
        assert sum(1 for item in queue if "RQ1" in item["subquestion_ids"]) == 2

    def test_one_document_can_serve_several_subquestions(self):
        settings = get_settings(refresh=True)
        settings.budget.max_documents_per_iteration = 1
        corpus = {"doc_0001": doc("doc_0001", relevance={"RQ1": 0.9, "RQ2": 0.9})}
        queue, selected = _select(corpus, [sq("RQ1"), sq("RQ2")], settings, BudgetUsage())
        assert len(selected) == 1
        assert queue[0]["subquestion_ids"] == ["RQ1", "RQ2"]

    def test_failed_documents_are_excluded(self):
        settings = get_settings(refresh=True)
        corpus = {"doc_0001": doc("doc_0001", relevance={"RQ1": 0.9}, status="failed")}
        queue, _ = _select(corpus, [sq("RQ1")], settings, BudgetUsage())
        assert queue == []


def prov(intent_id: str = "RQ1-i1", source: str = "openalex", via: str = "search") -> Provenance:
    return Provenance(
        subquestion_id="RQ1", intent_id=intent_id, source=source, query="q", via=via
    )


class TestIntentYields:
    """The critic judges saturation from these counts, so they must not grow
    just because the corpus was looked at again."""

    def intents(self, **counts) -> dict[str, SearchIntent]:
        return {
            "RQ1-i1": SearchIntent(id="RQ1-i1", subquestion_id="RQ1", intent="i", **counts)
        }

    def test_counts_distinct_documents(self):
        documents = {
            "doc_0001": doc("doc_0001", relevance={"RQ1": 0.9}, discovered_by=[prov()]),
            "doc_0002": doc("doc_0002", relevance={"RQ1": 0.1}, discovered_by=[prov()]),
        }
        updated = _intent_yields(documents, self.intents(), 0.45)["RQ1-i1"]
        assert (updated.documents_found, updated.relevant_found) == (2, 1)

    def test_rescreening_does_not_credit_again(self):
        """Live regression: adding RQ6 re-screened the corpus and every
        first-iteration intent reported exactly twice its true yield."""
        documents = {
            "doc_0001": doc(
                "doc_0001", relevance={"RQ1": 0.9, "RQ6": 0.2}, discovered_by=[prov()]
            ),
        }
        intents = self.intents()
        for _ in range(2):  # the original screening pass, then the RQ6 re-screen
            intents = {**intents, **_intent_yields(documents, intents, 0.45)}
        assert (intents["RQ1-i1"].documents_found, intents["RQ1-i1"].relevant_found) == (1, 1)

    def test_one_document_from_two_sources_counts_once(self):
        documents = {
            "doc_0001": doc(
                "doc_0001",
                relevance={"RQ1": 0.9},
                discovered_by=[prov(source="openalex"), prov(source="crossref")],
            ),
        }
        updated = _intent_yields(documents, self.intents(), 0.45)["RQ1-i1"]
        assert (updated.documents_found, updated.relevant_found) == (1, 1)

    def test_snowball_hits_do_not_count_toward_search_yield(self):
        """Snowball hits inherit their seed's intent id, but say nothing about
        whether that intent's queries work."""
        documents = {
            "doc_0001": doc("doc_0001", relevance={"RQ1": 0.9}, discovered_by=[prov()]),
            "doc_0002": doc(
                "doc_0002",
                relevance={"RQ1": 0.9},
                discovered_by=[prov(via="backward_snowball")],
            ),
        }
        updated = _intent_yields(documents, self.intents(), 0.45)["RQ1-i1"]
        assert (updated.documents_found, updated.relevant_found) == (1, 1)
