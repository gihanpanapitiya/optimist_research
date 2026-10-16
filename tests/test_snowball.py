"""Tests for citation snowballing: seed choice, direction, and saturation.

Depth is not fixed in this design -- it is governed by budget plus a
saturation criterion -- so the properties worth testing are when expansion
stops and which direction it picks.
"""

from __future__ import annotations

from optimiztresearch.config import get_settings
from optimiztresearch.nodes.snowball import (
    _auto_seeds,
    _is_saturated,
    expand_snowball,
    plan_snowball,
    should_snowball,
)
from optimiztresearch.state import (
    BudgetUsage,
    Document,
    Provenance,
    SnowballRound,
    Subquestion,
    new_state,
)


def rnd(index: int, direction: str, examined: int, relevant: int) -> SnowballRound:
    return SnowballRound(
        round_index=index,
        direction=direction,
        candidates_examined=examined,
        newly_relevant=relevant,
    )


def doc(doc_id: str, year: int, relevance: dict, **kwargs) -> Document:
    defaults = dict(
        title=f"Paper {doc_id}",
        sources=["semantic_scholar"],
        doi=f"10.1/{doc_id}",
        relevance=relevance,
        year=year,
        discovered_by=[
            Provenance(subquestion_id="RQ1", intent_id="RQ1-i1", source="arxiv", query="q")
        ],
    )
    defaults.update(kwargs)
    return Document(doc_id=doc_id, **defaults)


def state_with(documents: dict, **overrides):
    state = new_state("Q?")
    state["documents"] = documents
    state["subquestions"] = {
        "RQ1": Subquestion(id="RQ1", question="How do agents plan?", priority=0.9)
    }
    state.update(overrides)
    return state


class TestSaturation:
    def test_one_low_round_is_not_saturation(self):
        """A single bad round can be one seed with poor citation coverage."""
        history = [rnd(1, "backward", 40, 2)]
        assert _is_saturated(history, 0.10) is False

    def test_two_consecutive_low_rounds_are_saturation(self):
        history = [
            rnd(1, "backward", 40, 20),
            rnd(2, "forward", 50, 2),
            rnd(3, "forward", 60, 1),
        ]
        assert _is_saturated(history, 0.10) is True

    def test_a_recovering_round_resets(self):
        history = [
            rnd(1, "backward", 40, 1),
            rnd(2, "forward", 40, 20),
        ]
        assert _is_saturated(history, 0.10) is False

    def test_declining_yield_stops_planning(self):
        get_settings(refresh=True)
        state = state_with(
            {"doc_0001": doc("doc_0001", 2024, {"RQ1": 0.9})},
            snowball_history=[
                rnd(1, "backward", 40, 1),
                rnd(2, "forward", 60, 0),
            ],
        )
        result = plan_snowball(state)
        assert result["snowball_seeds"] == []
        assert "saturated" in result["log"][0]


class TestBudgetGates:
    def test_round_budget_stops_planning(self, monkeypatch):
        monkeypatch.setenv("OPTIMIZTRESEARCH_BUDGET__MAX_SNOWBALL_ROUNDS", "2")
        get_settings(refresh=True)
        state = state_with(
            {"doc_0001": doc("doc_0001", 2024, {"RQ1": 0.9})},
            snowball_history=[
                rnd(1, "backward", 10, 9),
                rnd(2, "forward", 10, 9),
            ],
        )
        result = plan_snowball(state)
        assert result["snowball_seeds"] == []
        assert "budget spent" in result["log"][0]

    def test_exhausted_read_budget_stops_planning(self, monkeypatch):
        monkeypatch.setenv("OPTIMIZTRESEARCH_BUDGET__MAX_DOCUMENTS_READ", "5")
        get_settings(refresh=True)
        state = state_with(
            {"doc_0001": doc("doc_0001", 2024, {"RQ1": 0.9})},
            budget=BudgetUsage(documents_read=5),
        )
        assert plan_snowball(state)["snowball_seeds"] == []


class TestDirectionChoice:
    def test_recent_clustered_corpus_expands_backward(self):
        """All-recent evidence is missing its foundations."""
        documents = {
            "doc_0001": doc("doc_0001", 2024, {"RQ1": 0.9}),
            "doc_0002": doc("doc_0002", 2025, {"RQ1": 0.85}),
            "doc_0003": doc("doc_0003", 2024, {"RQ1": 0.8}),
        }
        seeds = _auto_seeds(documents, [Subquestion(id="RQ1", question="q")], set(), 3)
        assert seeds and all(s["direction"] == "backward" for s in seeds)

    def test_old_corpus_expands_forward(self):
        """An old median means the current state of the art is missing."""
        documents = {
            "doc_0001": doc("doc_0001", 2016, {"RQ1": 0.9}, citation_count=400),
            "doc_0002": doc("doc_0002", 2018, {"RQ1": 0.85}, citation_count=200),
            "doc_0003": doc("doc_0003", 2019, {"RQ1": 0.8}, citation_count=100),
        }
        seeds = _auto_seeds(documents, [Subquestion(id="RQ1", question="q")], set(), 3)
        assert seeds and seeds[0]["direction"] == "forward"
        # Forward expansion seeds on the most-cited paper, which has the most
        # citing work to find.
        assert seeds[0]["doc_id"] == "doc_0001"

    def test_documents_without_identifiers_are_not_seeded(self):
        documents = {
            "doc_0001": Document(
                doc_id="doc_0001", title="No ids", year=2024, relevance={"RQ1": 0.9}
            )
        }
        assert _auto_seeds(documents, [Subquestion(id="RQ1", question="q")], set(), 3) == []

    def test_low_relevance_documents_are_not_seeded(self):
        documents = {"doc_0001": doc("doc_0001", 2024, {"RQ1": 0.5})}
        assert _auto_seeds(documents, [Subquestion(id="RQ1", question="q")], set(), 3) == []

    def test_already_snowballed_documents_are_not_reseeded(self):
        documents = {
            "doc_0001": doc("doc_0001", 2024, {"RQ1": 0.9}, snowball_depth=1),
        }
        assert _auto_seeds(documents, [Subquestion(id="RQ1", question="q")], set(), 3) == []


class TestCriticRequests:
    def test_explicit_requests_take_precedence(self):
        get_settings(refresh=True)
        documents = {
            "doc_0001": doc("doc_0001", 2024, {"RQ1": 0.9}),
            "doc_0002": doc("doc_0002", 2024, {"RQ1": 0.95}),
        }
        state = state_with(
            documents,
            snowball_seeds=[
                {"doc_id": "doc_0001", "direction": "forward", "reason": "critic asked"}
            ],
        )
        seeds = plan_snowball(state)["snowball_seeds"]
        assert seeds[0] == {
            "doc_id": "doc_0001",
            "direction": "forward",
            "reason": "critic asked",
        }

    def test_requests_for_unknown_documents_are_dropped(self):
        get_settings(refresh=True)
        state = state_with(
            {"doc_0001": doc("doc_0001", 2024, {"RQ1": 0.9})},
            snowball_seeds=[{"doc_id": "doc_9999", "direction": "forward"}],
        )
        seeds = plan_snowball(state)["snowball_seeds"]
        assert all(s["doc_id"] != "doc_9999" for s in seeds)


class TestRouting:
    def test_seeds_route_to_expansion(self):
        state = new_state("Q?")
        state["snowball_seeds"] = [{"doc_id": "doc_0001", "direction": "backward"}]
        assert should_snowball(state) == "expand_snowball"

    def test_no_seeds_routes_to_reading(self):
        assert should_snowball(new_state("Q?")) == "prepare_reading"


class TestBreadthLimits:
    """Snowball breadth must track the reading budget.

    On a live run with an 8-document reading budget, the defaults (6 seeds at
    40 edges) produced 120 new candidates from one round -- every one of
    which then had to be screened before it could be discarded.
    """

    def test_seed_count_is_configurable(self, monkeypatch):
        monkeypatch.setenv("OPTIMIZTRESEARCH_BUDGET__MAX_SNOWBALL_SEEDS", "2")
        get_settings(refresh=True)
        documents = {
            f"doc_{i:04d}": doc(f"doc_{i:04d}", 2024, {f"RQ{i}": 0.9})
            for i in range(1, 6)
        }
        state = state_with(documents)
        state["subquestions"] = {
            f"RQ{i}": Subquestion(id=f"RQ{i}", question=f"q{i}?", priority=0.9)
            for i in range(1, 6)
        }
        assert len(plan_snowball(state)["snowball_seeds"]) <= 2

    def test_explicit_requests_are_also_capped(self, monkeypatch):
        monkeypatch.setenv("OPTIMIZTRESEARCH_BUDGET__MAX_SNOWBALL_SEEDS", "1")
        get_settings(refresh=True)
        documents = {
            "doc_0001": doc("doc_0001", 2024, {"RQ1": 0.9}),
            "doc_0002": doc("doc_0002", 2024, {"RQ1": 0.9}),
        }
        state = state_with(
            documents,
            snowball_seeds=[
                {"doc_id": "doc_0001", "direction": "forward"},
                {"doc_id": "doc_0002", "direction": "forward"},
            ],
        )
        assert len(plan_snowball(state)["snowball_seeds"]) == 1

    async def test_edge_limit_is_passed_to_the_source(self, monkeypatch):
        monkeypatch.setenv("OPTIMIZTRESEARCH_BUDGET__MAX_SNOWBALL_EDGES_PER_SEED", "7")
        get_settings(refresh=True)
        from optimiztresearch.nodes import snowball as snowball_module

        seen: list[int] = []

        class Recorder:
            name = "fake"
            supports_backward = True
            supports_forward = True

            async def references(self, doc, limit):
                seen.append(limit)
                return []

            async def citations(self, doc, limit):
                seen.append(limit)
                return []

        monkeypatch.setattr(
            snowball_module, "snowball_sources", lambda direction: [Recorder()]
        )
        state = state_with({"doc_0001": doc("doc_0001", 2024, {"RQ1": 0.9})})
        state["snowball_seeds"] = [{"doc_id": "doc_0001", "direction": "forward"}]
        await snowball_module.expand_snowball(state)
        assert seen == [7]


class TestExpandedSeedsAreNotRepeated:
    """Seed choice is a pure function of a corpus that barely changes between
    rounds. On a live run, round 2 re-picked round 1's four seeds and spent
    the round re-fetching 48 edges that were all already in the corpus."""

    def test_auto_seeds_skip_an_expanded_pair(self):
        documents = {
            "doc_0001": doc("doc_0001", 2024, {"RQ1": 0.95}),
            "doc_0002": doc("doc_0002", 2024, {"RQ1": 0.9}),
        }
        active = [Subquestion(id="RQ1", question="q")]
        first = _auto_seeds(documents, active, set(), 1)
        assert first[0]["doc_id"] == "doc_0001"
        assert first[0]["direction"] == "backward"

        later = _auto_seeds(
            documents, active, set(), 1, expanded=frozenset({"doc_0001:backward"})
        )
        assert later[0]["doc_id"] == "doc_0002"

    def test_other_direction_of_an_expanded_seed_is_still_available(self):
        documents = {"doc_0001": doc("doc_0001", 2018, {"RQ1": 0.9}, citation_count=500)}
        active = [Subquestion(id="RQ1", question="q")]
        seeds = _auto_seeds(
            documents, active, set(), 1, expanded=frozenset({"doc_0001:backward"})
        )
        assert (seeds[0]["doc_id"], seeds[0]["direction"]) == ("doc_0001", "forward")

    def test_critic_request_for_an_expanded_pair_is_dropped(self):
        get_settings(refresh=True)
        state = state_with(
            {
                "doc_0001": doc("doc_0001", 2024, {"RQ1": 0.95}),
                "doc_0002": doc("doc_0002", 2024, {"RQ1": 0.9}),
            },
            snowball_seeds=[{"doc_id": "doc_0001", "direction": "backward"}],
            snowball_expanded=["doc_0001:backward"],
        )
        seeds = plan_snowball(state)["snowball_seeds"]
        assert [s["doc_id"] for s in seeds] == ["doc_0002"]

    async def test_second_round_moves_on_to_new_seeds(self, monkeypatch):
        get_settings(refresh=True)
        from optimiztresearch.nodes import snowball as snowball_module

        class NoEdges:
            name = "fake"

            async def references(self, doc, limit):
                return []

            async def citations(self, doc, limit):
                return []

        monkeypatch.setattr(snowball_module, "snowball_sources", lambda d: [NoEdges()])
        state = state_with(
            {
                "doc_0001": doc("doc_0001", 2024, {"RQ1": 0.95}),
                "doc_0002": doc("doc_0002", 2024, {"RQ1": 0.9}),
            }
        )
        state["snowball_seeds"] = plan_snowball(state)["snowball_seeds"]
        assert [s["doc_id"] for s in state["snowball_seeds"]] == ["doc_0001"]

        result = await expand_snowball(state)
        # A seed with no citation data counts as expanded: retrying it would
        # only spend another round.
        assert result["snowball_expanded"] == ["doc_0001:backward"]

        state["snowball_expanded"] = result["snowball_expanded"]
        state["snowball_history"] = result["snowball_history"]
        state["snowball_seeds"] = []
        seeds = plan_snowball(state)["snowball_seeds"]
        assert [s["doc_id"] for s in seeds] == ["doc_0002"]
