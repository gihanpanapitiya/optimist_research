"""Test fixtures.

The graph is exercised end to end with a fake model and fake sources. Nothing
here tests model output quality -- only that the machinery around it is
correct: that state flows, reducers merge, fan-out converges, budgets are
respected and loops terminate.
"""

from __future__ import annotations

import os

import pytest

from litresearch.config import Settings, get_settings
from litresearch.schemas import (
    CoverageScore,
    CritiqueOutput,
    ExtractedEvidence,
    GeneratedQuery,
    GlobalSynthesisOutput,
    IntentOutput,
    PlannedIntent,
    PlannedSubquestion,
    PlanOutput,
    QueryPlanOutput,
    ReadingOutput,
    ReportSection,
    ScreenedDocument,
    ScreeningOutput,
    SubquestionScore,
    SubquestionSynthesisOutput,
)
from litresearch.sources.base import SearchSource
from litresearch.state import Document, Provenance, QueryPlan, RawHit


@pytest.fixture(autouse=True)
def isolate_settings(monkeypatch):
    """Keep the developer's local configuration out of the tests.

    Settings resolve from ``litresearch.yaml``, ``.env`` and ``LITRESEARCH_*``
    on instantiation, so without this a local file silently changes test
    outcomes. Both layers have done exactly that: first a ``.env`` lowering a
    budget ceiling, then the ``litresearch.yaml`` that replaced it, each
    failing a snowball saturation test for reasons unrelated to the code
    under test.

    Every configuration layer added in future needs disabling here too.
    """
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    monkeypatch.setitem(Settings.model_config, "yaml_file", ())
    monkeypatch.delenv("LITRESEARCH_CONFIG_FILE", raising=False)
    for key in list(os.environ):
        if key.startswith("LITRESEARCH_"):
            monkeypatch.delenv(key, raising=False)
    get_settings(refresh=True)
    yield
    get_settings(refresh=True)


# ---------------------------------------------------------------------------
# Fake literature source
# ---------------------------------------------------------------------------


class FakeSource(SearchSource):
    """Returns a deterministic set of hits, with no network access."""

    name = "fake"
    min_request_interval = 0.0
    max_concurrency = 8
    supports_backward = True
    supports_forward = True

    def __init__(self, hit_count: int = 4, start: int = 1) -> None:
        super().__init__()
        self.hit_count = hit_count
        self.start = start
        self.searches: list[QueryPlan] = []
        self.reference_calls: list[str] = []
        self.citation_calls: list[str] = []

    async def search(self, plan: QueryPlan) -> list[RawHit]:
        self.searches.append(plan)
        return [
            RawHit(
                source=self.name,
                source_id=f"fake-{index}",
                title=f"Paper Number {index} On Agent Planning Mechanisms",
                abstract=(
                    f"We present method M{index} for planning in agents and "
                    f"evaluate it on benchmark B{index}, improving success rate "
                    f"from 40% to {50 + index}%."
                ),
                authors=[f"Author{index} Surname{index}", "Second Writer"],
                year=2020 + (index % 5),
                venue="Journal of Fake Research",
                doi=f"10.1234/fake.{index}",
                citation_count=index * 7,
                provenance=Provenance(
                    subquestion_id=plan.subquestion_id,
                    intent_id=plan.intent_id,
                    source=self.name,
                    query=plan.query,
                    rank=index,
                ),
            )
            for index in range(self.start, self.start + self.hit_count)
        ]

    async def references(self, doc: Document, limit: int = 40) -> list[RawHit]:
        self.reference_calls.append(doc.doc_id)
        return await self._edge(doc, "backward_snowball", offset=100)

    async def citations(self, doc: Document, limit: int = 40) -> list[RawHit]:
        self.citation_calls.append(doc.doc_id)
        return await self._edge(doc, "forward_snowball", offset=200)

    async def _edge(self, doc: Document, via: str, offset: int) -> list[RawHit]:
        return [
            RawHit(
                source=self.name,
                source_id=f"fake-{offset + index}",
                title=f"Cited Work {offset + index} About Planning Failure Recovery",
                abstract=f"An earlier study {offset + index} of plan repair.",
                authors=[f"Cited{offset + index} Person"],
                year=2015 + index,
                doi=f"10.1234/fake.{offset + index}",
                citation_count=200 - index,
                provenance=Provenance(
                    subquestion_id="RQ1",
                    intent_id="RQ1-i1",
                    source=self.name,
                    query=f"{via} from {doc.doc_id}",
                    rank=index,
                    via=via,  # type: ignore[arg-type]
                    seed_doc_id=doc.doc_id,
                ),
            )
            for index in range(2)
        ]


# ---------------------------------------------------------------------------
# Fake model
# ---------------------------------------------------------------------------


class FakeStructuredRunnable:
    def __init__(self, responses: FakeModel, schema: type) -> None:
        self.responses = responses
        self.schema = schema

    async def ainvoke(self, messages, **kwargs):
        self.responses.calls.append(self.schema.__name__)
        if self.schema.__name__ in self.responses.failures:
            raise RuntimeError(f"simulated failure for {self.schema.__name__}")
        builder = self.responses.handlers.get(self.schema.__name__)
        if builder is None:
            raise AssertionError(f"no fake response configured for {self.schema.__name__}")
        return builder(messages)


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeChatModel:
    def __init__(self, responses: FakeModel) -> None:
        self.responses = responses

    async def ainvoke(self, messages, **kwargs):
        self.responses.calls.append("prose")
        return FakeMessage(self.responses.prose)


class FakeModel:
    """Canned structured responses, keyed by schema name."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.failures: set[str] = set()
        self.critique_queue: list[CritiqueOutput] = []
        self.prose = (
            "Methods for plan repair cluster into two families [doc_0001]. "
            "A single study reports large gains [doc_0002], which has not been "
            "replicated."
        )
        self.handlers = {
            "PlanOutput": self._plan,
            "IntentOutput": self._intents,
            "QueryPlanOutput": self._queries,
            "ScreeningOutput": self._screening,
            "ReadingOutput": self._reading,
            "SubquestionSynthesisOutput": self._sq_synthesis,
            "GlobalSynthesisOutput": self._global_synthesis,
            "CritiqueOutput": self._critique,
        }

    # -- handlers ----------------------------------------------------------

    def _plan(self, _messages):
        return PlanOutput(
            interpretation="Asking about mechanisms for agent planning.",
            rationale="Two facets must be established.",
            subquestions=[
                PlannedSubquestion(
                    id="RQ1",
                    question="How do agents generate plans?",
                    rationale="Core mechanism.",
                    priority=0.9,
                ),
                PlannedSubquestion(
                    id="RQ2",
                    question="How do agents recover from planning failures?",
                    rationale="Failure handling is distinct.",
                    priority=0.8,
                ),
            ],
        )

    def _intents(self, _messages):
        return IntentOutput(
            intents=[
                PlannedIntent(
                    intent="Find methods for generating plans in agents.",
                    rationale="Primary mechanism evidence.",
                    desired_evidence=["method descriptions", "evaluations"],
                    preferred_sources=["fake"],
                )
            ]
        )

    def _queries(self, messages):
        # Vary the query by subquestion, as a real generator would: an
        # identical string for every intent would be deduplicated away.
        text = messages[-1].content
        sq_id = "RQ1"
        for line in text.splitlines():
            if line.startswith("Subquestion "):
                sq_id = line.split(" ")[1].rstrip(":")
                break
        return QueryPlanOutput(
            reasoning="One source available.",
            queries=[GeneratedQuery(source="fake", query=f"agent planning {sq_id}")],
        )

    def _screening(self, messages):
        text = messages[-1].content
        # Score every offered document against whichever subquestions the
        # prompt actually listed, so this works for any batch shape.
        sq_ids = [
            line.split(":")[0].strip("- ").strip()
            for line in text.splitlines()
            if line.startswith("- RQ")
        ]
        count = text.count("\n  ") // 2 + 1
        return ScreeningOutput(
            documents=[
                ScreenedDocument(
                    index=index,
                    scores=[
                        SubquestionScore(subquestion_id=sq_id, relevance=0.85)
                        for sq_id in (sq_ids or ["RQ1"])
                    ],
                    note="Directly relevant method paper.",
                )
                for index in range(count)
            ]
        )

    def _reading(self, messages):
        text = messages[-1].content
        # One evidence item per assigned subquestion, as a real reader would
        # produce when a paper bears on several.
        sq_ids = [
            line.split(":")[0].strip("- ").strip()
            for line in text.splitlines()
            if line.startswith("- RQ")
        ] or ["RQ1"]
        evidence = [
            ExtractedEvidence(
                claim=f"Tree search improves plan success on benchmark B ({sq_id}).",
                support="Success rate rises from 40% to 55%.",
                subquestion_id=sq_id,
                location="Section 4.2",
                evidence_type="experimental",
                strength=0.8,
            )
            for sq_id in sq_ids
        ]
        evidence.append(
            ExtractedEvidence(
                claim="Replanning cost grows with horizon length.",
                support="Latency scales linearly with steps.",
                subquestion_id=sq_ids[0],
                location="Table 3",
                evidence_type="experimental",
                strength=0.6,
                is_limitation=True,
            )
        )
        return ReadingOutput(
            relevant=True,
            main_contribution="A planning method.",
            method="Tree search over LLM-proposed actions.",
            experimental_setup="Benchmark B, two baselines.",
            major_findings=["Success rate improves by 10 points."],
            limitations=["Single benchmark."],
            evidence=evidence,
            terminology_notes=["plan repair"],
        )

    def _sq_synthesis(self, _messages):
        return SubquestionSynthesisOutput(
            synthesis="Two families of method appear [doc_0001].",
            consensus_points=["Tree search helps."],
            competing_approaches=["Search versus reactive replanning."],
            empirical_findings=["+15 points on benchmark B."],
            limitations=["Single-benchmark evaluations."],
            unresolved_questions=["Long-horizon behaviour."],
            knowledge_gaps=["No studies beyond 20-step horizons."],
            coverage=0.85,
            answered=True,
        )

    def _global_synthesis(self, _messages):
        return GlobalSynthesisOutput(
            narrative="The literature converges on search-based planning [doc_0001].",
            major_themes=["Search-based planning"],
            cross_cutting_findings=["Evaluation is benchmark-bound."],
            outline=[
                ReportSection(
                    heading="Search-based planning",
                    purpose="Establish the dominant approach.",
                    subquestion_ids=["RQ1"],
                ),
                ReportSection(
                    heading="Failure recovery",
                    purpose="Cover replanning.",
                    subquestion_ids=["RQ2"],
                ),
            ],
            open_questions=["Long-horizon planning"],
            emerging_directions=["World-model-based planning"],
        )

    def _critique(self, _messages):
        if self.critique_queue:
            return self.critique_queue.pop(0)
        return CritiqueOutput(
            status="sufficient",
            reasoning="Evidence answers the question.",
            overall_coverage=0.85,
            subquestion_coverage=[
                CoverageScore(subquestion_id="RQ1", coverage=0.85),
                CoverageScore(subquestion_id="RQ2", coverage=0.8),
            ],
            coverage_assessment="Both subquestions addressed.",
            evidence_quality_assessment="Adequate.",
            source_diversity_assessment="Acceptable.",
            recency_assessment="Current.",
            saturation_assessment="Yield falling.",
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_source() -> FakeSource:
    return FakeSource()


@pytest.fixture
def fake_model() -> FakeModel:
    return FakeModel()


@pytest.fixture
def wired(monkeypatch, fake_model, fake_source, tmp_path):
    """Patch models and sources, and point output at a temp directory.

    Node modules bind ``active_sources`` and ``get_model`` at import time, so
    each binding is patched where it is used rather than at the definition.
    """
    monkeypatch.setenv("LITRESEARCH_OUTPUT_DIR", str(tmp_path / "research"))
    monkeypatch.setenv("LITRESEARCH_ENABLED_SOURCES", "arxiv")
    monkeypatch.setenv("LITRESEARCH_READ_FULL_TEXT", "false")
    get_settings(refresh=True)

    def structured(role: str, schema: type):
        return FakeStructuredRunnable(fake_model, schema)

    monkeypatch.setattr("litresearch.llm.structured", structured)
    monkeypatch.setattr(
        "litresearch.nodes.report.get_model", lambda role: FakeChatModel(fake_model)
    )

    sources = {"fake": fake_source}
    for module in (
        "litresearch.nodes.planner",
        "litresearch.nodes.intents",
        "litresearch.nodes.queries",
        "litresearch.nodes.search",
    ):
        monkeypatch.setattr(f"{module}.active_sources", lambda: sources)
    monkeypatch.setattr(
        "litresearch.nodes.snowball.snowball_sources", lambda direction: [fake_source]
    )

    yield fake_model
    get_settings(refresh=True)
