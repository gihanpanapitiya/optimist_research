"""Tests for on-disk persistence and state round-tripping."""

from __future__ import annotations

from litresearch.state import (
    BudgetUsage,
    Contradiction,
    Critique,
    Document,
    DocumentReading,
    Evidence,
    Provenance,
    SearchIntent,
    Subquestion,
    new_state,
)
from litresearch.store.corpus import ResearchCorpus, slugify


def sample_state():
    state = new_state("How do agents plan?")
    state["subquestions"] = {
        "RQ1": Subquestion(id="RQ1", question="How are plans generated?", coverage=0.8,
                           synthesis="Two families appear [doc_0001].", documents_read=2,
                           intent_ids=["RQ1-i1"]),
        "RQ2": Subquestion(id="RQ2", question="How do agents replan?", status="deprioritized"),
    }
    state["intents"] = {
        "RQ1-i1": SearchIntent(id="RQ1-i1", subquestion_id="RQ1",
                               intent="Find plan generation methods.", status="searched",
                               documents_found=10, relevant_found=4)
    }
    state["documents"] = {
        "doc_0001": Document(
            doc_id="doc_0001", title="Tree Search For Plans", authors=["Jane Doe"],
            year=2024, venue="NeurIPS", doi="10.1/x", sources=["arxiv", "openalex"],
            status="read", read_for=["RQ1"], relevance={"RQ1": 0.9},
            screening_note="Central method paper.", citation_count=42,
            discovered_by=[Provenance(subquestion_id="RQ1", intent_id="RQ1-i1",
                                      source="arxiv", query="tree search plans")],
            abstract="We present a tree search method.",
        )
    }
    state["evidence"] = [
        Evidence(claim="Tree search improves success.", support="40% to 55%.",
                 doc_id="doc_0001", subquestion_id="RQ1", location="Section 4.2",
                 evidence_type="experimental", strength=0.8)
    ]
    state["readings"] = {
        "doc_0001": DocumentReading(doc_id="doc_0001", main_contribution="A method.",
                                    method="Tree search.", major_findings=["+15 points"],
                                    read_for=["RQ1"])
    }
    state["contradictions"] = [
        Contradiction(description="doc_0001 and doc_0002 disagree on scaling.",
                      doc_ids=["doc_0001"])
    ]
    state["critique"] = Critique(status="sufficient", reasoning="Enough.", overall_coverage=0.8)
    state["budget"] = BudgetUsage(documents_read=2, search_operations=5, research_iterations=1)
    state["final_report"] = "# Report\n\nBody citing [doc_0001]."
    state["global_synthesis"] = "A narrative."
    return state


class TestSlug:
    def test_slug_is_filesystem_safe(self):
        assert (
            slugify("What are the *recent* developments in AI?")
            == "what-are-the-recent-developments-in-ai"
        )

    def test_empty_question_has_a_fallback(self):
        assert slugify("???") == "research"

    def test_slug_is_truncated_without_a_trailing_dash(self):
        slug = slugify("a " * 100)
        assert len(slug) <= 60 and not slug.endswith("-")


class TestLayout:
    def test_question_gets_its_own_subdirectory(self, tmp_path):
        corpus = ResearchCorpus(tmp_path, "How do agents plan?")
        assert corpus.root == tmp_path / "how-do-agents-plan"

    def test_write_all_produces_the_documented_layout(self, tmp_path):
        corpus = ResearchCorpus(tmp_path, "How do agents plan?")
        corpus.write_all(sample_state())
        root = corpus.root

        assert (root / "research_state.json").exists()
        assert (root / "research_plan.md").exists()
        assert (root / "titles.md").exists()
        assert (root / "final_report.md").exists()
        assert (root / "evidence" / "claims.md").exists()
        assert (root / "evidence" / "contradictions.md").exists()
        assert (root / "sources" / "doc_0001" / "metadata.json").exists()
        assert (root / "sources" / "doc_0001" / "notes.md").exists()
        assert list((root / "subtopics").glob("*.md"))

    def test_notes_record_provenance_and_evidence(self, tmp_path):
        corpus = ResearchCorpus(tmp_path, "Q")
        corpus.write_all(sample_state())
        notes = (corpus.root / "sources" / "doc_0001" / "notes.md").read_text()
        assert "tree search plans" in notes          # the query that found it
        assert "RQ1/RQ1-i1 via arxiv" in notes       # discovery path
        assert "Section 4.2" in notes                # evidence location
        assert "Tree search improves success." in notes

    def test_unread_documents_get_no_directory(self, tmp_path):
        state = sample_state()
        state["documents"]["doc_0002"] = Document(
            doc_id="doc_0002", title="Unread paper", status="candidate"
        )
        corpus = ResearchCorpus(tmp_path, "Q")
        corpus.write_all(state)
        assert not (corpus.root / "sources" / "doc_0002").exists()
        # It still appears in the human-readable index.
        assert "doc_0002" in (corpus.root / "titles.md").read_text()

    def test_plan_marks_deprioritized_subquestions(self, tmp_path):
        corpus = ResearchCorpus(tmp_path, "Q")
        corpus.write_all(sample_state())
        plan = (corpus.root / "research_plan.md").read_text()
        assert "deprioritized" in plan
        assert "RQ1-i1" in plan

    def test_weak_claims_are_flagged_in_subtopics(self, tmp_path):
        corpus = ResearchCorpus(tmp_path, "Q")
        corpus.write_all(sample_state())
        text = "".join(p.read_text() for p in (corpus.root / "subtopics").glob("*.md"))
        assert "weakly supported" in text  # single-source claim


class TestRoundTrip:
    def test_state_survives_a_round_trip(self, tmp_path):
        corpus = ResearchCorpus(tmp_path, "Q")
        original = sample_state()
        corpus.save_state(original)
        loaded = corpus.load_state()

        assert loaded is not None
        assert loaded["main_question"] == original["main_question"]
        assert loaded["subquestions"]["RQ1"].coverage == 0.8
        assert loaded["documents"]["doc_0001"].relevance == {"RQ1": 0.9}
        assert loaded["documents"]["doc_0001"].discovered_by[0].query == "tree search plans"
        assert loaded["evidence"][0].location == "Section 4.2"
        assert loaded["readings"]["doc_0001"].method == "Tree search."
        assert loaded["contradictions"][0].doc_ids == ["doc_0001"]
        assert loaded["critique"].status == "sufficient"
        assert loaded["budget"].documents_read == 2
        assert loaded["intents"]["RQ1-i1"].relevant_found == 4

    def test_scratch_channels_are_not_persisted(self, tmp_path):
        import json

        corpus = ResearchCorpus(tmp_path, "Q")
        corpus.save_state(sample_state())
        payload = json.loads(corpus.state_path.read_text())
        for key in ("pending_hits", "query_plans", "snowball_history"):
            assert key not in payload

    def test_missing_state_returns_none(self, tmp_path):
        assert ResearchCorpus(tmp_path, "Q").load_state() is None

    def test_corrupt_state_returns_none(self, tmp_path):
        corpus = ResearchCorpus(tmp_path, "Q")
        corpus.ensure()
        corpus.state_path.write_text("{ not json")
        assert corpus.load_state() is None


class TestProjectionRobustness:
    def test_intents_appear_even_without_a_back_link(self, tmp_path):
        """A missing subquestion->intent link must not hide the intent."""
        state = sample_state()
        state["subquestions"]["RQ1"].intent_ids = []
        corpus = ResearchCorpus(tmp_path, "Q")
        corpus.write_all(state)
        assert "RQ1-i1" in (corpus.root / "research_plan.md").read_text()

    def test_intents_are_not_duplicated_when_linked_both_ways(self, tmp_path):
        corpus = ResearchCorpus(tmp_path, "Q")
        corpus.write_all(sample_state())
        plan = (corpus.root / "research_plan.md").read_text()
        assert plan.count("_intent RQ1-i1_") == 1


class TestPathResolution:
    """A relative output path resolves against the working directory.

    From a notebook kernel that is usually not the project, and on macOS a
    kernel started outside a project often has "/" as its cwd -- where
    "research" resolves to "/research" and the filesystem is read-only.
    """

    def test_paths_are_resolved_to_absolute(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        corpus = ResearchCorpus("research", "Q?")
        assert corpus.root.is_absolute()
        assert corpus.root == (tmp_path / "research" / "q").resolve()

    def test_user_home_is_expanded(self):
        assert "~" not in str(ResearchCorpus("~/research", "Q?").root)

    def test_an_unwritable_destination_explains_itself(self, monkeypatch):
        """The bare pathlib error does not reveal where the path resolved."""
        import pytest

        corpus = ResearchCorpus("/research-which-cannot-be-created", "Q?")
        with pytest.raises(OSError) as excinfo:
            corpus.ensure()
        message = str(excinfo.value)
        assert "research-which-cannot-be-created" in message
        assert "current working directory" in message
        assert "output_dir" in message


class TestEnvDiscovery:
    """Configuration must be findable from outside the project directory."""

    def test_the_project_env_is_a_candidate(self):
        from litresearch.config import PROJECT_ROOT, _env_files

        assert PROJECT_ROOT / ".env" in _env_files()

    def test_nearer_env_files_take_precedence(self, tmp_path, monkeypatch):
        from litresearch.config import PROJECT_ROOT, _env_files

        monkeypatch.chdir(tmp_path)
        files = _env_files()
        # Later entries win in pydantic-settings, so the cwd must come last
        # and the project root must not override it.
        assert files[-1] == tmp_path.resolve() / ".env"
        assert files.index(PROJECT_ROOT / ".env") < len(files) - 1

    def test_candidates_are_unique(self, monkeypatch):
        """The project root is also an ancestor of the cwd when run in-tree."""
        from litresearch.config import PROJECT_ROOT, _env_files

        monkeypatch.chdir(PROJECT_ROOT)
        files = _env_files()
        assert len(files) == len(set(files))
