"""Tests for source availability policy and verified identifier resolution.

Two concerns:

* The Semantic Scholar toggle, which decides whether an unkeyed run spends
  budget on a source that will rate-limit to nothing.
* OpenAlex work resolution, which must never seed snowballing from the wrong
  paper. OpenAlex genuinely has some arXiv DOIs attached to unrelated
  records, so a lookup is only trusted once verified.
"""

from __future__ import annotations

import pytest

from litresearch.config import get_settings
from litresearch.sources.openalex import OpenAlexSource
from litresearch.sources.registry import reset_sources, snowball_sources
from litresearch.state import Document
from litresearch.store.canonicalize import title_is_prefix, titles_match


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Pin the key-bearing env vars so these tests do not depend on the host."""
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("LITRESEARCH_SEMANTIC_SCHOLAR", raising=False)
    monkeypatch.setenv("LITRESEARCH_ENABLED_SOURCES", ",".join(
        ["arxiv", "semantic_scholar", "openalex", "crossref", "pubmed", "web"]
    ))
    reset_sources()
    yield
    reset_sources()
    get_settings(refresh=True)


class TestSemanticScholarToggle:
    def test_auto_disables_without_a_key(self, monkeypatch):
        settings = get_settings(refresh=True)
        assert settings.semantic_scholar == "auto"
        assert settings.use_semantic_scholar is False
        assert "semantic_scholar" not in settings.available_sources()
        assert "SEMANTIC_SCHOLAR_API_KEY" in settings.source_unavailable_reason(
            "semantic_scholar"
        )

    def test_auto_enables_with_a_key(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "sk-test")
        settings = get_settings(refresh=True)
        assert settings.use_semantic_scholar is True
        assert "semantic_scholar" in settings.available_sources()
        assert settings.source_unavailable_reason("semantic_scholar") is None

    def test_on_forces_it_without_a_key(self, monkeypatch):
        monkeypatch.setenv("LITRESEARCH_SEMANTIC_SCHOLAR", "on")
        settings = get_settings(refresh=True)
        assert settings.use_semantic_scholar is True
        assert "semantic_scholar" in settings.available_sources()

    def test_off_disables_it_despite_a_key(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "sk-test")
        monkeypatch.setenv("LITRESEARCH_SEMANTIC_SCHOLAR", "off")
        settings = get_settings(refresh=True)
        assert settings.use_semantic_scholar is False
        assert "off" in settings.source_unavailable_reason("semantic_scholar")

    def test_the_source_object_agrees_with_the_policy(self, monkeypatch):
        from litresearch.sources.semantic_scholar import SemanticScholarSource

        get_settings(refresh=True)
        assert SemanticScholarSource().is_available() is False
        monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "sk-test")
        get_settings(refresh=True)
        assert SemanticScholarSource().is_available() is True

    def test_disabling_it_leaves_the_other_sources(self):
        settings = get_settings(refresh=True)
        available = settings.available_sources()
        assert "openalex" in available and "arxiv" in available
        # web also drops out, having no key in this fixture.
        assert "web" not in available


class TestCitationSourcePreference:
    def test_openalex_leads_when_unkeyed(self):
        get_settings(refresh=True)
        assert [s.name for s in snowball_sources("forward")] == ["openalex"]
        # Crossref is a backward-only fallback for records whose OpenAlex
        # referenced_works list is empty.
        assert [s.name for s in snowball_sources("backward")] == ["openalex", "crossref"]

    def test_semantic_scholar_leads_when_keyed(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "sk-test")
        get_settings(refresh=True)
        assert [s.name for s in snowball_sources("forward")] == [
            "semantic_scholar",
            "openalex",
        ]
        assert [s.name for s in snowball_sources("backward")] == [
            "semantic_scholar",
            "openalex",
            "crossref",
        ]

    def test_crossref_is_backward_only(self):
        get_settings(refresh=True)
        assert "crossref" not in [s.name for s in snowball_sources("forward")]

    def test_backward_survives_losing_openalex(self, monkeypatch):
        """Crossref alone can still supply reference lists."""
        monkeypatch.setenv("LITRESEARCH_ENABLED_SOURCES", "arxiv,crossref")
        get_settings(refresh=True)
        assert [s.name for s in snowball_sources("backward")] == ["crossref"]
        assert snowball_sources("forward") == []

    def test_a_citation_source_always_remains(self, monkeypatch):
        """Snowballing must survive an unkeyed run."""
        get_settings(refresh=True)
        assert snowball_sources("backward")
        assert snowball_sources("forward")


class TestTitleMatching:
    def test_case_and_punctuation_are_ignored(self):
        assert titles_match(
            "Reflexion: Language Agents with Verbal Reinforcement Learning",
            "Reflexion: language agents with verbal reinforcement learning",
        )

    def test_unrelated_titles_do_not_match(self):
        assert not titles_match(
            "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
            "BNAI, NO-TOKEN, and MIND-UNITY: Pillars of a Systemic Revolution",
        )

    def test_a_three_word_prefix_matches(self):
        assert titles_match(
            "Chain-of-Thought Prompting",
            "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        )

    def test_a_generic_two_word_prefix_does_not(self):
        """Otherwise "Deep Learning" would match every paper about it."""
        assert not titles_match("Deep Learning", "Deep Learning for Protein Folding")
        assert not titles_match("Planning", "Planning for AI Agents in Dynamic Worlds")

    def test_numerals_veto_a_match(self):
        assert not titles_match(
            "Llama 2: Open Foundation and Fine-Tuned Chat Models",
            "Llama 3: Open Foundation and Fine-Tuned Chat Models",
        )

    def test_prefix_helper_is_weaker_than_match(self):
        # Too generic for titles_match, but a genuine prefix.
        assert title_is_prefix("LLM-Planner", "LLM-Planner: Few-Shot Grounded Planning")
        assert not titles_match("LLM-Planner", "LLM-Planner: Few-Shot Grounded Planning")

    def test_prefix_helper_still_respects_numerals(self):
        assert not title_is_prefix("Llama 2", "Llama 3: Open Foundation")


class TestOpenAlexVerification:
    def setup_method(self):
        get_settings(refresh=True)
        self.source = OpenAlexSource()

    @staticmethod
    def record(work_id: str, title: str, authors: list[str]) -> dict:
        return {
            "id": f"https://openalex.org/{work_id}",
            "display_name": title,
            "authorships": [{"author": {"display_name": a}} for a in authors],
        }

    def test_a_matching_record_is_accepted(self):
        doc = Document(
            doc_id="doc_0001",
            title="Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
            authors=["Jason Wei", "Xuezhi Wang"],
        )
        rec = self.record(
            "W7133224126",
            "Chain-Of-Thought Prompting Elicits Reasoning in Large Language Models",
            ["Jason Wei", "Denny Zhou"],
        )
        assert self.source._verify(doc, rec, "doi:x") == "W7133224126"

    def test_a_mismatched_title_is_rejected(self):
        """The real OpenAlex arXiv-DOI corruption, as a regression test."""
        doc = Document(
            doc_id="doc_0001",
            title="Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
            authors=["Jason Wei"],
        )
        rec = self.record(
            "W4221143046",
            "BNAI, NO-TOKEN, and MIND-UNITY: Pillars of a Systemic Revolution in AI",
            ["Someone Else"],
        )
        assert self.source._verify(doc, rec, "doi:10.48550/arXiv.2201.11903") is None

    def test_a_full_title_match_is_not_vetoed_by_author_spellings(self):
        """Deliberate policy: a full title match stands on its own.

        Sources render names inconsistently -- initials only, collective
        names, transliterations, dash variants -- so a disjoint surname set
        is weaker evidence of a different paper than a matching full title is
        of the same one. Letting authors veto here rejected a correct Nature
        resolution in live testing.
        """
        doc = Document(
            doc_id="doc_0001",
            title="A Benchmark for Evaluating Agent Planning",
            authors=["Alice Alpha", "Bob Beta"],
        )
        rec = self.record(
            "W1", "A Benchmark for Evaluating Agent Planning", ["Carol Gamma"]
        )
        assert self.source._verify(doc, rec, "doi:x") == "W1"

    def test_a_truncated_title_with_disjoint_authors_is_rejected(self):
        """Weak title evidence still needs the authors to agree."""
        doc = Document(
            doc_id="doc_0001", title="LLM-Planner", authors=["Alice Alpha"]
        )
        rec = self.record(
            "W1", "LLM-Planner: Few-Shot Grounded Planning", ["Carol Gamma"]
        )
        assert self.source._verify(doc, rec, "doi:x") is None

    def test_a_short_title_is_rescued_by_a_shared_author(self):
        doc = Document(
            doc_id="doc_0002", title="LLM-Planner", authors=["Chan Hee Song"]
        )
        rec = self.record(
            "W4390874280",
            "LLM-Planner: Few-Shot Grounded Planning for Embodied Agents",
            ["Chan Hee Song", "Jiaman Wu"],
        )
        assert self.source._verify(doc, rec, "title search") == "W4390874280"

    def test_a_short_title_alone_is_not_enough(self):
        """No authors on either side means the match cannot be corroborated."""
        doc = Document(doc_id="doc_0002", title="LLM-Planner")
        rec = self.record(
            "W4390874280", "LLM-Planner: Few-Shot Grounded Planning", []
        )
        assert self.source._verify(doc, rec, "title search") is None

    def test_missing_authors_on_one_side_does_not_veto_a_title_match(self):
        doc = Document(
            doc_id="doc_0003",
            title="Reflexion: Language Agents with Verbal Reinforcement Learning",
        )
        rec = self.record(
            "W7133222210",
            "Reflexion: language agents with verbal reinforcement learning",
            ["Noah Shinn"],
        )
        assert self.source._verify(doc, rec, "doi:x") == "W7133222210"

    def test_an_empty_record_is_rejected(self):
        doc = Document(doc_id="doc_0001", title="Anything At All Here")
        assert self.source._verify(doc, None, "doi:x") is None
        assert self.source._verify(doc, {}, "doi:x") is None

    @pytest.mark.asyncio
    async def test_an_openalex_id_needs_no_verification(self):
        """OpenAlex's own identifier is its identity."""
        doc = Document(doc_id="doc_0001", title="T", openalex_id="W123")
        assert await self.source._resolve_work_id(doc) == "W123"


class TestUnicodeFolding:
    """Name and title matching must survive encoding differences.

    OpenAlex writes "Romera‐Paredes" with U+2010 HYPHEN while other sources
    use an ASCII hyphen; without folding, the surnames compared as "paredes"
    against "romera-paredes" and a correct resolution was rejected.
    """

    def test_dash_variants_fold_together(self):
        from litresearch.store.canonicalize import surname

        assert surname("Bernardino Romera‐Paredes") == surname(
            "Bernardino Romera-Paredes"
        )

    def test_accents_fold(self):
        from litresearch.store.canonicalize import surname

        assert surname("Jürgen Müller") == surname("Jurgen Muller")

    def test_titles_fold_accents_and_dashes(self):
        from litresearch.state import normalize_title

        assert normalize_title("Schrödinger–Poisson Systems") == normalize_title(
            "Schrodinger-Poisson Systems"
        )

    def test_accented_titles_match(self):
        assert titles_match(
            "Lévy Flights in Stochastic Optimization Methods",
            "Levy Flights in Stochastic Optimization Methods",
        )


class TestCrossrefReferenceParsing:
    """Crossref reference lists are publisher-supplied and inconsistent."""

    @staticmethod
    def parse(entry: dict):
        from litresearch.sources.crossref import CrossrefSource

        return CrossrefSource._parse_reference(entry)

    def test_article_title_is_preferred(self):
        stub = self.parse(
            {
                "article-title": "Language models are few-shot learners",
                "volume-title": "Advances in Neural Information Processing Systems",
                "author": "Brown",
                "year": "2020",
                "DOI": "10.1/x",
            }
        )
        assert stub.title == "Language models are few-shot learners"
        assert stub.venue == "Advances in Neural Information Processing Systems"
        assert stub.author == "Brown"
        assert stub.year == 2020
        assert stub.doi == "10.1/x"
        assert stub.is_usable()

    def test_a_title_misplaced_in_journal_title_is_recovered(self):
        """Depositors routinely put the cited paper's title in journal-title."""
        stub = self.parse(
            {
                "journal-title": "Do as i can and not as i say: Grounding language "
                "in robotic affordances",
                "author": "Ahn",
                "year": "2022",
            }
        )
        assert stub.title.startswith("Do as i can")
        assert stub.is_usable()

    def test_a_venue_name_is_not_mistaken_for_a_title(self):
        stub = self.parse({"journal-title": "Conference on Robot Learning", "author": "X"})
        assert stub.title is None
        assert stub.venue == "Conference on Robot Learning"
        assert not stub.is_usable()

    def test_a_doi_only_entry_is_usable(self):
        """Hydration will supply the metadata."""
        stub = self.parse({"DOI": "10.1109/CVPR.2018.00387", "doi-asserted-by": "publisher"})
        assert stub.doi == "10.1109/CVPR.2018.00387"
        assert stub.title is None
        assert stub.is_usable()

    def test_an_empty_entry_is_not_usable(self):
        assert not self.parse({"key": "ref12"}).is_usable()

    def test_a_malformed_year_is_tolerated(self):
        assert self.parse({"DOI": "10.1/x", "year": "n.d."}).year is None
        assert self.parse({"DOI": "10.1/x", "year": "2021-03"}).year == 2021


class TestCrossrefBackwardGuards:
    async def test_an_arxiv_doi_makes_no_request(self, monkeypatch):
        """arXiv DOIs are registered with DataCite and carry no references."""
        from litresearch.sources.crossref import CrossrefSource

        source = CrossrefSource()

        async def fail(*args, **kwargs):
            raise AssertionError("should not have issued a request")

        monkeypatch.setattr(source, "get_json", fail)
        doc = Document(doc_id="d", title="T", doi="10.48550/arXiv.2201.11903")
        assert await source.references(doc) == []

    async def test_no_doi_makes_no_request(self, monkeypatch):
        from litresearch.sources.crossref import CrossrefSource

        source = CrossrefSource()

        async def fail(*args, **kwargs):
            raise AssertionError("should not have issued a request")

        monkeypatch.setattr(source, "get_json", fail)
        assert await source.references(Document(doc_id="d", title="T")) == []

    async def test_sparse_references_still_become_candidates(self, monkeypatch):
        """With hydration unavailable, titled entries survive as stubs."""
        from litresearch.sources import crossref as crossref_module

        source = crossref_module.CrossrefSource()

        async def fake_get_json(url, params=None, headers=None):
            return {
                "message": {
                    "reference": [
                        {"article-title": "A foundational study of planning",
                         "author": "Fikes", "year": "1971"},
                        {"DOI": "10.1/unhydrated"},
                        {"key": "junk"},
                    ]
                }
            }

        async def no_hydration(*args, **kwargs):
            return {}

        monkeypatch.setattr(source, "get_json", fake_get_json)
        monkeypatch.setattr(crossref_module, "hydrate_by_doi", no_hydration)

        hits = await source.references(
            Document(doc_id="doc_0001", title="Seed", doi="10.1/seed")
        )
        # The titled entry survives; the bare unhydrated DOI and the junk
        # entry are dropped, having nothing for screening to judge.
        assert [h.title for h in hits] == ["A foundational study of planning"]
        assert hits[0].provenance.via == "backward_snowball"
        assert hits[0].provenance.seed_doc_id == "doc_0001"
