"""Tests for document deduplication and canonicalization.

This is the component where a bug is most expensive: a false merge silently
destroys evidence, and a missed merge wastes reading budget on the same paper
several times.
"""

from __future__ import annotations

from litresearch.state import Provenance, RawHit, normalize_arxiv_id, normalize_doi
from litresearch.store.canonicalize import Canonicalizer


def hit(
    title: str,
    *,
    source: str = "arxiv",
    doi: str | None = None,
    arxiv_id: str | None = None,
    pubmed_id: str | None = None,
    s2: str | None = None,
    authors: list[str] | None = None,
    year: int | None = 2024,
    abstract: str | None = "An abstract.",
    sq: str = "RQ1",
    intent: str = "RQ1-i1",
) -> RawHit:
    return RawHit(
        source=source,
        source_id=title[:20],
        title=title,
        abstract=abstract,
        authors=authors if authors is not None else ["Jane Doe", "John Roe"],
        year=year,
        doi=doi,
        arxiv_id=arxiv_id,
        pubmed_id=pubmed_id,
        s2_paper_id=s2,
        provenance=Provenance(
            subquestion_id=sq, intent_id=intent, source=source, query="q"
        ),
    )


class TestIdentifierNormalization:
    def test_doi_forms_collapse(self):
        assert (
            normalize_doi("https://doi.org/10.1234/ABC")
            == normalize_doi("doi:10.1234/abc")
            == normalize_doi("10.1234/abc")
            == "10.1234/abc"
        )

    def test_arxiv_versions_and_urls_collapse(self):
        assert (
            normalize_arxiv_id("arXiv:2301.12345v3")
            == normalize_arxiv_id("https://arxiv.org/abs/2301.12345")
            == normalize_arxiv_id("https://arxiv.org/pdf/2301.12345v1.pdf")
            == "2301.12345"
        )


class TestTier1And2:
    def test_same_doi_merges_across_sources(self):
        canon = Canonicalizer({})
        result = canon.ingest(
            [
                hit("Planning With Language Models", source="arxiv", doi="10.1/x"),
                hit(
                    "Planning with Language Models",
                    source="crossref",
                    doi="https://doi.org/10.1/X",
                ),
            ]
        )
        assert result.new_count == 1
        assert result.merged_count == 1
        doc = next(iter(result.documents.values()))
        assert set(doc.sources) == {"arxiv", "crossref"}
        assert len(doc.discovered_by) == 2

    def test_same_arxiv_id_merges_despite_title_drift(self):
        canon = Canonicalizer({})
        result = canon.ingest(
            [
                hit("Agents That Plan", arxiv_id="2301.00001v1"),
                hit(
                    "Agents That Plan: An Extended Study",
                    source="semantic_scholar",
                    arxiv_id="arXiv:2301.00001",
                ),
            ]
        )
        assert len(result.documents) == 1

    def test_identifier_learned_later_is_indexed(self):
        """A DOI arriving on a second record must become a usable dedup key."""
        canon = Canonicalizer({})
        canon.ingest([hit("A Study Of Plan Repair Mechanisms", arxiv_id="2301.00002")])
        canon.ingest(
            [
                hit(
                    "A Study of Plan Repair Mechanisms",
                    source="crossref",
                    doi="10.9/late",
                    arxiv_id="2301.00002",
                )
            ]
        )
        # A third source that knows only the DOI should still match.
        result = canon.ingest(
            [hit("Plan Repair Mechanisms", source="pubmed", doi="10.9/late")]
        )
        assert result.new_count == 0
        assert result.merged_count == 1
        assert len(canon.corpus) == 1


class TestTier3TitleAuthor:
    def test_normalized_title_and_author_merge(self):
        canon = Canonicalizer({})
        result = canon.ingest(
            [
                hit("Plan Repair in LLM Agents!", doi=None),
                hit("plan repair in llm agents", source="openalex", doi=None),
            ]
        )
        assert len(result.documents) == 1

    def test_web_hit_without_authors_matches_on_title(self):
        canon = Canonicalizer({})
        result = canon.ingest(
            [
                hit("Hierarchical Task Networks For Agents", doi=None),
                hit(
                    "Hierarchical Task Networks for Agents",
                    source="web",
                    authors=[],
                    doi=None,
                ),
            ]
        )
        assert len(result.documents) == 1


class TestTier4Fuzzy:
    def test_near_identical_titles_merge_with_corroboration(self):
        canon = Canonicalizer({})
        result = canon.ingest(
            [
                hit(
                    "Reflexion: Language Agents with Verbal Reinforcement Learning",
                    doi=None,
                ),
                hit(
                    "Reflexion: Language Agents with Verbal Reinforcement Learning.",
                    source="openalex",
                    authors=["Jane Doe"],
                    doi=None,
                ),
            ]
        )
        assert len(result.documents) == 1

    def test_paper_series_does_not_merge(self):
        """Titles differing by a part number are different papers."""
        canon = Canonicalizer({})
        result = canon.ingest(
            [
                hit("Automated Planning Foundations Part I Theory", doi=None),
                hit("Automated Planning Foundations Part II Practice", doi=None),
            ]
        )
        assert len(result.documents) == 2

    def test_disjoint_authors_block_a_fuzzy_merge(self):
        canon = Canonicalizer({})
        result = canon.ingest(
            [
                hit(
                    "A Benchmark For Evaluating Agent Planning Capability",
                    authors=["Alice Alpha", "Bob Beta"],
                    doi=None,
                ),
                hit(
                    "A Benchmark for Evaluating Agent Planning Capability",
                    authors=["Carol Gamma", "Dave Delta"],
                    source="openalex",
                    doi=None,
                ),
            ]
        )
        assert len(result.documents) == 2

    def test_distant_years_block_a_fuzzy_merge(self):
        canon = Canonicalizer({})
        result = canon.ingest(
            [
                hit(
                    "Survey Of Planning Methods For Autonomous Agents",
                    year=2015,
                    authors=[],
                    doi=None,
                ),
                hit(
                    "Survey of Planning Methods for Autonomous Agents",
                    year=2024,
                    authors=[],
                    source="openalex",
                    doi=None,
                ),
            ]
        )
        assert len(result.documents) == 2


class TestProvenanceAndSharing:
    def test_provenance_accumulates_across_subquestions(self):
        """The same paper found for two subquestions keeps both discoveries."""
        canon = Canonicalizer({})
        canon.ingest([hit("Shared Paper On Agent Planning", doi="10.5/shared", sq="RQ1")])
        result = canon.ingest(
            [
                hit(
                    "Shared Paper On Agent Planning",
                    doi="10.5/shared",
                    sq="RQ3",
                    intent="RQ3-i2",
                    source="openalex",
                )
            ]
        )
        doc = next(iter(result.documents.values()))
        assert {p.subquestion_id for p in doc.discovered_by} == {"RQ1", "RQ3"}

    def test_duplicate_provenance_is_not_repeated(self):
        canon = Canonicalizer({})
        canon.ingest([hit("Repeat Paper About Planning Agents", doi="10.6/r")])
        result = canon.ingest([hit("Repeat Paper About Planning Agents", doi="10.6/r")])
        doc = next(iter(result.documents.values()))
        assert len(doc.discovered_by) == 1

    def test_richer_record_wins_on_merge(self):
        canon = Canonicalizer({})
        canon.ingest(
            [hit("Rich Metadata Paper On Planning", doi="10.7/rich", abstract="Short.")]
        )
        long_abstract = "A considerably longer and more informative abstract. " * 5
        result = canon.ingest(
            [
                hit(
                    "Rich Metadata Paper On Planning",
                    doi="10.7/rich",
                    abstract=long_abstract,
                    source="semantic_scholar",
                )
            ]
        )
        doc = next(iter(result.documents.values()))
        assert doc.abstract == long_abstract


class TestRobustness:
    def test_junk_titles_are_skipped(self):
        canon = Canonicalizer({})
        result = canon.ingest([hit("Short", doi=None)])
        assert result.new_count == 0
        assert result.skipped_count == 1

    def test_ids_are_unique_and_sequential_across_batches(self):
        canon = Canonicalizer({})
        canon.ingest(
            [
                hit("Plan Generation Via Tree Search", doi="10.8/a"),
                hit("Task Decomposition For Long Horizons", doi="10.8/b"),
                hit("Monitoring Execution Of Agent Plans", doi="10.8/c"),
            ]
        )
        result = canon.ingest(
            [
                hit("Benchmarks For Evaluating Agent Planning", doi="10.9/a"),
                hit("Limitations Of Current Planning Approaches", doi="10.9/b"),
            ]
        )
        assert sorted(canon.corpus) == [
            "doc_0001",
            "doc_0002",
            "doc_0003",
            "doc_0004",
            "doc_0005",
        ]
        assert sorted(result.documents) == ["doc_0004", "doc_0005"]

    def test_existing_corpus_is_respected(self):
        canon = Canonicalizer({})
        first = canon.ingest([hit("Original Paper About Agent Plans", doi="10.10/a")])

        # Simulate a later iteration, reconstructed from persisted state.
        resumed = Canonicalizer(canon.corpus, {"doi:10.10/a": "doc_0001"})
        result = resumed.ingest(
            [hit("Original Paper About Agent Plans", doi="10.10/a", source="openalex")]
        )
        assert result.new_count == 0
        assert result.merged_count == 1
        assert list(first.documents) == ["doc_0001"]


class TestNumericSignature:
    """Numerals in titles distinguish works that are otherwise near-identical."""

    def test_model_generations_do_not_merge(self):
        from litresearch.store.canonicalize import numeric_signature

        canon = Canonicalizer({})
        result = canon.ingest(
            [
                hit(
                    "Llama 2: Open Foundation and Fine-Tuned Chat Models",
                    year=2023,
                    doi=None,
                ),
                hit(
                    "Llama 3: Open Foundation and Fine-Tuned Chat Models",
                    year=2024,
                    source="openalex",
                    doi=None,
                ),
            ]
        )
        assert numeric_signature("Llama 2: Open Foundation") == ("2",)
        assert len(result.documents) == 2

    def test_roman_numeral_parts_do_not_merge(self):
        canon = Canonicalizer({})
        result = canon.ingest(
            [
                hit("Planning Foundations Part II Of The Series", doi=None),
                hit(
                    "Planning Foundations Part III Of The Series",
                    source="openalex",
                    doi=None,
                ),
            ]
        )
        assert len(result.documents) == 2

    def test_matching_numerals_still_merge(self):
        canon = Canonicalizer({})
        result = canon.ingest(
            [
                hit("GPT-4 Technical Report On Reasoning", doi=None),
                hit("GPT-4 Technical Report on Reasoning.", source="openalex", doi=None),
            ]
        )
        assert len(result.documents) == 1


class TestIngestIsolation:
    def test_each_ingest_reports_only_its_own_changes(self):
        canon = Canonicalizer({})
        first = canon.ingest([hit("First Paper About Agent Planning", doi="10.20/a")])
        assert first.new_count == 1 and first.merged_count == 0

        second = canon.ingest([hit("Second Paper About Agent Planning", doi="10.20/b")])
        assert second.new_count == 1
        assert second.merged_count == 0
        assert list(second.documents) == ["doc_0002"]

        third = canon.ingest([hit("First Paper About Agent Planning", doi="10.20/a")])
        assert third.new_count == 0
        assert third.merged_count == 1
        assert len(canon.corpus) == 2
