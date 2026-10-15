"""Tests for source adapters against recorded payload shapes.

No network access: each adapter's parser is fed a payload in the shape the
real API returns. These catch the failure mode that matters here -- a quiet
parsing change that drops abstracts or identifiers and therefore silently
degrades dedup and evidence extraction.
"""

from __future__ import annotations

import pytest

from forestresearch.sources.arxiv import ArxivSource
from forestresearch.sources.crossref import CrossrefSource, strip_jats
from forestresearch.sources.openalex import OpenAlexSource, reconstruct_abstract
from forestresearch.sources.pubmed import PubMedSource
from forestresearch.sources.semantic_scholar import SemanticScholarSource
from forestresearch.sources.web import WebSource
from forestresearch.state import Document, QueryPlan


@pytest.fixture
def plan() -> QueryPlan:
    return QueryPlan(
        intent_id="RQ1-i1", subquestion_id="RQ1", source="x", query="agent planning"
    )


ARXIV_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2305.16291v2</id>
    <published>2023-05-25T17:49:35Z</published>
    <title>Voyager: An Open-Ended Embodied Agent
      with Large Language Models</title>
    <summary>  We introduce Voyager, the first LLM-powered embodied
  lifelong learning agent.
    </summary>
    <author><name>Guanzhi Wang</name></author>
    <author><name>Yuqi Xie</name></author>
    <arxiv:doi>10.5555/voyager</arxiv:doi>
    <arxiv:journal_ref>TMLR 2024</arxiv:journal_ref>
    <link href="http://arxiv.org/abs/2305.16291v2" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2305.16291v2" rel="related"/>
    <arxiv:primary_category term="cs.AI"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2310.00001v1</id>
    <published>2023-10-01T00:00:00Z</published>
    <title>A Preprint With No Journal Reference</title>
    <summary>Abstract text.</summary>
    <author><name>Solo Author</name></author>
    <arxiv:primary_category term="cs.LG"/>
  </entry>
</feed>"""


EMPTY_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <opensearch:totalResults xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
    >0</opensearch:totalResults>
</feed>"""


class TestArxiv:
    def test_parses_entries(self, plan):
        hits = ArxivSource()._parse(ARXIV_ATOM, plan)
        assert len(hits) == 2
        first = hits[0]
        # Newlines inside a title must collapse, or dedup keys will not match.
        assert first.title == "Voyager: An Open-Ended Embodied Agent with Large Language Models"
        assert first.arxiv_id == "2305.16291v2"
        assert first.year == 2023
        assert first.authors == ["Guanzhi Wang", "Yuqi Xie"]
        assert first.doi == "10.5555/voyager"
        assert first.pdf_url.endswith("2305.16291v2")
        assert "lifelong learning agent" in first.abstract
        assert first.provenance.subquestion_id == "RQ1"

    def test_journal_ref_marks_it_published(self, plan):
        hits = ArxivSource()._parse(ARXIV_ATOM, plan)
        assert hits[0].is_preprint is False
        assert hits[1].is_preprint is True

    def test_pdf_url_is_synthesized_when_absent(self, plan):
        hits = ArxivSource()._parse(ARXIV_ATOM, plan)
        assert hits[1].pdf_url == "https://arxiv.org/pdf/2310.00001v1"

    def test_malformed_xml_returns_nothing(self, plan):
        assert ArxivSource()._parse("<not xml", plan) == []

    def test_query_uses_field_prefixes_and_preserves_phrases(self):
        source = ArxivSource()
        built = source._build_query(
            QueryPlan(
                intent_id="i", subquestion_id="RQ1", source="arxiv",
                query='"LLM agents" planning failure', year_from=2023,
            )
        )
        assert 'all:"LLM agents"' in built
        assert "all:planning" in built
        assert "submittedDate:[2023" in built


S2_PAYLOAD = {
    "data": [
        {
            "paperId": "abc123",
            "externalIds": {"DOI": "10.1/x", "ArXiv": "2302.01234", "PubMed": "99887766"},
            "title": "Reflexion: Language Agents with Verbal Reinforcement Learning",
            "abstract": "We propose Reflexion.",
            "year": 2023,
            "venue": "NeurIPS",
            "authors": [{"name": "Noah Shinn"}, {"name": None}],
            "citationCount": 1200,
            "referenceCount": 60,
            "publicationTypes": ["JournalArticle", "Conference"],
            "openAccessPdf": {"url": "https://example.org/p.pdf"},
            "url": "https://semanticscholar.org/paper/abc123",
        },
        {"paperId": "no-title", "title": ""},
    ]
}


class TestSemanticScholar:
    def test_normalizes_external_ids(self, plan):
        source = SemanticScholarSource()
        hits = [
            h
            for h in (
                source._to_hit(p, source.provenance_for(plan, i))
                for i, p in enumerate(S2_PAYLOAD["data"])
            )
            if h
        ]
        assert len(hits) == 1  # the untitled record is dropped
        hit = hits[0]
        assert hit.doi == "10.1/x"
        assert hit.arxiv_id == "2302.01234"
        assert hit.pubmed_id == "99887766"
        assert hit.s2_paper_id == "abc123"
        assert hit.citation_count == 1200
        assert hit.authors == ["Noah Shinn"]  # null author name filtered out
        assert hit.is_preprint is False  # it has a venue

    def test_resolves_id_in_preference_order(self):
        resolve = SemanticScholarSource._resolve_id
        assert resolve(Document(doc_id="d", title="t", s2_paper_id="s2id")) == "s2id"
        assert resolve(Document(doc_id="d", title="t", doi="10.1/x")) == "DOI:10.1/x"
        assert resolve(Document(doc_id="d", title="t", arxiv_id="2301.1")) == "ARXIV:2301.1"
        assert resolve(Document(doc_id="d", title="t", pubmed_id="777")) == "PMID:777"
        assert resolve(Document(doc_id="d", title="t")) is None


class TestOpenAlex:
    def test_reconstructs_inverted_abstract(self):
        index = {"Agents": [0], "that": [1], "plan": [2, 5], "can": [3], "also": [4]}
        assert reconstruct_abstract(index) == "Agents that plan can also plan"

    def test_empty_inverted_index(self):
        assert reconstruct_abstract(None) is None
        assert reconstruct_abstract({}) is None

    def test_normalizes_a_work(self, plan):
        work = {
            "id": "https://openalex.org/W123456",
            "doi": "https://doi.org/10.2/y",
            "display_name": "Task Decomposition for Agents",
            "publication_year": 2024,
            "type": "article",
            "cited_by_count": 42,
            "referenced_works_count": 31,
            "authorships": [
                {"author": {"display_name": "Ada Lovelace"}},
                {"author": {}},
            ],
            "primary_location": {
                "landing_page_url": "https://journal.example/article",
                "source": {"display_name": "Journal of Agents", "type": "journal"},
            },
            "open_access": {"oa_url": "https://journal.example/pdf"},
            "abstract_inverted_index": {"Short": [0], "abstract": [1]},
            "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/12345"},
        }
        hit = OpenAlexSource()._to_hit(work, OpenAlexSource().provenance_for(plan, 0))
        assert hit.openalex_id == "W123456"
        assert hit.doi == "https://doi.org/10.2/y"  # normalized later, at canonicalization
        assert hit.pubmed_id == "12345"
        assert hit.authors == ["Ada Lovelace"]
        assert hit.venue == "Journal of Agents"
        assert hit.abstract == "Short abstract"
        assert hit.is_preprint is False

    def test_repository_location_is_a_preprint(self, plan):
        work = {
            "id": "https://openalex.org/W9",
            "display_name": "A Preprint On Planning",
            "type": "preprint",
            "primary_location": {"source": {"type": "repository", "display_name": "arXiv"}},
        }
        hit = OpenAlexSource()._to_hit(work, OpenAlexSource().provenance_for(plan, 0))
        assert hit.is_preprint is True


class TestCrossref:
    def test_strips_jats_markup(self):
        jats = "<jats:p>Abstract We study planning.</jats:p>"
        assert strip_jats(jats) == "We study planning."
        assert strip_jats(None) is None

    def test_normalizes_an_item(self, plan):
        item = {
            "DOI": "10.3/z",
            "title": ["Monitoring   Agent\nPlans"],
            "abstract": "<jats:p>We monitor plans.</jats:p>",
            "author": [
                {"given": "Grace", "family": "Hopper"},
                {"family": "OnlyFamily"},
                {},
            ],
            "issued": {"date-parts": [[2022, 5, 1]]},
            "container-title": ["ACM Computing Surveys"],
            "is-referenced-by-count": 17,
            "references-count": 80,
            "type": "journal-article",
            "URL": "https://doi.org/10.3/z",
            "link": [
                {"content-type": "text/html", "URL": "https://x/html"},
                {"content-type": "application/pdf", "URL": "https://x/pdf"},
            ],
        }
        hit = CrossrefSource()._to_hit(item, plan, 0)
        assert hit.title == "Monitoring Agent Plans"
        assert hit.authors == ["Grace Hopper", "OnlyFamily"]
        assert hit.year == 2022
        assert hit.venue == "ACM Computing Surveys"
        assert hit.abstract == "We monitor plans."
        assert hit.pdf_url == "https://x/pdf"
        assert hit.citation_count == 17

    def test_posted_content_is_a_preprint(self, plan):
        hit = CrossrefSource()._to_hit(
            {"DOI": "10.4/p", "title": ["A Posted Preprint"], "type": "posted-content"}, plan, 0
        )
        assert hit.is_preprint is True

    def test_missing_title_is_dropped(self, plan):
        assert CrossrefSource()._to_hit({"DOI": "10.5/q", "title": []}, plan, 0) is None


PUBMED_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
 <PubmedArticle>
  <MedlineCitation>
   <PMID>31234567</PMID>
   <Article>
    <Journal><Title>Nature Medicine</Title>
     <JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue>
    </Journal>
    <ArticleTitle>Clinical <i>decision</i> support with language models</ArticleTitle>
    <Abstract>
     <AbstractText Label="BACKGROUND">Clinicians need support.</AbstractText>
     <AbstractText Label="RESULTS">Accuracy improved by 12%.</AbstractText>
    </Abstract>
    <AuthorList>
     <Author><LastName>Chen</LastName><ForeName>Wei</ForeName></Author>
     <Author><CollectiveName>The Study Group</CollectiveName></Author>
    </AuthorList>
    <PublicationTypeList>
     <PublicationType>Journal Article</PublicationType>
     <PublicationType>Review</PublicationType>
    </PublicationTypeList>
   </Article>
  </MedlineCitation>
  <PubmedData><ArticleIdList>
   <ArticleId IdType="pubmed">31234567</ArticleId>
   <ArticleId IdType="doi">10.1038/nm.1234</ArticleId>
  </ArticleIdList></PubmedData>
 </PubmedArticle>
</PubmedArticleSet>"""


class TestPubMed:
    def test_parses_structured_abstract_and_ids(self, plan):
        from xml.etree import ElementTree

        root = ElementTree.fromstring(PUBMED_XML)
        article = root.find(".//PubmedArticle")
        hit = PubMedSource()._parse_article(article, plan, {"31234567": 0})
        # Inline markup must be flattened, not dropped.
        assert hit.title == "Clinical decision support with language models"
        assert hit.pubmed_id == "31234567"
        assert hit.doi == "10.1038/nm.1234"
        assert hit.year == 2021
        assert hit.venue == "Nature Medicine"
        assert "BACKGROUND: Clinicians need support." in hit.abstract
        assert "RESULTS: Accuracy improved by 12%." in hit.abstract
        assert hit.authors == ["Wei Chen", "The Study Group"]
        assert "Review" in hit.publication_types

    def test_review_type_blocks_primary_research(self, plan):
        from xml.etree import ElementTree

        root = ElementTree.fromstring(PUBMED_XML)
        hit = PubMedSource()._parse_article(root.find(".//PubmedArticle"), plan, {})
        doc = Document.from_hit(hit, "doc_0001")
        assert doc.is_primary_research is False


class TestWeb:
    def test_extracts_scholarly_ids_from_urls(self, plan):
        source = WebSource()
        hit = source._to_hit(
            {
                "title": "Voyager project page",
                "url": "https://arxiv.org/abs/2305.16291",
                "content": "An LLM-powered embodied agent.",
            },
            plan,
            0,
        )
        # Recognizing the arXiv id is what lets this merge with the arXiv hit
        # instead of becoming a duplicate corpus entry.
        assert hit.arxiv_id == "2305.16291"
        assert hit.venue == "arxiv.org"
        assert hit.publication_types == ["web"]

    def test_extracts_doi_and_pmid(self, plan):
        source = WebSource()
        assert source._to_hit(
            {"title": "A paper", "url": "https://publisher.com/10.1038/s41586-023-06924-6"},
            plan, 0,
        ).doi == "10.1038/s41586-023-06924-6"
        assert source._to_hit(
            {"title": "A paper", "url": "https://pubmed.ncbi.nlm.nih.gov/31234567"}, plan, 0
        ).pubmed_id == "31234567"

    def test_requires_title_and_url(self, plan):
        source = WebSource()
        assert source._to_hit({"title": "", "url": "https://x"}, plan, 0) is None
        assert source._to_hit({"title": "T", "url": ""}, plan, 0) is None

    def test_unavailable_without_api_key(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        assert WebSource().is_available() is False


class TestFieldListRegressions:
    """Guards for live-API constraints that are easy to break silently.

    Each of these cost a whole source, or a whole capability, when wrong: the
    APIs reject the entire request rather than ignoring the bad field.
    """

    def test_citation_endpoints_avoid_the_nested_author_field(self):
        from forestresearch.sources.semantic_scholar import CITATION_FIELDS, PAPER_FIELDS

        # /paper/{id}/references and /citations 400 on "authors.name" with
        # "Unrecognized or unsupported fields", while /paper/search requires it.
        assert "authors.name" not in CITATION_FIELDS
        assert "authors" in CITATION_FIELDS.split(",")
        assert "authors.name" in PAPER_FIELDS

    def test_crossref_select_omits_subtype(self):
        from forestresearch.sources.crossref import SELECT_FIELDS

        # Crossref rejects the whole /works request if any select field is
        # invalid, and "subtype" is not selectable on that route.
        assert "subtype" not in SELECT_FIELDS.split(",")


class TestSemanticScholarIdResolution:
    def test_bare_arxiv_id_is_tried_before_an_arxiv_doi(self):
        """S2 indexes 10.48550/arXiv.* papers by arXiv id, not by that DOI."""
        doc = Document(
            doc_id="doc_0001",
            title="Chain-of-Thought Prompting",
            arxiv_id="2201.11903",
            doi="10.48550/arXiv.2201.11903",
        )
        candidates = SemanticScholarSource._candidate_ids(doc)
        assert candidates[0] == "ARXIV:2201.11903"
        # The DOI is still attempted, just last.
        assert "DOI:10.48550/arXiv.2201.11903" in candidates

    def test_a_real_doi_outranks_the_arxiv_id(self):
        doc = Document(
            doc_id="doc_0001", title="T", arxiv_id="2201.11903", doi="10.1109/iccv.2023.1"
        )
        assert SemanticScholarSource._candidate_ids(doc)[0] == "DOI:10.1109/iccv.2023.1"

    def test_s2_paper_id_always_wins(self):
        doc = Document(doc_id="doc_0001", title="T", s2_paper_id="abc", doi="10.1/x")
        assert SemanticScholarSource._candidate_ids(doc)[0] == "abc"

    def test_no_identifiers_yields_no_candidates(self):
        assert SemanticScholarSource._candidate_ids(Document(doc_id="d", title="T")) == []


class TestSemanticScholarPreprintDetection:
    def test_a_preprint_server_venue_is_not_peer_review(self, plan):
        source = SemanticScholarSource()
        hit = source._to_hit(
            {
                "paperId": "x",
                "title": "Beyond the Imitation Game",
                "venue": "arXiv.org",
                "externalIds": {"ArXiv": "2206.04615"},
            },
            source.provenance_for(plan, 0),
        )
        assert hit.is_preprint is True

    def test_a_journal_venue_means_published(self, plan):
        source = SemanticScholarSource()
        hit = source._to_hit(
            {
                "paperId": "y",
                "title": "Emergent Abilities of Large Language Models",
                "venue": "Trans. Mach. Learn. Res.",
                "externalIds": {"ArXiv": "2206.07682"},
            },
            source.provenance_for(plan, 0),
        )
        assert hit.is_preprint is False

    def test_no_venue_with_an_arxiv_id_is_a_preprint(self, plan):
        source = SemanticScholarSource()
        hit = source._to_hit(
            {"paperId": "z", "title": "A new preprint", "externalIds": {"ArXiv": "2301.1"}},
            source.provenance_for(plan, 0),
        )
        assert hit.is_preprint is True


class TestArxivQueryRelaxation:
    """arXiv ANDs its clauses, so a verbose query returns literally nothing.

    This is not hypothetical: on a live run every arXiv query a local model
    generated returned zero results while OpenAlex and Crossref returned 40
    documents for the same intents. Measured on one of those queries, ANDing
    all eight terms gave 0 hits, the four most distinctive gave 82, and three
    gave 514.
    """

    def plan_for(self, query: str, **kwargs) -> QueryPlan:
        return QueryPlan(
            intent_id="i", subquestion_id="RQ1", source="arxiv", query=query, **kwargs
        )

    def test_research_generic_words_are_dropped(self):
        clauses = ArxivSource()._clauses(
            self.plan_for(
                "hybrid architectures for AI agent planning combining LLMs "
                "and traditional search methods"
            )
        )
        terms = [c.removeprefix("all:") for c in clauses]
        # The topical words survive...
        assert terms == ["hybrid", "agent", "planning", "LLMs", "search"]
        # ...and the vocabulary that matches every arXiv paper does not.
        for generic in ("architectures", "methods", "traditional", "combining"):
            assert generic not in terms

    def test_written_order_is_preserved(self):
        """Sorting by length ranked "traditional" above "LLMs"."""
        clauses = ArxivSource()._clauses(
            self.plan_for("grounding high-level plans into executable actions")
        )
        assert [c.removeprefix("all:") for c in clauses] == [
            "grounding",
            "high-level",
            "plans",
            "executable",
            "actions",
        ]

    def test_quoted_phrases_lead(self):
        clauses = ArxivSource()._clauses(
            self.plan_for('planning failure "large language model" detection')
        )
        assert clauses[0] == 'all:"large language model"'

    def test_a_query_of_only_generic_words_still_searches(self):
        clauses = ArxivSource()._clauses(self.plan_for("recent developments"))
        assert clauses == ['all:"recent developments"']

    def test_date_filter_is_applied_at_every_width(self):
        source = ArxivSource()
        plan = self.plan_for("agent planning failure recovery", year_from=2023)
        clauses = source._clauses(plan)
        for width in (len(clauses), 3):
            assert "submittedDate:[2023" in source._assemble(clauses[:width], plan)

    async def test_an_empty_result_triggers_relaxation(self, monkeypatch):
        source = ArxivSource()
        plan = self.plan_for(
            "theoretical algorithmic long-horizon planning sequential decision-making"
        )
        widths: list[int] = []

        async def fake_get_text(url, params=None):
            widths.append(params["search_query"].count(" AND ") + 1)
            # Only the relaxed, 4-clause query returns anything.
            return ARXIV_ATOM if widths[-1] == 4 else EMPTY_ATOM

        monkeypatch.setattr(source, "get_text", fake_get_text)
        hits = await source.search(plan)
        assert widths == [6, 4]
        assert len(hits) == 2

    async def test_a_working_query_is_not_relaxed(self, monkeypatch):
        """Relaxation costs a request at 3s pacing, so it must not run eagerly."""
        source = ArxivSource()
        calls = []

        async def fake_get_text(url, params=None):
            calls.append(params["search_query"])
            return ARXIV_ATOM

        monkeypatch.setattr(source, "get_text", fake_get_text)
        hits = await source.search(self.plan_for("agent planning failure recovery"))
        assert len(calls) == 1
        assert len(hits) == 2

    async def test_relaxation_gives_up_rather_than_looping(self, monkeypatch):
        source = ArxivSource()
        calls = []

        async def fake_get_text(url, params=None):
            calls.append(params["search_query"])
            return EMPTY_ATOM

        monkeypatch.setattr(source, "get_text", fake_get_text)
        hits = await source.search(
            self.plan_for("one two three four five six seven eight")
        )
        assert hits == []
        # Full width, then 4, then 3 -- and no further.
        assert len(calls) == 3


class TestWebYear:
    """Publication years must come from dates, never from incidental digits.

    Every URL below produced a year on a live run -- 2089, 2097, 2076, 2030
    -- because the first 19xx/20xx digit run anywhere in the URL was taken.
    """

    @pytest.mark.parametrize(
        "url",
        [
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC12089777",
            "https://www.linkedin.com/posts/someone-376142244_tools-activity-7457744824209784832-5iAY",
            "https://www.mdpi.com/2076-3417/16/8/3966",
            "https://www.sciencedirect.com/science/article/abs/pii/S0099133320300938",
            "https://example.com/p/2024-1234",
            "https://example.com/page?id=2024",
        ],
    )
    def test_digit_runs_in_urls_are_not_years(self, url):
        assert WebSource._year("", url) is None

    @pytest.mark.parametrize(
        ("url", "year"),
        [
            ("https://blog.example.com/2024/05/12/a-post", 2024),
            ("https://example.com/news/2025-03-01-launch", 2025),
            ("https://example.com/archive/2023/", 2023),
            ("https://example.com/2022", 2022),
        ],
    )
    def test_date_shaped_url_paths_give_a_year(self, url, year):
        assert WebSource._year("", url) == year

    def test_published_date_wins_over_the_url(self):
        assert WebSource._year("2025-02-02T00:00:00Z", "https://x.com/2019/01/a") == 2025

    def test_future_years_are_rejected(self):
        assert WebSource._year("", "https://example.com/2099/01/x") is None
        assert WebSource._year("2099-01-01", "https://example.com/") is None

    def test_to_hit_uses_the_safe_parser(self, plan):
        hit = WebSource()._to_hit(
            {"title": "ToR-Lite", "url": "https://www.mdpi.com/2076-3417/16/8/3966"},
            plan,
            0,
        )
        assert hit.year is None


class TestRateLimitHandling:
    """A rate-limited source must back off, and must not fail silently.

    On a live run arXiv answered every request with 429. Retries landed 1.5s
    and 3s later, inside its roughly 30s penalty, and the search round then
    reported "arxiv:0" for 52 queries -- exactly what a source with nothing
    relevant reports.
    """

    @pytest.fixture
    def responses(self, monkeypatch):
        """Script the HTTP responses; record requests and requested sleeps."""
        import asyncio as _asyncio

        import httpx

        from forestresearch.sources import base

        script: list[int] = []
        seen: list[str] = []
        sleeps: list[float] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            status = script.pop(0) if script else 200
            return httpx.Response(status, text="<feed/>")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        monkeypatch.setattr(base, "get_http_client", lambda: client)

        real_sleep = _asyncio.sleep

        async def fake_sleep(delay, *args, **kwargs):
            sleeps.append(delay)
            await real_sleep(0)

        monkeypatch.setattr(_asyncio, "sleep", fake_sleep)
        return script, seen, sleeps

    async def test_429_holds_the_source_for_the_cooldown(self, responses):
        script, seen, sleeps = responses
        script.extend([429, 200])
        source = ArxivSource()

        assert await source.get_text("https://export.arxiv.org/api/query") == "<feed/>"
        assert len(seen) == 2
        # The retry waited out arXiv's cooldown, not just tenacity's 1.5s.
        assert max(sleeps) >= ArxivSource.rate_limit_cooldown - 1
        assert source.failures == {}

    async def test_429_widens_the_interval_and_success_eases_it(self, responses):
        script, _, _ = responses
        script.extend([429, 200])
        source = ArxivSource()

        await source.get_text("https://export.arxiv.org/api/query")
        # Doubled to 6s by the 429, then eased by one success.
        assert source.limiter.min_interval == pytest.approx(4.5)
        for _ in range(5):
            await source.get_text("https://export.arxiv.org/api/query")
        assert source.limiter.min_interval == source.limiter.base_interval

    async def test_exhausted_retries_are_recorded_by_reason(self, responses):
        script, seen, _ = responses
        script.extend([429, 429, 429])
        source = ArxivSource()

        assert await source.get_text("https://export.arxiv.org/api/query") is None
        assert len(seen) == 3
        assert source.failures == {"HTTP 429 rate limited": 1}

    async def test_a_persistently_failing_source_is_skipped(self, responses):
        script, seen, _ = responses
        script.extend([503] * 9)
        source = ArxivSource()

        for _ in range(source.trip_after):
            assert await source.get_text("https://export.arxiv.org/api/query") is None
        requests_made = len(seen)

        # Tripped: the next call fails at once instead of waiting out retries.
        assert await source.get_text("https://export.arxiv.org/api/query") is None
        assert len(seen) == requests_made
        assert source.failures["skipped while unavailable"] == 1
        assert source.failures["HTTP 503"] == source.trip_after

    async def test_client_errors_do_not_trip_the_source(self, responses):
        script, _, _ = responses
        script.extend([404] * 5)
        source = ArxivSource()
        for _ in range(5):
            assert await source.get_text("https://export.arxiv.org/api/query") is None
        assert source.failures == {}

    async def test_search_round_reports_failed_requests(self, monkeypatch):
        from forestresearch.nodes import search

        class FailingSource(ArxivSource):
            async def search(self, plan):
                self.failures["HTTP 429 rate limited"] += 1
                return []

        source = FailingSource()
        monkeypatch.setattr(search, "active_sources", lambda: {"arxiv": source})
        plans = [
            QueryPlan(intent_id="i", subquestion_id="RQ1", source="arxiv", query=q)
            for q in ("a", "b")
        ]

        result = await search.search_all({"query_plans": plans, "intents": {}})
        log = "\n".join(result["log"])
        assert "arxiv:0" in log
        assert "arxiv: 2 request(s) failed (HTTP 429 rate limited x2)" in log


class TestArxivPhraseRelaxation:
    """ANDed quoted phrases relax to their words, not to fewer phrases.

    On one run, 13 of 14 generated arXiv queries were three or more quoted
    phrases. Three phrases returned nothing on 13 of them and two phrases on
    11; the first four words, unquoted, found results for 6 of 7 tested.
    """

    QUERY = '"LLM-based systems" AND "query decomposition" AND "literature research"'

    def plan_for(self, query: str) -> QueryPlan:
        return QueryPlan(intent_id="i", subquestion_id="RQ1", source="arxiv", query=query)

    def test_words_keep_phrase_order_without_generic_vocabulary(self):
        words = ArxivSource()._word_clauses(self.plan_for(self.QUERY))
        # "systems" and "research" are generic; "AND" is a stopword.
        assert words == ["all:LLM-based", "all:query", "all:decomposition", "all:literature"]

    def test_date_ranges_are_not_text_clauses(self):
        plan = self.plan_for('"LLM agents" AND "literature reviews" AND "2023..2026"')
        assert ArxivSource()._clauses(plan) == ['all:"LLM agents"', 'all:"literature reviews"']
        assert "all:2023" not in ArxivSource()._word_clauses(plan)

    async def test_a_phrase_query_relaxes_to_its_words(self, monkeypatch):
        source = ArxivSource()
        queries: list[str] = []

        async def fake_get_text(url, params=None):
            queries.append(params["search_query"])
            return EMPTY_ATOM if '"' in params["search_query"] else ARXIV_ATOM

        monkeypatch.setattr(source, "get_text", fake_get_text)
        hits = await source.search(self.plan_for(self.QUERY))
        assert len(hits) == 2
        assert queries == [
            'all:"LLM-based systems" AND all:"query decomposition" AND all:"literature research"',
            "all:LLM-based AND all:query AND all:decomposition AND all:literature",
        ]

    async def test_relaxation_still_stops_after_three_requests(self, monkeypatch):
        source = ArxivSource()
        calls: list[str] = []

        async def fake_get_text(url, params=None):
            calls.append(params["search_query"])
            return EMPTY_ATOM

        monkeypatch.setattr(source, "get_text", fake_get_text)
        assert await source.search(self.plan_for(self.QUERY)) == []
        assert len(calls) == 3

    async def test_a_single_phrase_does_not_relax_to_one_word(self, monkeypatch):
        source = ArxivSource()
        calls: list[str] = []

        async def fake_get_text(url, params=None):
            calls.append(params["search_query"])
            return EMPTY_ATOM

        monkeypatch.setattr(source, "get_text", fake_get_text)
        # "research" is generic, so the words would be just "deep".
        await source.search(self.plan_for('"deep research"'))
        assert calls == ['all:"deep research"']
