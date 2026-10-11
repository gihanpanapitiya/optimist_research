"""Tests for the evidence-store views the critic reasons over."""

from __future__ import annotations

import datetime

from litresearch.state import Document, Evidence
from litresearch.store.evidence import (
    cluster_claims,
    diversity_report,
    group_by_subquestion,
    weakly_supported_claims,
)


def ev(claim, doc_id, sq="RQ1", strength=0.8, etype="experimental") -> Evidence:
    return Evidence(
        claim=claim, support="support", doc_id=doc_id, subquestion_id=sq,
        strength=strength, evidence_type=etype,
    )


def doc(doc_id, **kwargs) -> Document:
    defaults = dict(title=f"Paper {doc_id}", authors=["Jane Doe"], year=2024,
                    venue="Venue A", sources=["arxiv"])
    defaults.update(kwargs)
    return Document(doc_id=doc_id, **defaults)


class TestClustering:
    def test_same_claim_from_two_documents_forms_one_cluster(self):
        clusters = cluster_claims([ev("X improves Y", "doc_0001"), ev("x improves y", "doc_0002")])
        assert len(clusters) == 1
        assert clusters[0].support_count == 2

    def test_clusters_are_scoped_by_subquestion(self):
        clusters = cluster_claims([ev("X improves Y", "doc_0001", sq="RQ1"),
                                   ev("X improves Y", "doc_0002", sq="RQ2")])
        assert len(clusters) == 2

    def test_best_supported_cluster_is_first(self):
        clusters = cluster_claims([
            ev("Weak", "doc_0001"),
            ev("Strong", "doc_0002"), ev("Strong", "doc_0003"), ev("Strong", "doc_0004"),
        ])
        assert clusters[0].claim == "Strong"

    def test_filters_to_one_subquestion(self):
        items = [ev("A", "doc_0001", sq="RQ1"), ev("B", "doc_0002", sq="RQ2")]
        assert [c.claim for c in cluster_claims(items, "RQ2")] == ["B"]


class TestSupportStrength:
    def test_single_source_is_weak(self):
        cluster = cluster_claims([ev("Solo claim", "doc_0001")])[0]
        assert cluster.is_weakly_supported

    def test_two_strong_sources_is_not_weak(self):
        cluster = cluster_claims([ev("C", "doc_0001"), ev("C", "doc_0002")])[0]
        assert not cluster.is_weakly_supported

    def test_low_strength_is_weak_despite_corroboration(self):
        """Two papers asserting something without support is still weak."""
        cluster = cluster_claims([
            ev("C", "doc_0001", strength=0.3, etype="position"),
            ev("C", "doc_0002", strength=0.3, etype="position"),
        ])[0]
        assert cluster.is_weakly_supported
        assert not cluster.has_empirical_support

    def test_empirical_support_is_detected(self):
        cluster = cluster_claims([ev("C", "doc_0001", etype="benchmark")])[0]
        assert cluster.has_empirical_support

    def test_weak_claims_are_reported(self):
        items = [ev("Solo", "doc_0001"), ev("Pair", "doc_0002"), ev("Pair", "doc_0003")]
        assert [c.claim for c in weakly_supported_claims(items)] == ["Solo"]


class TestGrouping:
    def test_groups_by_subquestion(self):
        grouped = group_by_subquestion([ev("A", "doc_0001", sq="RQ1"),
                                        ev("B", "doc_0002", sq="RQ2"),
                                        ev("C", "doc_0003", sq="RQ1")])
        assert set(grouped) == {"RQ1", "RQ2"}
        assert len(grouped["RQ1"]) == 2


class TestDiversity:
    def test_empty_evidence_is_safe(self):
        report = diversity_report([], {})
        assert report.document_count == 0
        assert report.summary()

    def test_concentration_is_detected(self):
        """One paper supplying most evidence is a concentrated base."""
        documents = {f"doc_000{i}": doc(f"doc_000{i}") for i in range(1, 4)}
        evidence = [ev(f"Claim {i}", "doc_0001") for i in range(6)]
        evidence += [ev("Other", "doc_0002")]
        report = diversity_report(evidence, documents)
        assert report.top_document_share > 0.35
        assert report.is_concentrated

    def test_spread_base_is_not_concentrated(self):
        documents = {
            f"doc_{i:04d}": doc(f"doc_{i:04d}", authors=[f"Author{i} Surname{i}"],
                                venue=f"Venue {i}")
            for i in range(1, 9)
        }
        evidence = [ev(f"Claim {i}", f"doc_{i:04d}") for i in range(1, 9)]
        report = diversity_report(evidence, documents)
        assert report.distinct_first_authors == 8
        assert not report.is_concentrated

    def test_preprint_and_web_shares(self):
        documents = {
            "doc_0001": doc("doc_0001", is_preprint=True),
            "doc_0002": doc("doc_0002", sources=["web"]),
        }
        report = diversity_report([ev("A", "doc_0001"), ev("B", "doc_0002")], documents)
        assert report.preprint_share == 0.5
        assert report.web_share == 0.5

    def test_recent_share_uses_a_three_year_window(self):
        this_year = datetime.date.today().year
        documents = {
            "doc_0001": doc("doc_0001", year=this_year),
            "doc_0002": doc("doc_0002", year=this_year - 10),
        }
        report = diversity_report([ev("A", "doc_0001"), ev("B", "doc_0002")], documents)
        assert report.recent_share == 0.5

    def test_evidence_citing_unknown_documents_is_tolerated(self):
        report = diversity_report([ev("A", "doc_9999")], {})
        assert report.document_count == 0
