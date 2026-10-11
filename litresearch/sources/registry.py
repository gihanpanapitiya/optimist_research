"""Source registry: name -> adapter instance.

Adapters are instantiated lazily and cached, because each one owns a rate
limiter whose state must be shared across every call in a run.
"""

from __future__ import annotations

import logging

from ..config import get_settings
from .arxiv import ArxivSource
from .base import SearchSource
from .crossref import CrossrefSource
from .openalex import OpenAlexSource
from .pubmed import PubMedSource
from .semantic_scholar import SemanticScholarSource
from .web import WebSource

logger = logging.getLogger(__name__)

SOURCE_CLASSES: dict[str, type[SearchSource]] = {
    "arxiv": ArxivSource,
    "semantic_scholar": SemanticScholarSource,
    "openalex": OpenAlexSource,
    "crossref": CrossrefSource,
    "pubmed": PubMedSource,
    "web": WebSource,
}

_instances: dict[str, SearchSource] = {}


def get_source(name: str) -> SearchSource:
    if name not in SOURCE_CLASSES:
        raise ValueError(f"Unknown source {name!r}")
    if name not in _instances:
        _instances[name] = SOURCE_CLASSES[name]()
    return _instances[name]


def active_sources() -> dict[str, SearchSource]:
    """Enabled sources that are actually usable right now.

    A source enabled in config but missing its API key is dropped with a
    warning rather than failing the run.
    """
    settings = get_settings()
    active: dict[str, SearchSource] = {}
    for name in settings.sources:
        reason = settings.source_unavailable_reason(name)
        if reason is not None:
            logger.info("source %r enabled but unavailable: %s", name, reason)
            continue
        active[name] = get_source(name)
    if not active:
        raise RuntimeError(
            "No literature sources are available. Check LITRESEARCH_ENABLED_SOURCES "
            "and any required API keys."
        )
    return active


def snowball_sources(direction: str) -> list[SearchSource]:
    """Active sources able to traverse citations in ``direction``.

    Preference is key-aware. Semantic Scholar has the richer citation graph
    and goes first when authenticated; without a key it is either disabled or
    rate-limited to uselessness, so OpenAlex leads and carries the citation
    graph on its own. Later entries are fallbacks, tried in order until one
    returns edges -- see ``_expand_one`` in the snowball node.
    """
    attribute = "supports_backward" if direction == "backward" else "supports_forward"
    settings = get_settings()
    if settings.semantic_scholar_has_key:
        preference = ["semantic_scholar", "openalex"]
    else:
        preference = ["openalex", "semantic_scholar"]
    # Crossref is backward-only and goes last: publishers deposit reference
    # lists there for many records whose OpenAlex referenced_works is empty,
    # which is the gap it exists to cover.
    if direction == "backward":
        preference.append("crossref")
    active = active_sources()
    return [
        active[name]
        for name in preference
        if name in active and getattr(active[name], attribute, False)
    ]


def reset_sources() -> None:
    """Drop cached adapters (used by tests after changing settings)."""
    _instances.clear()
