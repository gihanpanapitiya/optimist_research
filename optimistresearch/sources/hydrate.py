"""Enrich bare DOIs into full document records.

Some discovery mechanisms yield identifiers rather than metadata. Crossref's
reference lists are the main case: they name the works a paper cites, often
with only a DOI and a surname, which is not enough for the screening and
reading stages to do anything useful with.

Hydration closes that gap by looking the DOIs up in OpenAlex, which returns
titles, authors, venues, citation counts and -- the reason it is worth a
request at all -- abstracts. A candidate with no abstract is heavily
discounted by the ranker and in practice never read, so an unhydrated stub
would be discovery without benefit.

Discovery credit stays with whichever adapter found the DOI; OpenAlex is only
supplying the metadata.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

from ..state import Provenance, RawHit, normalize_doi

logger = logging.getLogger(__name__)


async def hydrate_by_doi(
    dois: Iterable[str],
    make_provenance: Callable[[str], Provenance],
    *,
    source_name: str,
) -> dict[str, RawHit]:
    """Look up ``dois`` in OpenAlex and return normalized hits.

    Keyed by normalized DOI. Returns ``{}`` when OpenAlex is unavailable --
    hydration is an enhancement, so its absence degrades result quality
    rather than failing the caller.
    """
    wanted = [d for d in (normalize_doi(d) for d in dois) if d]
    if not wanted:
        return {}

    # Imported at call time: the registry imports every adapter, and the
    # adapters that hydrate import this module, so a module-level import
    # would close a cycle.
    from .openalex import work_to_hit
    from .registry import active_sources

    try:
        openalex = active_sources().get("openalex")
    except RuntimeError:
        openalex = None
    if openalex is None:
        logger.info(
            "hydration skipped: openalex is not enabled, so %d DOI(s) from %s "
            "stay as sparse records",
            len(wanted),
            source_name,
        )
        return {}

    records = await openalex.fetch_by_dois(wanted)  # type: ignore[attr-defined]

    hits: dict[str, RawHit] = {}
    for doi, work in records.items():
        hit = work_to_hit(work, make_provenance(doi), source_name)
        if hit is not None:
            hits[doi] = hit

    missing = len(wanted) - len(hits)
    if missing:
        logger.debug("hydration: %d of %d DOIs unknown to OpenAlex", missing, len(wanted))
    return hits
