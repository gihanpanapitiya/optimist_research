"""PDF fetching and section-aware text extraction.

Gated behind ``FORESTRESEARCH_READ_FULL_TEXT``. In the default abstract-only
mode the reader works from title, abstract and metadata, which is fast and
cheap enough to iterate on the graph logic; turning this on gives the
evidence extractor real section text so that ``location`` fields point at
"Section 4.2" rather than "Abstract".

Extraction is section-aware on purpose. A long paper does not fit in a
reader's context, and a naive head-truncation keeps the introduction and
throws away the results -- precisely the part evidence lives in. This module
segments the paper by heading and keeps the sections that carry findings.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from .sources.base import get_http_client
from .state import Document

logger = logging.getLogger(__name__)

MAX_PDF_BYTES = 30 * 1024 * 1024

# Headings worth keeping when a paper must be trimmed, in priority order.
# Results and evaluation outrank the introduction: that is where claims are
# actually established rather than previewed.
SECTION_PRIORITY = [
    (r"abstract", 100),
    (r"(result|finding|evaluation|experimental result)", 95),
    (r"(conclusion|discussion)", 90),
    (r"(method|approach|model|architecture|framework|algorithm|design)", 85),
    (r"(experiment|setup|implementation|dataset|benchmark)", 80),
    (r"(limitation|threat|failure|ablation|analysis)", 78),
    (r"(introduction|motivation)", 60),
    (r"(related work|background|preliminaries)", 40),
    (r"(acknowledg|reference|bibliograph|appendix|supplement)", 5),
]

HEADING_RE = re.compile(
    r"^\s*(?:(?P<num>\d+(?:\.\d+)*)\.?\s+)?(?P<title>[A-Z][^\n]{2,80})\s*$"
)


@dataclass
class Section:
    heading: str
    text: str
    priority: int

    @property
    def length(self) -> int:
        return len(self.text)


@dataclass
class ExtractedText:
    text: str
    char_count: int
    sections: list[str]
    truncated: bool


async def fetch_pdf(doc: Document, destination: Path) -> Path | None:
    """Download ``doc``'s PDF to ``destination``, or return None.

    Cached: a PDF already on disk is not re-fetched, which makes re-runs over
    the same corpus cheap.
    """
    if destination.exists() and destination.stat().st_size > 1024:
        return destination
    if not doc.pdf_url:
        return None

    client = get_http_client()
    try:
        async with client.stream("GET", doc.pdf_url) as response:
            if response.status_code != 200:
                logger.info(
                    "pdf fetch for %s returned HTTP %s", doc.doc_id, response.status_code
                )
                return None
            content_type = response.headers.get("content-type", "")
            if "pdf" not in content_type.lower():
                # Publisher landing pages and paywall interstitials return HTML
                # with a 200; writing that to paper.pdf would poison the cache.
                logger.info(
                    "pdf fetch for %s returned %r, not a PDF", doc.doc_id, content_type
                )
                return None

            destination.parent.mkdir(parents=True, exist_ok=True)
            size = 0
            with destination.open("wb") as handle:
                async for chunk in response.aiter_bytes(65536):
                    size += len(chunk)
                    if size > MAX_PDF_BYTES:
                        logger.warning("pdf for %s exceeds size cap; aborting", doc.doc_id)
                        handle.close()
                        destination.unlink(missing_ok=True)
                        return None
                    handle.write(chunk)
    except Exception as exc:  # noqa: BLE001
        logger.warning("pdf fetch failed for %s: %s", doc.doc_id, exc)
        destination.unlink(missing_ok=True)
        return None

    return destination if destination.stat().st_size > 1024 else None


def extract_text(pdf_path: Path, max_chars: int = 60_000) -> ExtractedText | None:
    """Extract paper text, keeping the most evidence-bearing sections."""
    try:
        import pymupdf  # type: ignore
    except ImportError:
        logger.warning(
            "full-text reading is enabled but pymupdf is not installed; "
            "install with `pip install 'forestresearch[pdf]'`"
        )
        return None

    try:
        with pymupdf.open(pdf_path) as document:
            pages = [page.get_text("text") for page in document]
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not parse %s: %s", pdf_path, exc)
        return None

    raw = "\n".join(pages)
    raw = _clean(raw)
    if len(raw) < 500:
        # Scanned or image-only PDF; nothing useful without OCR.
        logger.info("%s yielded almost no text (%d chars)", pdf_path.name, len(raw))
        return None

    sections = _segment(raw)
    kept, truncated = _select_sections(sections, max_chars)

    text = "\n\n".join(f"## {section.heading}\n{section.text}" for section in kept)
    return ExtractedText(
        text=text,
        char_count=len(text),
        sections=[section.heading for section in kept],
        truncated=truncated,
    )


def _clean(text: str) -> str:
    # Join words broken across line endings by hyphenation.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Drop lone page numbers on their own line.
    text = re.sub(r"\n\s*\d{1,3}\s*\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _segment(text: str) -> list[Section]:
    """Split a paper into sections by detected headings."""
    lines = text.split("\n")
    sections: list[Section] = []
    current_heading = "Front matter"
    buffer: list[str] = []

    for line in lines:
        stripped = line.strip()
        if _looks_like_heading(stripped):
            if buffer:
                sections.append(_make_section(current_heading, "\n".join(buffer)))
                buffer = []
            current_heading = stripped
        else:
            buffer.append(line)

    if buffer:
        sections.append(_make_section(current_heading, "\n".join(buffer)))

    # No headings detected (common with two-column or unusual layouts): treat
    # the whole paper as one body section rather than returning nothing.
    if len(sections) <= 1:
        return [Section(heading="Body", text=text, priority=70)]
    return sections


def _looks_like_heading(line: str) -> bool:
    if not 3 <= len(line) <= 80:
        return False
    if line.endswith((".", ",", ";", ":")) and not re.match(r"^\d", line):
        return False
    match = HEADING_RE.match(line)
    if not match:
        return False
    title = match.group("title")
    words = title.split()
    if len(words) > 10:
        return False
    # A heading is title-case or all-caps, and is not a sentence.
    capitalized = sum(1 for w in words if w[:1].isupper())
    return capitalized / max(len(words), 1) >= 0.5


def _make_section(heading: str, text: str) -> Section:
    lowered = heading.lower()
    priority = 70
    for pattern, value in SECTION_PRIORITY:
        if re.search(pattern, lowered):
            priority = value
            break
    return Section(heading=heading.strip(), text=text.strip(), priority=priority)


def _select_sections(sections: list[Section], max_chars: int) -> tuple[list[Section], bool]:
    """Keep the highest-priority sections that fit in the budget.

    Returns them in original document order, so the reader sees a coherent
    paper rather than a priority-sorted jumble.
    """
    total = sum(section.length for section in sections)
    if total <= max_chars:
        return sections, False

    indexed = list(enumerate(sections))
    indexed.sort(key=lambda pair: (-pair[1].priority, pair[0]))

    kept: list[tuple[int, Section]] = []
    budget = max_chars
    for index, section in indexed:
        if section.priority <= 5:
            continue  # references and appendices are never worth the budget
        if section.length <= budget:
            kept.append((index, section))
            budget -= section.length
        elif budget > 2000:
            # Partially include a long high-priority section rather than
            # dropping it wholesale.
            truncated = section.text[:budget].rsplit(" ", 1)[0] + "\n[… section truncated]"
            kept.append((index, Section(section.heading, truncated, section.priority)))
            budget = 0
        if budget <= 0:
            break

    kept.sort(key=lambda pair: pair[0])
    return [section for _, section in kept], True


async def load_full_text(
    doc: Document, pdf_path: Path, max_chars: int = 60_000
) -> ExtractedText | None:
    """Fetch and extract in one step. Returns None if either stage fails."""
    path = await fetch_pdf(doc, pdf_path)
    if path is None:
        return None
    return extract_text(path, max_chars=max_chars)
