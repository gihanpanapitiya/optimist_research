"""Literature source adapters."""

from .base import SearchSource, close_http_client, get_http_client
from .registry import SOURCE_CLASSES, active_sources, get_source, snowball_sources

__all__ = [
    "SOURCE_CLASSES",
    "SearchSource",
    "active_sources",
    "close_http_client",
    "get_http_client",
    "get_source",
    "snowball_sources",
]
