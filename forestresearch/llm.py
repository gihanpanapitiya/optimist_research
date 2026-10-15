"""Provider-agnostic model access.

Every node asks for a model by *role* ("planner", "reader", ...) rather than
by name. The mapping from role to concrete model lives in config, so the whole
system can be repointed at a different provider without touching node code.
"""

from __future__ import annotations

import logging
from functools import cache
from typing import TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from .config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

STRUCTURED_ROLES = frozenset(
    {"planner", "screener", "reader", "critic", "intent"}
)
"""Roles whose output is parsed into a schema -- run these at temperature 0."""


@cache
def get_model(role: str) -> BaseChatModel:
    """Return the chat model configured for ``role``.

    Cached per role so we build one client per role per process.
    """
    settings = get_settings()
    spec = getattr(settings.models, role, None)
    if spec is None:
        raise ValueError(
            f"Unknown model role {role!r}. Known roles: "
            f"{', '.join(sorted(type(settings.models).model_fields))}"
        )

    temperature = (
        settings.models.temperature_structured
        if role in STRUCTURED_ROLES
        else settings.models.temperature_creative
    )

    # init_chat_model lives in `langchain` proper and understands
    # "provider:model" specs, dispatching to whichever langchain-<provider>
    # package is installed.
    from langchain.chat_models import init_chat_model

    extra = _provider_kwargs(spec, settings)

    try:
        return init_chat_model(spec, temperature=temperature, **extra)
    except ImportError as exc:  # provider package not installed
        provider = spec.split(":", 1)[0]
        raise RuntimeError(
            f"Role {role!r} is configured for provider {provider!r} but its "
            f"integration package is missing. Install it, e.g. "
            f"`pip install 'forestresearch[{provider}]'`."
        ) from exc


def _provider_kwargs(spec: str, settings) -> dict:
    """Provider-specific constructor arguments.

    Only Ollama needs any: its default context window is 4096 tokens no
    matter what the model supports, and it truncates silently rather than
    erroring, so the window has to be set explicitly.
    """
    provider = spec.split(":", 1)[0].lower()
    if provider == "ollama":
        return {"num_ctx": settings.ollama_num_ctx}
    return {}


def structured(role: str, schema: type[T]) -> Runnable:
    """A runnable for ``role`` that emits validated instances of ``schema``."""
    return get_model(role).with_structured_output(schema)


async def ainvoke_structured(
    role: str,
    schema: type[T],
    messages: list,
    *,
    fallback: T | None = None,
) -> T | None:
    """Invoke a structured-output model, tolerating a single hard failure.

    Structured-output parsing can fail on a bad generation. Individual node
    failures should degrade the research run, not abort it: a document whose
    evidence extraction fails is a lost document, not a lost report. Nodes
    that genuinely cannot proceed pass ``fallback=None`` and handle the
    ``None`` return explicitly.
    """
    try:
        return await structured(role, schema).ainvoke(messages)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
        logger.warning(
            "structured call failed for role=%s schema=%s: %s",
            role,
            schema.__name__,
            exc,
        )
        return fallback
