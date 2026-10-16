"""Configuration for the literature research agent.

Every tunable is a field here, so the graph nodes stay free of magic numbers.
Values resolve from four layers, lowest precedence first:

1. the defaults in this module;
2. ``optimiztresearch.yaml`` -- the readable place for nested config such as model
   routing and budgets, and the one to commit;
3. ``.env`` -- for API keys and machine-local overrides, which should not be
   committed;
4. environment variables, including those the CLI sets from its flags, which
   is what makes ``--max-documents 20`` win over a committed file.

Both files are discovered by walking up from the working directory and then
falling back to the project root, with the nearest winning. YAML sits below
the environment on purpose: commit the file, override it per run.

Environment variables use the ``OPTIMIZTRESEARCH_`` prefix and a double
underscore for nesting (``OPTIMIZTRESEARCH_BUDGET__MAX_DOCUMENTS_READ=50``), which
is serviceable for one override and miserable for a whole configuration --
hence the YAML.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

CONFIG_FILENAMES = ("optimiztresearch.yaml", "optimiztresearch.yml")
"""Recognized config filenames, in preference order within one directory."""

PROJECT_ROOT = Path(__file__).resolve().parent.parent
"""Directory containing the package -- where this project's .env lives."""


def _env_files() -> tuple[Path, ...]:
    """Candidate ``.env`` files, lowest precedence first.

    pydantic-settings resolves ``env_file`` against the working directory,
    which is fine for the CLI and wrong everywhere else: a notebook kernel
    started outside the project silently loads no configuration at all, so
    the run reaches for the default hosted model instead of the one that was
    configured. Walking up from the working directory and then falling back
    to the project root makes the config findable from anywhere, while a
    nearer file still wins.
    """
    return _discover(".env")


def _config_files() -> tuple[Path, ...]:
    """Candidate ``optimiztresearch.yaml`` files, lowest precedence first.

    An explicit ``OPTIMIZTRESEARCH_CONFIG_FILE`` wins outright, which is how a
    notebook or a CI job pins a configuration without depending on where it
    happens to be running.
    """
    explicit = os.getenv("OPTIMIZTRESEARCH_CONFIG_FILE", "").strip()
    if explicit:
        return (Path(explicit).expanduser(),)
    found: list[Path] = []
    for name in CONFIG_FILENAMES:
        found.extend(_discover(name))
    return tuple(found)


def _discover(filename: str) -> tuple[Path, ...]:
    """Locate ``filename`` from the project root up through the cwd's parents.

    Returns existing paths only, ordered so the nearest to the working
    directory comes last -- pydantic-settings gives later entries precedence.
    """
    candidates: list[Path] = [PROJECT_ROOT / filename]
    try:
        cwd = Path.cwd().resolve()
    except OSError:
        cwd = None
    if cwd is not None:
        for directory in reversed([cwd, *cwd.parents]):
            candidates.append(directory / filename)
    unique = list(dict.fromkeys(candidates))
    # `.env` is passed through even when absent (pydantic-settings tolerates
    # it), but a missing YAML file makes the source raise, so filter here.
    if filename.endswith((".yaml", ".yml")):
        return tuple(p for p in unique if p.is_file())
    return tuple(unique)

SourceName = Literal[
    "arxiv",
    "semantic_scholar",
    "openalex",
    "crossref",
    "pubmed",
    "web",
]

ALL_SOURCES: tuple[SourceName, ...] = (
    "arxiv",
    "semantic_scholar",
    "openalex",
    "crossref",
    "pubmed",
    "web",
)


class ModelRouting(BaseModel):
    """Which model backs each agent role.

    Specs are ``"provider:model"`` strings understood by
    ``langchain.chat_models.init_chat_model``, so swapping providers is a
    config change rather than a code change. Roles are separated because the
    cost profiles differ by orders of magnitude: screening and reading run
    once per document, while planning and criticism run once per iteration.
    """

    planner: str = "anthropic:claude-opus-5"
    intent: str = "anthropic:claude-sonnet-5"
    query: str = "anthropic:claude-sonnet-5"
    screener: str = "anthropic:claude-haiku-4-5-20251001"
    reader: str = "anthropic:claude-sonnet-5"
    synthesizer: str = "anthropic:claude-opus-5"
    critic: str = "anthropic:claude-opus-5"
    writer: str = "anthropic:claude-opus-5"

    # Roles where we want near-deterministic structure, vs. roles where a
    # little variation improves coverage of the query space.
    temperature_structured: float = 0.0
    temperature_creative: float = 0.4


class Budget(BaseModel):
    """Hard ceilings on the research process.

    Per the methodology these are *limits, not targets*: the critic may
    terminate well before any of them is reached, and "documents retrieved"
    is never itself a stopping criterion.
    """

    max_documents_read: int = 100
    max_search_operations: int = 60
    max_snowball_rounds: int = 3
    max_research_iterations: int = 4

    # Per-iteration shaping
    max_subquestions: int = 8
    max_intents_per_subquestion: int = 4
    max_results_per_query: int = 20
    screen_batch_size: int = 10
    """Documents sent to the screener in one call.

    Ten is comfortable for a hosted model. Lower it for a local one: the call
    has to emit a relevance score per document per subquestion, and a long
    constrained generation is both slow and where a smaller model starts
    losing track of which index it is scoring.
    """
    max_documents_per_iteration: int = 24
    max_documents_per_subquestion: int = 8

    # Saturation: stop snowballing when the yield of newly relevant documents
    # per round falls below this fraction of candidates examined.
    snowball_min_yield: float = 0.10

    max_snowball_seeds: int = 6
    max_snowball_edges_per_seed: int = 40
    """Breadth of one snowball round.

    These should scale with the reading budget, because every candidate a
    round produces has to be screened before it can be ignored. Six seeds at
    40 edges can yield 120 new documents from a single round -- worth it when
    you will read a hundred, disproportionate when you will read eight, and
    expensive either way against a local model that screens slowly.
    """

    # Coverage the critic must certify before the run may terminate happily.
    target_overall_coverage: float = 0.80


class RankingWeights(BaseModel):
    """Weights for the document ranking score.

    Relevance dominates deliberately; the rest are tie-breakers that encode
    the retrieval objectives from the methodology (recency, source quality,
    primary-source and multi-source corroboration).
    """

    relevance: float = 1.0
    citation_impact: float = 0.15
    recency: float = 0.20
    source_quality: float = 0.10
    multi_source_corroboration: float = 0.10
    primary_research: float = 0.10
    novelty: float = 0.15

    # Documents scoring below this after screening are never read.
    min_relevance_to_read: float = 0.45


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OPTIMIZTRESEARCH_",
        env_nested_delimiter="__",
        env_file=_env_files(),
        env_file_encoding="utf-8",
        yaml_file=_config_files(),
        yaml_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Insert the YAML layer below the environment.

        Earlier sources win. ``deep_merge`` matters: without it a nearer YAML
        file replaces a whole block, so setting one budget key in a local file
        would silently reset every other key in that block to its default.
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls, deep_merge=True),
            file_secret_settings,
        )

    models: ModelRouting = Field(default_factory=ModelRouting)
    budget: Budget = Field(default_factory=Budget)
    ranking: RankingWeights = Field(default_factory=RankingWeights)

    # Stored as a comma-separated string rather than list[str]: pydantic-settings
    # insists on JSON for complex types read from the environment, and
    # `a,b,c` is a far nicer thing to type into a .env file than `["a","b"]`.
    enabled_sources: str = ",".join(ALL_SOURCES)

    semantic_scholar: Literal["auto", "on", "off"] = "auto"
    """Whether to use Semantic Scholar.

    ``auto`` (the default) enables it only when ``SEMANTIC_SCHOLAR_API_KEY``
    is set. That is not gatekeeping for its own sake: the unauthenticated
    pool is shared per-IP and returns 429 under even light use, so an
    unkeyed Semantic Scholar spends query-generation calls and search budget
    to return nothing. OpenAlex covers the same ground, including both
    citation directions, with no key.

    ``on`` forces it on regardless -- useful if you want to try the
    unauthenticated pool anyway, or you supply a key by another route.
    ``off`` disables it even when a key is present.
    """

    read_full_text: bool = False
    """When false, evidence extraction works from title/abstract/metadata only.

    The full-text path is implemented (see ``optimiztresearch.fulltext``) but gated,
    because PDF fetching and parsing dominates both latency and token spend
    while the graph logic is still being iterated on.
    """

    ollama_num_ctx: int = 32768
    """Context window for Ollama models, in tokens.

    Ollama defaults to 4096 regardless of what the model supports, and
    silently truncates beyond it. That is far too small here: a screening
    prompt carries ten abstracts, and the critic sees every subquestion
    synthesis. Truncation would not error -- it would quietly degrade the
    research -- so this is set explicitly and passed on every Ollama call.

    Raise it for bigger batches; lower it if memory is tight, since the KV
    cache grows with it.
    """

    output_dir: str = "research"
    contact_email: str = ""
    http_timeout_seconds: float = 30.0
    max_concurrent_requests: int = 8
    max_concurrent_readers: int = 6

    @field_validator("enabled_sources", mode="before")
    @classmethod
    def _accept_a_list(cls, value: Any) -> Any:
        """Allow ``enabled_sources`` to be a YAML list as well as a string.

        The field stays a comma-separated string because that is what an
        environment variable can carry, but writing a list in YAML is the
        obvious thing to do, so both are accepted.
        """
        if isinstance(value, (list, tuple)):
            return ",".join(str(v).strip() for v in value if str(v).strip())
        return value

    @property
    def config_files(self) -> tuple[Path, ...]:
        """YAML files that contributed, nearest last. For diagnostics."""
        return _config_files()

    @property
    def env_files(self) -> tuple[Path, ...]:
        """``.env`` files that exist and contributed, nearest last."""
        return tuple(p for p in _env_files() if p.is_file())

    @property
    def sources(self) -> list[str]:
        """Enabled sources, in the order declared, filtered to known names."""
        requested = [s.strip() for s in self.enabled_sources.split(",") if s.strip()]
        unknown = [s for s in requested if s not in ALL_SOURCES]
        if unknown:
            raise ValueError(
                f"Unknown source(s) {unknown}. Valid sources: {', '.join(ALL_SOURCES)}"
            )
        return requested

    @property
    def semantic_scholar_has_key(self) -> bool:
        return bool(os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip())

    @property
    def use_semantic_scholar(self) -> bool:
        """Resolve the three-way toggle against whether a key is present."""
        if self.semantic_scholar == "on":
            return True
        if self.semantic_scholar == "off":
            return False
        return self.semantic_scholar_has_key

    def source_unavailable_reason(self, name: str) -> str | None:
        """Why ``name`` is not usable, or None if it is.

        The single policy point for source availability: the registry, the
        graph nodes and the `optimiztresearch sources` display all defer to this
        so they cannot disagree.
        """
        if name == "web" and not os.getenv("TAVILY_API_KEY"):
            return "TAVILY_API_KEY is not set"
        if name == "semantic_scholar" and not self.use_semantic_scholar:
            if self.semantic_scholar == "off":
                return "disabled (semantic_scholar=off)"
            return (
                "no SEMANTIC_SCHOLAR_API_KEY; the unauthenticated pool rate-limits "
                "to near-zero yield. Force it on with semantic_scholar=on"
            )
        return None

    @property
    def user_agent(self) -> str:
        base = "optimiztresearch/0.1 (https://github.com/; deep literature research agent)"
        if self.contact_email:
            return f"optimiztresearch/0.1 (mailto:{self.contact_email})"
        return base

    def available_sources(self) -> list[str]:
        """Enabled sources that are actually usable right now."""
        return [
            name for name in self.sources if self.source_unavailable_reason(name) is None
        ]


_settings: Settings | None = None


def get_settings(refresh: bool = False) -> Settings:
    """Process-wide settings singleton."""
    global _settings
    if _settings is None or refresh:
        _settings = Settings()
    return _settings
