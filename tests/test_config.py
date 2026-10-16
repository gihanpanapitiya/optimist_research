"""Tests for the configuration layers.

Precedence, lowest first: defaults < optimiztresearch.yaml < .env < environment.
YAML sits below the environment deliberately, so a committed file can always
be overridden for a single run by a CLI flag (which sets an env var).
"""

from __future__ import annotations

import pytest

from optimiztresearch.config import CONFIG_FILENAMES, PROJECT_ROOT, Settings, _discover


@pytest.fixture
def yaml_settings(monkeypatch):
    """Build Settings against a chosen set of YAML files."""

    def build(*paths, **env):
        monkeypatch.setitem(Settings.model_config, "env_file", None)
        monkeypatch.setitem(
            Settings.model_config, "yaml_file", tuple(str(p) for p in paths)
        )
        for key in list(__import__("os").environ):
            if key.startswith("OPTIMIZTRESEARCH_"):
                monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return Settings()

    return build


def write(path, text):
    path.write_text(text)
    return path


class TestYamlLayer:
    def test_nested_blocks_load(self, tmp_path, yaml_settings):
        config = write(
            tmp_path / "optimiztresearch.yaml",
            "models:\n"
            "  planner: openai:gpt-5.4\n"
            "  screener: openai:gpt-5.4-nano\n"
            "budget:\n"
            "  max_documents_read: 42\n"
            "ranking:\n"
            "  min_relevance_to_read: 0.6\n",
        )
        settings = yaml_settings(config)
        assert settings.models.planner == "openai:gpt-5.4"
        assert settings.models.screener == "openai:gpt-5.4-nano"
        assert settings.budget.max_documents_read == 42
        assert settings.ranking.min_relevance_to_read == 0.6

    def test_unset_keys_keep_their_defaults(self, tmp_path, yaml_settings):
        config = write(
            tmp_path / "optimiztresearch.yaml", "budget:\n  max_documents_read: 42\n"
        )
        settings = yaml_settings(config)
        # Setting one key in a block must not reset the rest of the block.
        assert settings.budget.max_documents_read == 42
        assert settings.budget.max_subquestions == 8
        assert settings.models.planner.startswith("anthropic:")

    def test_enabled_sources_accepts_a_yaml_list(self, tmp_path, yaml_settings):
        config = write(
            tmp_path / "optimiztresearch.yaml",
            "enabled_sources:\n  - arxiv\n  - openalex\n  - crossref\n",
        )
        settings = yaml_settings(config)
        assert settings.enabled_sources == "arxiv,openalex,crossref"
        assert settings.sources == ["arxiv", "openalex", "crossref"]

    def test_enabled_sources_still_accepts_a_string(self, tmp_path, yaml_settings):
        config = write(
            tmp_path / "optimiztresearch.yaml", "enabled_sources: arxiv, openalex\n"
        )
        assert yaml_settings(config).sources == ["arxiv", "openalex"]

    def test_an_empty_file_is_harmless(self, tmp_path, yaml_settings):
        config = write(tmp_path / "optimiztresearch.yaml", "")
        assert yaml_settings(config).budget.max_documents_read == 100


class TestPrecedence:
    def test_the_environment_beats_the_yaml(self, tmp_path, yaml_settings):
        """A CLI flag sets an env var, so this is what --max-documents does."""
        config = write(
            tmp_path / "optimiztresearch.yaml", "budget:\n  max_documents_read: 42\n"
        )
        settings = yaml_settings(
            config, OPTIMIZTRESEARCH_BUDGET__MAX_DOCUMENTS_READ="7"
        )
        assert settings.budget.max_documents_read == 7

    def test_a_nearer_file_wins_key_by_key(self, tmp_path, yaml_settings):
        """Deep merge: overriding one key must not reset its neighbours."""
        far = write(
            tmp_path / "far.yaml",
            "budget:\n  max_documents_read: 11\n  max_subquestions: 3\n",
        )
        near = write(tmp_path / "near.yaml", "budget:\n  max_documents_read: 22\n")
        settings = yaml_settings(far, near)
        assert settings.budget.max_documents_read == 22
        assert settings.budget.max_subquestions == 3

    def test_models_merge_per_role(self, tmp_path, yaml_settings):
        far = write(
            tmp_path / "far.yaml",
            "models:\n  planner: openai:a\n  screener: openai:b\n",
        )
        near = write(tmp_path / "near.yaml", "models:\n  planner: openai:c\n")
        settings = yaml_settings(far, near)
        assert settings.models.planner == "openai:c"
        assert settings.models.screener == "openai:b"


class TestDiscovery:
    def test_the_project_config_is_found_from_anywhere(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        found = _discover("optimiztresearch.yaml")
        if (PROJECT_ROOT / "optimiztresearch.yaml").is_file():
            assert PROJECT_ROOT / "optimiztresearch.yaml" in found

    def test_a_nearer_config_is_ordered_last(self, monkeypatch, tmp_path):
        """Later entries take precedence in pydantic-settings."""
        local = tmp_path / "optimiztresearch.yaml"
        local.write_text("budget:\n  max_documents_read: 5\n")
        monkeypatch.chdir(tmp_path)
        found = _discover("optimiztresearch.yaml")
        assert found[-1] == local

    def test_missing_yaml_files_are_filtered_out(self, monkeypatch, tmp_path):
        """A nonexistent path makes the YAML source raise, so never pass one."""
        monkeypatch.chdir(tmp_path)
        assert all(p.is_file() for p in _discover("optimiztresearch.yaml"))

    def test_an_explicit_config_file_wins_outright(self, monkeypatch, tmp_path):
        from optimiztresearch.config import _config_files

        pinned = tmp_path / "pinned.yaml"
        pinned.write_text("budget:\n  max_documents_read: 3\n")
        monkeypatch.setenv("OPTIMIZTRESEARCH_CONFIG_FILE", str(pinned))
        assert _config_files() == (pinned,)

    def test_both_yaml_extensions_are_recognized(self):
        assert CONFIG_FILENAMES == ("optimiztresearch.yaml", "optimiztresearch.yml")


class TestSubquestionIdResolution:
    """Models do not reliably return a bare subquestion id.

    `gpt-5.4-nano` returned the whole prompt line -- "RQ1: How do AI agents
    generate and revise plans?" -- as the id. Dropping that would score every
    document zero and read nothing, so ids are repaired before matching.
    """

    @staticmethod
    def resolve(raw, valid=("RQ1", "RQ2", "RQ2.1", "RQ10")):
        from optimiztresearch.nodes.common import resolve_subquestion_id

        return resolve_subquestion_id(raw, set(valid))

    def test_a_bare_id_passes_through(self):
        assert self.resolve("RQ1") == "RQ1"

    def test_an_echoed_prompt_line_is_repaired(self):
        assert self.resolve("RQ1: How do AI agents generate and revise plans?") == "RQ1"

    def test_case_is_ignored(self):
        assert self.resolve("rq1") == "RQ1"

    def test_the_longest_matching_id_wins(self):
        """"RQ2.1: ..." must not collapse to RQ2."""
        assert self.resolve("RQ2.1: How are failures detected?") == "RQ2.1"

    def test_a_digit_boundary_is_respected(self):
        assert self.resolve("RQ10: tenth") == "RQ10"
        assert self.resolve("RQ10", valid=("RQ1",)) is None

    def test_separators_are_tolerated(self):
        assert self.resolve("RQ2 - recovery") == "RQ2"
        assert self.resolve("RQ1.") == "RQ1"

    def test_an_unknown_id_is_rejected(self):
        assert self.resolve("RQ7") is None
        assert self.resolve("How do agents plan?") is None
        assert self.resolve("") is None

    def test_a_leading_bullet_is_not_guessed_at(self):
        """Repair must not become invention; this is genuinely ambiguous."""
        assert self.resolve("- RQ1: something") is None
