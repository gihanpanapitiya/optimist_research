"""`optimistresearch init`, which writes the starter config after a pip install."""

from __future__ import annotations

from typer.testing import CliRunner

from optimistresearch.cli import app

runner = CliRunner()


def test_init_writes_both_files(tmp_path):
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "models:" in (tmp_path / "optimistresearch.yaml").read_text()
    assert "OPENAI_API_KEY=" in (tmp_path / ".env").read_text()


def test_init_keeps_existing_files_unless_forced(tmp_path):
    (tmp_path / ".env").write_text("OPENAI_API_KEY=mine\n")

    runner.invoke(app, ["init", str(tmp_path)])
    assert (tmp_path / ".env").read_text() == "OPENAI_API_KEY=mine\n"

    runner.invoke(app, ["init", str(tmp_path), "--force"])
    assert (tmp_path / ".env").read_text() != "OPENAI_API_KEY=mine\n"
