"""MCP server tools, driven against the fake model and sources.

These check the background-run lifecycle and the disk-backed lookups, not the
protocol transport itself, which the MCP SDK owns.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mcp")

from optimistresearch import mcp_server  # noqa: E402
from optimistresearch.store.corpus import slugify  # noqa: E402

QUESTION = "How do AI agents plan and recover from failures?"


@pytest.fixture(autouse=True)
def clear_runs():
    mcp_server._runs.clear()
    yield
    mcp_server._runs.clear()


async def _finish(run_id: str) -> None:
    await mcp_server._runs[run_id].task


async def test_tools_are_registered():
    names = {tool.name for tool in await mcp_server.mcp.list_tools()}
    assert names == {
        "start_research",
        "research_status",
        "cancel_research",
        "get_report",
        "get_corpus_file",
        "list_research",
        "list_sources",
    }


async def test_run_lifecycle(wired):
    started = await mcp_server.start_research(QUESTION, max_iterations=1)
    run_id = started["run_id"]
    assert run_id == slugify(QUESTION)
    assert started["status"] == "running"
    assert started["budget"]["max_research_iterations"] == 1

    await _finish(run_id)

    status = await mcp_server.research_status(run_id)
    assert status["status"] == "completed"
    assert status["summary"]["documents_read"] > 0
    assert status["recent_log"]

    assert (await mcp_server.get_report(run_id)).strip()
    assert "RQ1" in await mcp_server.get_corpus_file(run_id, "research_plan.md")

    listed = await mcp_server.list_research()
    assert [r["run_id"] for r in listed] == [run_id]
    assert listed[0]["status"] == "completed"


async def test_overrides_do_not_leak_into_environment(wired, monkeypatch):
    monkeypatch.delenv("OPTIMISTRESEARCH_BUDGET__MAX_DOCUMENTS_READ", raising=False)
    started = await mcp_server.start_research(QUESTION, max_documents=3)
    await _finish(started["run_id"])
    import os

    assert "OPTIMISTRESEARCH_BUDGET__MAX_DOCUMENTS_READ" not in os.environ


async def test_completed_run_is_found_from_disk_after_restart(wired):
    started = await mcp_server.start_research(QUESTION, max_iterations=1)
    await _finish(started["run_id"])
    mcp_server._runs.clear()  # as if the server process had restarted

    status = await mcp_server.research_status(started["run_id"])
    assert status["status"] == "completed"
    assert await mcp_server.get_report(started["run_id"])


async def test_second_run_is_refused_while_one_is_active(wired):
    started = await mcp_server.start_research(QUESTION)
    with pytest.raises(ValueError, match="still in progress"):
        await mcp_server.start_research("A different question")
    await mcp_server.cancel_research(started["run_id"])
    assert (await mcp_server.research_status(started["run_id"]))["status"] == "cancelled"


async def test_unknown_and_escaping_ids_are_rejected(wired):
    with pytest.raises(ValueError, match="no research run"):
        await mcp_server.research_status("never-ran")
    with pytest.raises(ValueError, match="invalid run_id"):
        await mcp_server.get_report("../outside")


async def test_corpus_file_cannot_escape_the_run(wired):
    started = await mcp_server.start_research(QUESTION, max_iterations=1)
    await _finish(started["run_id"])
    with pytest.raises(ValueError, match="inside the run"):
        await mcp_server.get_corpus_file(started["run_id"], "../../etc/passwd")
