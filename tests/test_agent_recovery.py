"""Regression coverage for interrupted Agent jobs; no paid model calls."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from pptx import Presentation

from backend.api.endpoints import download, generate
from backend.api.schemas import AgentFeedbackRequest
from backend.orchestrator import agent_pipeline as agent
from backend.orchestrator import agent_session_registry
from backend.runtime.scheduler import QueueFull, SchedulerDraining


class MemoryJobs:
    def __init__(self, job):
        self.job = job
        self.events = []

    def get_job(self, job_id):
        return self.job if job_id == "job" else None

    def update_job(self, job_id, **updates):
        for key, value in updates.items():
            setattr(self.job, key, value)

    def record_event(self, job_id, payload, **updates):
        self.events.append(payload)
        self.update_job(job_id, **updates)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    project = tmp_path / "deck"
    project.mkdir()
    (project / "agent_task.json").write_text(json.dumps({"presentation": {"target_pages": 3}}))
    job = SimpleNamespace(
        project_dir=str(project), provider="agent:codex", status="error",
        message="Disconnected", progress=0.6, slides_completed=1, total_slides=3, error="Disconnected",
        output_path=None, canvas_format="ppt169",
    )
    manager = MemoryJobs(job)
    monkeypatch.setattr(generate, "session_manager", manager)
    monkeypatch.setattr(download, "session_manager", manager)
    monkeypatch.setattr(download.settings, "workspaces_dir", tmp_path)
    monkeypatch.setattr(agent_session_registry, "_active_sessions", {})
    return project, job, manager


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["error", "cancelled", "complete"])
async def test_continue_terminal_job_uses_same_workspace_and_session(workspace, monkeypatch, state):
    project, job, manager = workspace
    job.status = state
    agent._save_agent_session(project, "saved-thread")
    scheduler = SimpleNamespace(submit=AsyncMock(return_value=0))
    monkeypatch.setattr(generate, "get_scheduler", lambda: scheduler)
    runner = AsyncMock()
    monkeypatch.setattr(generate, "_run_agent_feedback_job", runner)
    result = await generate.send_agent_generation_feedback("job", AgentFeedbackRequest(message="Continue"))
    assert result.status == "queued"
    assert job.status == "pending" and job.error is None
    assert job.slides_completed == 1 and job.progress == 0.6
    await scheduler.submit.call_args.args[1]()
    assert runner.call_args.kwargs["project_dir"] == project
    assert runner.call_args.kwargs["session_id"] == "saved-thread"
    assert runner.call_args.kwargs["runtime"] == "codex"
    assert runner.call_args.kwargs["total_slides_hint"] == 3


@pytest.mark.asyncio
async def test_continue_without_sdk_checkpoint_still_uses_saved_files(workspace, monkeypatch):
    project, job, _ = workspace
    scheduler = SimpleNamespace(submit=AsyncMock(return_value=0))
    monkeypatch.setattr(generate, "get_scheduler", lambda: scheduler)
    runner = AsyncMock()
    monkeypatch.setattr(generate, "_run_agent_feedback_job", runner)
    await generate.send_agent_generation_feedback("job", AgentFeedbackRequest(message="Continue"))
    await scheduler.submit.call_args.args[1]()
    assert runner.call_args.kwargs["session_id"] is None
    prompt = agent._build_feedback_runtime_prompt(project, "Continue")
    assert "Complete only missing or invalid artifacts" in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("error,code", [(QueueFull("busy"), 503), (SchedulerDraining("closing"), 503), (RuntimeError("already running"), 409)])
async def test_rejected_resume_preserves_job_state(workspace, monkeypatch, error, code):
    _, job, _ = workspace
    scheduler = SimpleNamespace(submit=AsyncMock(side_effect=error))
    monkeypatch.setattr(generate, "get_scheduler", lambda: scheduler)
    with pytest.raises(HTTPException) as caught:
        await generate.send_agent_generation_feedback("job", AgentFeedbackRequest(message="Continue"))
    assert caught.value.status_code == code
    assert job.status == "error" and job.error == "Disconnected"


@pytest.mark.asyncio
async def test_workspace_not_prepared_is_rejected(workspace):
    project, _, _ = workspace
    (project / "agent_task.json").unlink()
    with pytest.raises(HTTPException) as caught:
        await generate.send_agent_generation_feedback("job", AgentFeedbackRequest(message="Continue"))
    assert caught.value.status_code == 409


@pytest.mark.asyncio
async def test_provider_job_cannot_use_agent_resume(workspace):
    _, job, _ = workspace
    job.provider = "openai"
    with pytest.raises(HTTPException) as caught:
        await generate.send_agent_generation_feedback("job", AgentFeedbackRequest(message="Continue"))
    assert caught.value.status_code == 400


@pytest.mark.parametrize("runtime", ["claude_code", "codex"])
def test_session_id_is_checkpointed_immediately(workspace, runtime):
    project, _, _ = workspace
    session = agent.PersistentAgentSession(project, runtime, {})
    if runtime == "claude_code":
        init = type("SystemMessage", (), {"subtype": "init", "data": {"session_id": "live-thread"}})()
        session._capture_session_id(init)
    else:
        session.session_id = "live-thread"
    assert agent._load_agent_session_id(project) == "live-thread"
    assert not list(project.glob("*.tmp"))


@pytest.mark.parametrize("value", ['{broken', '[]', 'null'])
def test_bad_checkpoint_falls_back_to_workspace(workspace, value):
    project, _, _ = workspace
    (project / "agent_session.json").write_text(value)
    assert agent._load_agent_session_id(project) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["error", "cancelled"])
async def test_export_saved_slides_uses_sources_and_preserves_recovery(workspace, state):
    project, job, _ = workspace
    job.status = state
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720"><rect width="1280" height="720" fill="#ffffff"/><text x="80" y="100" font-size="32">Saved slide</text></svg>'
    for folder, count in [("svg_output", 2), ("svg_final", 1)]:
        (project / folder).mkdir()
        for i in range(1, count + 1):
            (project / folder / f"slide_{i:03}.svg").write_text(svg)
    result = await download.reexport_presentation("job")
    assert len(Presentation(result.output_path).slides) == 2
    assert job.status == result.status == state
    assert job.error == "Disconnected"
    assert any("incomplete" in warning for warning in result.warnings)
    response = await download.download_presentation("job")
    assert Path(response.path) == Path(result.output_path)


@pytest.mark.asyncio
async def test_no_slides_export_does_not_mark_complete(workspace):
    _, job, _ = workspace
    with pytest.raises(HTTPException) as caught:
        await download.reexport_presentation("job")
    assert caught.value.status_code == 409
    assert job.status == "error" and job.output_path is None


@pytest.mark.asyncio
async def test_export_cannot_race_live_generation(workspace):
    _, job, _ = workspace
    job.status = "generation"
    with pytest.raises(HTTPException) as caught:
        await download.reexport_presentation("job")
    assert caught.value.status_code == 409
    assert job.status == "generation"



@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["error", "cancelled", "generation"])
async def test_old_export_does_not_hide_job_state(workspace, monkeypatch, state):
    from backend.api.endpoints import status
    from backend.runtime import job_event_monitor

    project, job, manager = workspace
    job.status = state
    job.model_name = "test-model"
    job.base_url = None
    old_export = project / "old.pptx"
    old_export.write_bytes(b"previous export")
    job.output_path = str(old_export)
    monkeypatch.setattr(status, "session_manager", manager)
    monkeypatch.setattr(job_event_monitor, "sync_job_events_once", AsyncMock())
    result = await status.get_job_status("job")
    assert result.status == state
    assert result.error == "Disconnected"


def test_server_restart_preserves_workspace_checkpoint(workspace, monkeypatch):
    from backend.session.manager import Job, SessionManager

    project, _, _ = workspace
    agent._save_agent_session(project, "before-crash")
    # Simulate metadata restored from a snapshot after process restart.
    restored = SessionManager.__new__(SessionManager)
    restored_job = Job(id="job", session_id="upload", status="generation", project_dir=str(project), provider="agent:codex")
    restored._jobs = {"job": restored_job}
    monkeypatch.setattr(restored, "_persist_state", lambda: None)
    restored._mark_orphaned_running_jobs()
    assert restored_job.status == "error"
    assert restored_job.project_dir == str(project)
    assert agent._load_agent_session_id(project) == "before-crash"


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh"])
def test_claude_launch_receives_model_and_effort(workspace, effort):
    project, _, _ = workspace
    session = agent.PersistentAgentSession(project, "claude_code", {
        "model": "custom-model", "reasoning_effort": effort,
    })
    options = session._build_claude_options()
    assert options.model == "custom-model"
    assert options.effort == effort


def test_claude_runtime_default_does_not_force_effort(workspace):
    project, _, _ = workspace
    options = agent.PersistentAgentSession(project, "claude_code", {})._build_claude_options()
    assert options.effort is None


def test_resume_restores_model_and_effort_without_credentials(workspace):
    project, _, _ = workspace
    original = {
        "model": "custom-model", "reasoning_effort": "xhigh", "max_turns": 120,
        "load_project_settings": False, "api_key": "secret", "auth_token": "secret",
        "base_url": "https://private.example", "_research_env": {"TOKEN": "secret"},
    }
    saved = agent._persistable_agent_runtime_config(original)
    assert set(saved) == {"model", "reasoning_effort", "max_turns", "load_project_settings"}
    (project / "agent_task.json").write_text(json.dumps({"runtime_config": saved}))
    restored = agent._agent_config_from_existing_task(project, "codex")
    for key, value in saved.items():
        assert restored[key] == value
    assert "api_key" not in restored and "auth_token" not in restored


def test_old_tasks_without_runtime_config_keep_defaults(workspace):
    project, _, _ = workspace
    restored = agent._agent_config_from_existing_task(project, "codex")
    assert "model" not in restored and "reasoning_effort" not in restored


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime", ["codex", "claude_code"])
async def test_generate_forwards_model_and_effort_independently(workspace, monkeypatch, runtime):
    from backend.api.schemas import GenerateRequest

    project, job, manager = workspace
    job.id = "job"
    manager.create_job = lambda session_id: job
    manager.get_session = lambda session_id: SimpleNamespace(
        id="upload", file_path=project / "paper.pdf", source_type="pdf", sources=[],
    )
    scheduler = SimpleNamespace(submit=AsyncMock(return_value=0))
    monkeypatch.setattr(generate, "get_scheduler", lambda: scheduler)
    monkeypatch.setattr(agent, "agent_runtime_status", AsyncMock(return_value={"available": True}))
    monkeypatch.setattr(agent, "agent_runtime_defaults", lambda runtime: {})
    runner = AsyncMock()
    monkeypatch.setattr(generate, "_run_generation_job", runner)
    request = GenerateRequest.model_validate({
        "session_id": "upload",
        "options": {
            "generation_backend": "agent", "detail_level": "very_high",
            "enable_deep_research": True,
            "agent_config": {"runtime": runtime, "model": "custom-model", "reasoning_effort": "low"},
        },
    })
    result = await generate.generate_presentation(request)
    assert result.job_id == "job"
    await scheduler.submit.call_args.args[1]()
    forwarded = runner.call_args.args[1].agent_config
    assert forwarded["model"] == "custom-model"
    assert forwarded["reasoning_effort"] == "low"
