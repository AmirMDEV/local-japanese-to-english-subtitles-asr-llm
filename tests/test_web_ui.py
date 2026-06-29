from pathlib import Path
from types import SimpleNamespace

import pytest

from local_subtitle_stack.domain import JOB_STATUS_WORKING
from local_subtitle_stack.web_ui import (
    HTML,
    WebServiceState,
    close_other_web_ui_processes,
    count_completed_outputs,
    count_sources,
    format_bytes,
    ollama_blob_paths,
    ollama_manifest_path,
)


def test_web_ui_counts_sources_and_completed_outputs(tmp_path: Path) -> None:
    folder = tmp_path / "course"
    nested = folder / "nested"
    nested.mkdir(parents=True)
    video = folder / "one.mp4"
    nested_video = nested / "two.mov"
    video.write_text("video", encoding="utf-8")
    nested_video.write_text("video", encoding="utf-8")
    (folder / "one.raw.meta.json").write_text("{}", encoding="utf-8")

    assert count_sources([folder], recursive=False) == 1
    assert count_sources([folder], recursive=True) == 2
    assert count_sources([video], recursive=True) == 1
    assert count_completed_outputs([folder], recursive=True) == 1


def test_web_ui_has_responsive_layout_shell() -> None:
    assert "Fast Multilanguage Transcriber" in HTML
    assert "className:\"layout\"" in HTML
    assert "@media (max-width: 980px)" in HTML
    assert "target-list" in HTML
    assert "max-height: min(34vh, 340px)" in HTML
    assert "/api/models" in HTML
    assert "Ollama storage" in HTML
    assert "Manifest file" in HTML
    assert "Stored on" in HTML
    assert "Blob files" in HTML
    assert "Preview and line editor" in HTML
    assert "review-stack" in HTML
    assert "context-panel" in HTML
    assert "preview-editor-panel" in HTML
    assert "fmt.collapsedPanels.v1" in HTML
    assert "panel-toggle" in HTML
    assert "aria-expanded" in HTML
    assert "data-panel-key" in HTML
    assert "flex-wrap: wrap" in HTML
    assert "flex: 1 1 220px" in HTML
    assert "overflow-wrap: anywhere" in HTML
    assert "Time display" in HTML
    assert "Seconds, e.g. 62.5s" in HTML
    assert "Hours:minutes:seconds, e.g. 00:01:02.500" in HTML
    assert "preview-header" in HTML
    assert "preview-time" in HTML
    assert "preview-text" in HTML
    assert "user-select: none" in HTML
    assert "Japanese subtitles" in HTML
    assert "Direct English translation" in HTML
    assert "Context-applied English" in HTML
    assert "No context-applied English loaded" in HTML
    assert "No direct English loaded" in HTML
    assert "Japanese source subtitles" in HTML
    assert "What the listening model heard" in HTML
    assert "Read-only model output" in HTML
    assert "editLineText" in HTML
    assert "lineDraftRef" in HTML
    assert "Saved ${label}." in HTML
    assert "auto-save to the context-applied English subtitle file" in HTML
    assert "Reference subtitles" in HTML
    assert "Optional outside subtitle track" in HTML
    assert "setInterval(() =>" in HTML
    assert "/api/job?id=" in HTML
    assert "Transcription and translation models" in HTML
    assert "model-settings-panel" in HTML
    assert ".review-stack > .model-settings-panel { order: -1; }" in HTML
    assert "grid-template-columns: minmax(420px, 1.08fr) minmax(360px, .92fr)" in HTML
    assert "text-overflow: ellipsis" in HTML
    assert "Saving" in HTML
    assert "Auto-saving" in HTML
    assert "api(\"/api/settings/save\"" in HTML
    assert "Save settings" not in HTML
    assert "Use Gemma e2b" not in HTML
    assert "Download Gemma" not in HTML
    assert "dirty-pill" in HTML
    assert "select option" in HTML
    assert "Downloaded Hugging Face Japanese ASR/listening model files are stored and reused here" in HTML
    assert "does not copy Gemma/Ollama English models" in HTML
    assert "Load subtitle files into preview" in HTML
    assert "Create preview job" in HTML
    assert "appears in Preview and line editor" in HTML
    assert "attachSubtitle" in HTML
    assert "Attach direct English translation to preview" in HTML
    assert "Check setup" in HTML
    assert "Health and redo log" in HTML
    assert "diagnostics-grid" in HTML
    assert "diagnostic-block" in HTML
    assert 'panel("health"' not in HTML
    assert 'panel("redo-log"' not in HTML
    assert "Start processing all jobs" in HTML
    assert "Clear finished jobs" in HTML
    assert "Clear all non-running jobs" in HTML
    assert "Resume this job" in HTML
    assert "Resume stuck job" in HTML
    assert "/api/job/resume" in HTML
    assert "canResumeJob" in HTML
    assert "Force stop now" in HTML
    assert "/api/job/force-stop" in HTML
    assert "canForceStopJob" in HTML
    assert "Stop this job after current step" in HTML
    assert "/api/job/stop" in HTML
    assert "stop_requested" in HTML
    assert "job-delete" in HTML
    assert "Remove job from list" in HTML
    assert "What to do now" in HTML
    assert "Open direct English translation in Subtitle Edit" in HTML
    assert "Context-applied English" in HTML
    assert "Open review bundle in Subtitle Edit" in HTML
    assert "Add files and start processing" in HTML
    assert "added to every English translation prompt after the Japanese audio has been transcribed" in HTML
    assert "Overall video context" in HTML
    assert "Time-range context" in HTML
    assert "Selected subtitle lines fill these times automatically" in HTML
    assert "Add time-range context" in HTML
    assert "Retranslate selected time range" in HTML
    assert "Run second-pass coherence review" in HTML
    assert "Second-pass coherence review progress" in HTML
    assert "Running step progress" in HTML
    assert "stage_progress_message" in HTML
    assert "progress_age_seconds" in HTML
    assert "Likely active" in HTML
    assert "Possible stuck" in HTML
    assert "No progress timestamp" in HTML
    assert "worker_resources" in HTML
    assert "workerResourceText" in HTML
    assert "CPU ${resources.cpu_percent}%" in HTML
    assert "GPU ${resources.gpu_util_percent}%" in HTML
    assert "VRAM ${resources.gpu_memory_used_mb}/${resources.gpu_memory_total_mb} MB" in HTML
    assert "progress_stage" in HTML
    assert "Second-pass changes" in HTML
    assert "Restore before" in HTML
    assert "selectCoherenceChange" in HTML
    assert "setSelectedCueIndexes([row ? row.cue_index : targetIndex])" in HTML
    assert "has_adapted_english: true" in HTML
    assert "Restoring..." in HTML
    assert "Restored" in HTML
    assert "Restore failed" in HTML
    assert "change-action" in HTML
    assert "line-edit-grid" in HTML
    assert "/api/job/coherence-pass" in HTML
    assert "Shift-click selects a continuous range" in HTML
    assert "Pick context-applied" in HTML
    assert "Pick natural" not in HTML
    assert "Natural English SRT" not in HTML
    assert "Series or project name" in HTML
    assert "Drop an .srt file here to edit existing subtitles" in HTML
    assert "/api/upload-subtitle" in HTML
    assert "Saved subtitle files" in HTML
    assert "Delete job from list" in HTML
    assert "/api/job/delete" in HTML
    assert "selectNewJob" in HTML
    assert "setSettingsDraft(current => current || data.settings)" in HTML


def test_resume_stuck_job_stops_owned_worker_before_requeue(monkeypatch) -> None:
    class FakeStore:
        def find_job(self, job_id: str):
            return Path("working") / job_id, SimpleNamespace(status=JOB_STATUS_WORKING)

        def active_worker_pid(self) -> int:
            return 123

    class FakeService:
        store = FakeStore()

        def __init__(self) -> None:
            self.resumed: list[str] = []

        def resume(self, job_id: str):
            self.resumed.append(job_id)
            return SimpleNamespace(job_id=job_id)

    class FakeProcess:
        pid = 123

        def __init__(self) -> None:
            self.terminated = False

        def cmdline(self) -> list[str]:
            return ["python", "-m", "local_subtitle_stack.cli", "worker"]

        def children(self, recursive: bool = False) -> list:
            return []

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            raise AssertionError("owned worker should terminate cleanly")

    process = FakeProcess()
    state = WebServiceState.__new__(WebServiceState)
    state.service = FakeService()
    state.worker_process = SimpleNamespace(pid=123)
    state.start_worker = lambda: {"message": "Worker started", "pid": 456}
    monkeypatch.setattr("local_subtitle_stack.web_ui.psutil.Process", lambda _pid: process)
    monkeypatch.setattr("local_subtitle_stack.web_ui.psutil.wait_procs", lambda processes, timeout: (processes, []))

    assert state.resume_job("job-1") == {"job_id": "job-1"}
    assert process.terminated is True
    assert state.service.resumed == ["job-1"]
    assert state.worker_process is None


def test_force_stop_working_job_stops_worker_tree_and_marks_paused(monkeypatch) -> None:
    class FakeStore:
        def __init__(self) -> None:
            self.manifest = SimpleNamespace(job_id="job-1", status=JOB_STATUS_WORKING)
            self.paused: list[tuple[Path, object]] = []

        def find_job(self, job_id: str):
            return Path("working") / job_id, self.manifest

        def active_worker_pid(self) -> int:
            return 123

        def mark_paused(self, job_dir: Path, manifest):
            manifest.status = "paused"
            self.paused.append((job_dir, manifest))
            return Path("incoming") / manifest.job_id, manifest

    class FakeService:
        def __init__(self) -> None:
            self.store = FakeStore()
            self.resume_states: list[tuple[Path, object]] = []

        def _write_resume_state(self, job_dir: Path, manifest) -> None:
            self.resume_states.append((job_dir, manifest))

    class FakeProcess:
        def __init__(self, pid: int, children: list | None = None) -> None:
            self.pid = pid
            self._children = list(children or [])
            self.terminated = False
            self.killed = False

        def cmdline(self) -> list[str]:
            return ["python", "-m", "local_subtitle_stack.cli", "worker"]

        def children(self, recursive: bool = False) -> list:
            return list(self._children)

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

    child = FakeProcess(456)
    parent = FakeProcess(123, [child])
    state = WebServiceState.__new__(WebServiceState)
    state.service = FakeService()
    state.worker_process = SimpleNamespace(pid=123)
    monkeypatch.setattr("local_subtitle_stack.web_ui.psutil.Process", lambda _pid: parent)
    monkeypatch.setattr("local_subtitle_stack.web_ui.psutil.wait_procs", lambda processes, timeout: (processes, []))

    assert state.force_stop_job("job-1") == {"job_id": "job-1"}
    assert parent.terminated is True
    assert child.terminated is True
    assert state.service.store.paused[0][0] == Path("working") / "job-1"
    assert state.service.resume_states[0][0] == Path("incoming") / "job-1"


def test_resume_stuck_job_refuses_unknown_worker_process(monkeypatch) -> None:
    class FakeStore:
        def find_job(self, job_id: str):
            return Path("working") / job_id, SimpleNamespace(status=JOB_STATUS_WORKING)

        def active_worker_pid(self) -> int:
            return 123

    class FakeService:
        store = FakeStore()

        def resume(self, _job_id: str):
            raise AssertionError("unknown process must block resume")

    class FakeProcess:
        def cmdline(self) -> list[str]:
            return ["python", "other-script.py"]

    state = WebServiceState.__new__(WebServiceState)
    state.service = FakeService()
    state.worker_process = None
    monkeypatch.setattr("local_subtitle_stack.web_ui.psutil.Process", lambda _pid: FakeProcess())

    with pytest.raises(RuntimeError, match="Refusing to stop unknown process"):
        state.resume_job("job-1")


def test_web_ui_formats_model_sizes() -> None:
    assert format_bytes(4_435_931_324) == "4.1 GB"


def test_web_ui_resolves_ollama_model_manifest_and_blob_paths(tmp_path: Path) -> None:
    root = tmp_path / "ollama-models"
    manifest_path = Path(ollama_manifest_path(str(root), "fredrezones55/Gemma-4-Uncensored-HauhauCS-Aggressive:e2b"))
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        '{"layers":[{"digest":"sha256:abc123"},{"digest":"sha256:def456"}]}',
        encoding="utf-8",
    )

    assert manifest_path == root / "manifests" / "registry.ollama.ai" / "fredrezones55" / "Gemma-4-Uncensored-HauhauCS-Aggressive" / "e2b"
    assert ollama_blob_paths(str(root), str(manifest_path), "sha256:fallback") == [
        str(root / "blobs" / "sha256-abc123"),
        str(root / "blobs" / "sha256-def456"),
        str(root / "blobs" / "sha256-fallback"),
    ]


def test_web_ui_closes_old_web_processes_only(monkeypatch) -> None:
    class FakeProcess:
        def __init__(self, pid: int, command: list[str]) -> None:
            self.pid = pid
            self.info = {"pid": pid, "cmdline": command}
            self.terminated = False
            self.killed = False

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

    current = FakeProcess(10, ["python", "-m", "local_subtitle_stack", "web-ui"])
    parent = FakeProcess(9, ["python", "-m", "local_subtitle_stack", "web-ui"])
    old_web = FakeProcess(11, ["python", "-m", "local_subtitle_stack.cli", "web-ui", "--port", "8767"])
    worker = FakeProcess(12, ["python", "-m", "local_subtitle_stack.cli", "worker"])
    other = FakeProcess(13, ["python", "-m", "other_app", "web-ui"])

    monkeypatch.setattr("local_subtitle_stack.web_ui.os.getpid", lambda: 10)
    monkeypatch.setattr("local_subtitle_stack.web_ui.psutil.Process", lambda _pid: type("Current", (), {"parents": lambda _self: [parent]})())
    monkeypatch.setattr("local_subtitle_stack.web_ui.psutil.process_iter", lambda _attrs: [parent, current, old_web, worker, other])
    monkeypatch.setattr("local_subtitle_stack.web_ui.psutil.wait_procs", lambda processes, timeout: (processes, []))

    close_other_web_ui_processes()

    assert old_web.terminated is True
    assert worker.terminated is False
    assert other.terminated is False
    assert current.terminated is False
    assert parent.terminated is False
