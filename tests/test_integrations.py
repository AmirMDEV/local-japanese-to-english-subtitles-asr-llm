from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest
import requests

from local_subtitle_stack.integrations import ExternalToolError, OllamaClient, ReazonSpeechK2ASRClient


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


def test_ollama_client_auto_starts_and_retries_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    server_started = False
    start_calls: list[list[str]] = []
    request_calls: list[tuple[str, str]] = []

    def fake_popen(args, **kwargs):
        nonlocal server_started
        server_started = True
        start_calls.append(list(args))

        class DummyProcess:
            pass

        return DummyProcess()

    def fake_request(method: str, url: str, timeout: float, **kwargs):
        request_calls.append((method, url))
        if not server_started:
            raise requests.ConnectionError("offline")
        if url.endswith("/api/tags"):
            return FakeResponse({"models": [{"name": "qwen3:4b-q8_0"}]})
        if url.endswith("/api/generate"):
            return FakeResponse({"response": json.dumps({"translations": ["line 1"]})})
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr("local_subtitle_stack.integrations.subprocess.Popen", fake_popen)
    monkeypatch.setattr("local_subtitle_stack.integrations.requests.request", fake_request)
    monkeypatch.setattr("local_subtitle_stack.integrations.time.sleep", lambda _seconds: None)

    client = OllamaClient(executable_path="ollama")

    payload = client.generate_json("qwen3:4b-q8_0", "translate this", temperature=0.0)

    assert payload == {"translations": ["line 1"]}
    assert start_calls == [["ollama", "serve"]]
    assert request_calls[0] == ("POST", "http://127.0.0.1:11434/api/generate")


def test_ollama_client_does_not_auto_start_for_remote_host(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_request(method: str, url: str, timeout: float, **kwargs):
        raise requests.ConnectionError("offline")

    def fail_popen(args, **kwargs):
        raise AssertionError("should not auto-start a remote Ollama host")

    monkeypatch.setattr("local_subtitle_stack.integrations.requests.request", fake_request)
    monkeypatch.setattr("local_subtitle_stack.integrations.subprocess.Popen", fail_popen)

    client = OllamaClient(base_url="http://192.168.0.6:11434", executable_path="ollama")

    with pytest.raises(requests.ConnectionError):
        client.list_models()


def test_ollama_client_reports_model_storage_and_details(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_request(method: str, url: str, timeout: float, **kwargs):
        assert url.endswith("/api/tags")
        return FakeResponse(
            {
                "models": [
                    {
                        "name": "gemma:e2b",
                        "size": 4_400_000_000,
                        "digest": "abc123",
                        "modified_at": "2026-06-29T12:00:00Z",
                    }
                ]
            }
        )

    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "ollama-models"))
    monkeypatch.setattr("local_subtitle_stack.integrations.requests.request", fake_request)

    client = OllamaClient()

    assert client.model_storage_root() == str(tmp_path / "ollama-models")
    assert client.list_model_details()["gemma:e2b"]["size"] == 4_400_000_000


def test_ollama_client_reports_remote_model_host() -> None:
    client = OllamaClient(base_url="http://192.168.0.6:11434")

    assert client.model_storage_root() == "Remote Ollama host: 192.168.0.6"


def test_reazonspeech_k2_missing_dependency_has_clear_message(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("reazonspeech"):
            raise ModuleNotFoundError("No module named 'reazonspeech'", name="reazonspeech")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    client = ReazonSpeechK2ASRClient("reazon-research/reazonspeech-k2-v2")

    with pytest.raises(ExternalToolError, match="ReazonSpeech k2 is optional"):
        client.transcribe_chunk(Path("missing.wav"), batch_size=1, device="cpu")
