from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import load_config
from .integrations import OllamaClient
from .utils import VIDEO_EXTENSIONS, no_window_creationflags


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Fast Transcriber</title>
  <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
  <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
  <style>
    :root {
      color-scheme: dark;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: #101113;
      color: #f7f9fb;
      --bg: #101113;
      --surface: #171b1f;
      --surface-2: #20272d;
      --line: rgba(232,236,238,.18);
      --muted: #b3bdc3;
      --soft: #f0f3f4;
      --accent: #ffd84d;
      --danger: #ffd1d1;
      --danger-text: #5f1515;
      --shadow: 0 18px 60px rgba(0,0,0,.28);
    }
    * { box-sizing: border-box; }
    html { min-width: 320px; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at top left, rgba(236,226,181,.10), transparent 34rem),
        linear-gradient(135deg, #101113, #171b1f 68%);
      overflow-y: auto;
    }
    button, select, input { font: inherit; }
    button {
      min-height: 38px;
      border: 1px solid transparent;
      border-radius: 6px;
      background: var(--accent);
      color: #111827;
      cursor: pointer;
      font-weight: 700;
      padding: 9px 12px;
      white-space: nowrap;
    }
    button.secondary { background: transparent; color: var(--soft); border-color: var(--line); }
    button.danger { background: var(--danger); color: var(--danger-text); }
    button:disabled { cursor: not-allowed; opacity: .45; }
    input, select {
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: rgba(255,255,255,.05);
      color: var(--soft);
      padding: 9px 10px;
      min-width: 0;
    }
    input::placeholder { color: #8797a4; }
    main {
      width: min(1480px, 100%);
      margin: 0 auto;
      padding: clamp(14px, 2vw, 24px);
    }
    .topbar {
      position: sticky;
      top: 0;
      z-index: 5;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 16px;
      align-items: center;
      padding: 14px 0 16px;
      background: linear-gradient(180deg, rgba(16,17,19,.98), rgba(16,17,19,.86) 72%, transparent);
      backdrop-filter: blur(10px);
    }
    h1 { margin: 0; font-size: clamp(24px, 3vw, 38px); letter-spacing: 0; line-height: 1.08; }
    p { color: var(--muted); line-height: 1.5; margin: 6px 0 0; }
    .layout {
      display: grid;
      grid-template-columns: minmax(360px, .9fr) minmax(420px, 1.35fr);
      gap: 14px;
      align-items: start;
    }
    .stack { display: grid; gap: 14px; min-width: 0; }
    .panel {
      background: color-mix(in srgb, var(--surface) 86%, transparent);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      min-width: 0;
      overflow: clip;
    }
    .panel-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,.03);
    }
    .panel-head strong { font-size: 15px; }
    .panel-body { padding: 16px; }
    .controls {
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      gap: 10px;
      align-items: end;
    }
    .control-span-12 { grid-column: span 12; }
    .control-span-6 { grid-column: span 6; }
    .control-span-4 { grid-column: span 4; }
    .control-span-3 { grid-column: span 3; }
    .field { display: grid; gap: 6px; min-width: 0; }
    .field label, .check-label { color: var(--muted); font-size: 12px; font-weight: 600; }
    .check-label {
      display: flex;
      gap: 8px;
      align-items: center;
      min-height: 38px;
      padding: 0 4px;
    }
    .button-row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    .path { font-family: Consolas, "Cascadia Mono", monospace; color: #d8e4e6; overflow-wrap: anywhere; }
    .target-list {
      display: grid;
      gap: 8px;
      max-height: min(34vh, 340px);
      overflow: auto;
      padding-right: 2px;
    }
    .item {
      display: grid;
      gap: 2px;
      padding: 10px 12px;
      background: rgba(0,0,0,.18);
      border: 1px solid rgba(255,255,255,.06);
      border-radius: 6px;
    }
    .empty {
      border: 1px dashed var(--line);
      border-radius: 8px;
      color: var(--muted);
      padding: 18px;
      text-align: center;
    }
    .status-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }
    .metric {
      border: 1px solid rgba(255,255,255,.08);
      border-radius: 8px;
      padding: 12px;
      background: rgba(0,0,0,.14);
      min-width: 0;
    }
    .metric span { color: var(--muted); display: block; font-size: 12px; }
    .metric strong { display: block; font-size: clamp(20px, 3vw, 32px); margin-top: 2px; }
    .model-list { display: grid; gap: 8px; }
    .model-row {
      display: grid;
      grid-template-columns: 150px minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
      padding: 8px 0;
      border-top: 1px solid rgba(255,255,255,.06);
    }
    .model-row:first-child { border-top: 0; padding-top: 0; }
    .model-row span:first-child { color: var(--muted); font-size: 12px; font-weight: 700; }
    progress { width: 100%; height: 18px; accent-color: var(--accent); margin-top: 14px; }
    pre {
      white-space: pre-wrap;
      overflow: auto;
      max-height: min(48vh, 560px);
      min-height: 260px;
      margin: 0;
      background: #07111a;
      border-top: 1px solid var(--line);
      color: #cbd5d7;
      padding: 14px 16px;
      line-height: 1.45;
      contain: content;
    }
    .error { border-color: rgba(255,209,209,.65); }
    .error .panel-head { color: var(--danger); }
    footer {
      margin: 18px 0 4px;
      color: #8ea0a4;
      font-size: 13px;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }
    a { color: var(--accent); }
    @media (max-width: 980px) {
      .layout { grid-template-columns: 1fr; }
      .topbar { grid-template-columns: 1fr; }
      .topbar .button-row { justify-content: flex-start; }
    }
    @media (max-width: 640px) {
      main { padding: 12px; }
      .controls { grid-template-columns: 1fr; }
      .control-span-12, .control-span-6, .control-span-4, .control-span-3 { grid-column: 1; }
      .button-row button, .button-row select { flex: 1 1 150px; }
      .status-grid { grid-template-columns: 1fr; }
      .model-row { grid-template-columns: 1fr; gap: 4px; }
      pre { min-height: 180px; }
    }
  </style>
</head>
<body>
  <main id="root"></main>
  <script>
    const e = React.createElement;
    const api = (path, body) => fetch(path, {
      method: body ? "POST" : "GET",
      headers: body ? {"Content-Type": "application/json"} : {},
      body: body ? JSON.stringify(body) : undefined
    }).then(async r => {
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || r.statusText);
      return data;
    });

    function App() {
      const [targets, setTargets] = React.useState([]);
      const [profile, setProfile] = React.useState("low_gpu");
      const [recursive, setRecursive] = React.useState(true);
      const [status, setStatus] = React.useState({running:false, completed:0, total:0, log:""});
      const [models, setModels] = React.useState({storage:"", hf_cache:"", selected:[]});
      const [error, setError] = React.useState("");
      const [manualPath, setManualPath] = React.useState("");

      const refresh = () => api("/api/status").then(setStatus).catch(err => setError(err.message));
      const refreshModels = () => api("/api/models").then(setModels).catch(err => setError(err.message));
      React.useEffect(() => { refresh(); refreshModels(); const id = setInterval(refresh, 1500); return () => clearInterval(id); }, []);

      const pickFolder = () => { setError(""); api("/api/pick-folder").then(d => d.path && setTargets([d.path])).catch(err => setError(err.message)); };
      const pickFiles = () => { setError(""); api("/api/pick-files").then(d => d.paths?.length && setTargets(d.paths)).catch(err => setError(err.message)); };
      const addManual = () => {
        const value = manualPath.trim();
        if (!value) return;
        setTargets(current => current.includes(value) ? current : current.concat(value));
        setManualPath("");
        setError("");
      };
      const start = () => { setError(""); api("/api/start", {targets, profile, recursive}).then(setStatus).catch(err => setError(err.message)); };
      const cancel = () => { setError(""); api("/api/cancel", {}).then(setStatus).catch(err => setError(err.message)); };
      const pct = status.total ? Math.round(status.completed / status.total * 100) : 0;
      const runningText = status.running ? "Running" : (status.state || "Idle");

      return e(React.Fragment, null,
        e("header", {className:"topbar"},
          e("div", null, e("h1", null, "Fast Transcriber"), e("p", null, "Local folder and file transcription using the existing adaptive runner.")),
          e("div", {className:"button-row"},
            e("button", {className:"secondary", onClick:refresh}, "Refresh"),
            e("button", {onClick:start, disabled:status.running || !targets.length}, "Start"),
            e("button", {className:"danger", onClick:cancel, disabled:!status.running}, "Cancel")
          )
        ),
        e("div", {className:"layout"},
          e("div", {className:"stack"},
            e("section", {className:"panel"},
              e("div", {className:"panel-head"}, e("strong", null, "Inputs"), e("span", null, `${targets.length} selected`)),
              e("div", {className:"panel-body stack"},
                e("div", {className:"controls"},
                  e("div", {className:"field control-span-12"},
                    e("label", null, "Manual path"),
                    e("input", {value:manualPath, onChange:ev=>setManualPath(ev.target.value), onKeyDown:ev=>{ if (ev.key === "Enter") addManual(); }, disabled:status.running, placeholder:"Paste a folder or video path"})
                  ),
                  e("button", {className:"control-span-4", onClick:pickFolder, disabled:status.running}, "Select folder"),
                  e("button", {className:"control-span-4", onClick:pickFiles, disabled:status.running}, "Select files"),
                  e("button", {className:"secondary control-span-4", onClick:addManual, disabled:status.running || !manualPath.trim()}, "Add path"),
                  e("div", {className:"field control-span-6"},
                    e("label", null, "Profile"),
                    e("select", {value:profile, onChange:ev=>setProfile(ev.target.value), disabled:status.running},
                      ["auto","high","balanced","low_gpu","cpu_fallback"].map(v => e("option", {key:v, value:v}, v))
                    )
                  ),
                  e("label", {className:"check-label control-span-6"},
                    e("input", {type:"checkbox", checked:recursive, onChange:ev=>setRecursive(ev.target.checked), disabled:status.running}),
                    "Search subfolders"
                  )
                ),
                e("div", {className:"button-row"},
                  e("button", {className:"secondary", onClick:()=>setTargets([]), disabled:status.running || !targets.length}, "Clear selected paths")
                ),
                targets.length
                  ? e("div", {className:"target-list"}, targets.map((path, index) => e("div", {className:"item", key:path}, e("span", null, `Source ${index + 1}`), e("span", {className:"path"}, path))))
                  : e("div", {className:"empty"}, "Select a folder, select files, or paste a path.")
              )
            ),
            error ? e("section", {className:"panel error"}, e("div", {className:"panel-head"}, e("strong", null, "Error")), e("div", {className:"panel-body"}, e("p", null, error))) : null
          ),
          e("div", {className:"stack"},
            e("section", {className:"panel"},
              e("div", {className:"panel-head"}, e("strong", null, "Run status"), e("span", null, runningText)),
              e("div", {className:"panel-body"},
                e("div", {className:"status-grid"},
                  e("div", {className:"metric"}, e("span", null, "Progress"), e("strong", null, `${pct}%`)),
                  e("div", {className:"metric"}, e("span", null, "Completed"), e("strong", null, status.completed || 0)),
                  e("div", {className:"metric"}, e("span", null, "Total"), e("strong", null, status.total || 0))
                ),
                e("progress", {value:status.completed || 0, max:status.total || 1}),
                status.current ? e("p", {className:"path"}, status.current) : e("p", null, "No active file.")
              )
            ),
            e("section", {className:"panel"},
              e("div", {className:"panel-head"}, e("strong", null, "Models"), e("button", {className:"secondary", onClick:refreshModels}, "Refresh")),
              e("div", {className:"panel-body model-list"},
                e("div", {className:"model-row"}, e("span", null, "Ollama storage"), e("span", {className:"path"}, models.storage || "Unknown"), e("span", null, "")),
                (models.selected || []).map(item => e("div", {className:"model-row", key:item.label}, e("span", null, item.label), e("span", {className:"path"}, item.name), e("span", null, item.size || "Not installed"))),
                e("div", {className:"model-row"}, e("span", null, "Japanese cache"), e("span", {className:"path"}, models.hf_cache || "Default Hugging Face cache"), e("span", null, ""))
              )
            ),
            e("section", {className:"panel"},
              e("div", {className:"panel-head"}, e("strong", null, "Log"), e("span", null, status.running ? "Live" : "Idle")),
              e("pre", null, status.log || "Run output appears here.")
            )
          )
        ),
        e("footer", null, e("span", null, "Built by Amir. Follow Amir at followamir.com."), e("a", {href:"https://www.paypal.com/donate/?hosted_button_id=2U2GXSKFJKJCA"}, "Donate"))
      );
    }
    ReactDOM.createRoot(document.getElementById("root")).render(e(App));
  </script>
</body>
</html>"""


def count_sources(targets: list[Path], recursive: bool) -> int:
    total = 0
    for target in targets:
        if target.is_file() and target.suffix.lower() in VIDEO_EXTENSIONS:
            total += 1
        elif target.is_dir():
            iterator = target.rglob("*") if recursive else target.iterdir()
            total += sum(1 for path in iterator if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)
    return total


def count_completed_outputs(targets: list[Path], recursive: bool) -> int:
    total = 0
    for target in targets:
        if target.is_file():
            total += int((target.parent / f"{target.stem}.raw.meta.json").exists())
        elif target.is_dir():
            iterator = target.rglob("*.raw.meta.json") if recursive else target.glob("*.raw.meta.json")
            total += sum(1 for _ in iterator)
    return total


class WebTranscriberState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.process: subprocess.Popen[str] | None = None
        self.targets: list[Path] = []
        self.recursive = True
        self.total = 0
        self.log = ""
        self.state = "idle"
        self.current = ""

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            running = self.process is not None and self.process.poll() is None
            completed = count_completed_outputs(self.targets, self.recursive) if self.targets else 0
            if self.process is not None and not running and self.state == "running":
                self.state = "complete" if self.process.returncode == 0 else "failed"
            return {
                "running": running,
                "state": self.state,
                "completed": min(completed, self.total),
                "total": self.total,
                "current": self.current,
                "log": self.log[-6000:],
            }

    def start(self, targets: list[Path], profile: str, recursive: bool) -> dict[str, Any]:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                raise RuntimeError("A transcription run is already active.")
            missing = [str(path) for path in targets if not path.exists()]
            if missing:
                raise RuntimeError(f"Path not found: {missing[0]}")
            self.targets = targets
            self.recursive = recursive
            self.total = count_sources(targets, recursive)
            if self.total == 0:
                raise RuntimeError("No supported video files found.")
            self.log = ""
            self.state = "running"
            self.current = str(targets[0]) if len(targets) == 1 else f"{len(targets)} selected files"
            commands = [self._command(path, profile, recursive) for path in targets]
            thread = threading.Thread(target=self._run_commands, args=(commands,), daemon=True)
            thread.start()
        return self.snapshot()

    def cancel(self) -> dict[str, Any]:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                self.state = "cancelled"
        return self.snapshot()

    def _command(self, target: Path, profile: str, recursive: bool) -> list[str]:
        output_root = target if target.is_dir() else target.parent
        command = [
            sys.executable,
            "-m",
            "local_subtitle_stack.cli",
            "transcribe-batch",
            "--input-dir",
            str(target),
            "--output-dir",
            str(output_root),
            "--language",
            "en",
            "--profile",
            profile,
            "--low-memory-policy",
            "downgrade",
        ]
        if recursive and target.is_dir():
            command.append("--recursive")
        return command

    def _run_commands(self, commands: list[list[str]]) -> None:
        return_code = 0
        for command in commands:
            with self.lock:
                if self.state == "cancelled":
                    return
                self.current = command[command.index("--input-dir") + 1]
                self.process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    creationflags=no_window_creationflags(),
                )
                process = self.process
            assert process.stdout is not None
            for line in process.stdout:
                with self.lock:
                    self.log += line
            return_code = process.wait()
            if return_code != 0:
                break
        with self.lock:
            self.state = "complete" if return_code == 0 and self.state != "cancelled" else "failed"


STATE = WebTranscriberState()


def format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{value} B"
        size /= 1024
    return f"{value} B"


def model_storage_snapshot() -> dict[str, Any]:
    config = load_config()
    ollama = OllamaClient(executable_path=config.tools.ollama)
    try:
        details = ollama.list_model_details()
    except Exception:
        details = {}

    def selected(label: str, name: str) -> dict[str, str]:
        detail = details.get(name)
        size = detail.get("size") if detail else None
        return {
            "label": label,
            "name": name,
            "size": format_bytes(int(size)) if isinstance(size, int) else "",
        }

    return {
        "storage": ollama.model_storage_root(),
        "hf_cache": config.cache_paths.hf_hub_cache,
        "selected": [
            selected("Direct English", config.models.literal_translation),
            selected("Natural English", config.models.adapted_translation),
        ],
    }


def _run_windows_picker(script: str) -> list[str]:
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-STA", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "Windows picker failed.")
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def pick_folder() -> str:
    paths = _run_windows_picker(
        """
        Add-Type -AssemblyName System.Windows.Forms
        $owner = New-Object System.Windows.Forms.Form
        $owner.TopMost = $true
        $owner.WindowState = 'Minimized'
        $owner.ShowInTaskbar = $false
        $owner.Show() | Out-Null
        $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
        $dialog.Description = 'Select folder to transcribe'
        if ($dialog.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) {
            [Console]::Out.WriteLine($dialog.SelectedPath)
        }
        $owner.Dispose()
        """
    )
    return paths[0] if paths else ""


def pick_files() -> list[str]:
    return _run_windows_picker(
        """
        Add-Type -AssemblyName System.Windows.Forms
        $owner = New-Object System.Windows.Forms.Form
        $owner.TopMost = $true
        $owner.WindowState = 'Minimized'
        $owner.ShowInTaskbar = $false
        $owner.Show() | Out-Null
        $dialog = New-Object System.Windows.Forms.OpenFileDialog
        $dialog.Title = 'Select videos to transcribe'
        $dialog.Multiselect = $true
        $dialog.Filter = 'Video files|*.mp4;*.mkv;*.avi;*.mov;*.wmv;*.m4v;*.webm|All files|*.*'
        if ($dialog.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) {
            foreach ($name in $dialog.FileNames) {
                [Console]::Out.WriteLine($name)
            }
        }
        $owner.Dispose()
        """
    )


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_html(HTML)
        elif path == "/api/status":
            self._send_json(STATE.snapshot())
        elif path == "/api/models":
            self._send_json(model_storage_snapshot())
        elif path == "/api/pick-folder":
            self._send_json({"path": pick_folder()})
        elif path == "/api/pick-files":
            self._send_json({"paths": pick_files()})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/api/start":
                targets = [Path(value).expanduser() for value in payload.get("targets", [])]
                self._send_json(STATE.start(targets, payload.get("profile", "low_gpu"), bool(payload.get("recursive", True))))
            elif self.path == "/api/cancel":
                self._send_json(STATE.cancel())
            else:
                self.send_error(404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_html(self, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main(argv: list[str] | None = None) -> int:
    port = int(argv[0]) if argv else int(os.environ.get("FAST_TRANSCRIBER_PORT", "8765"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Fast Transcriber UI: {url}")
    webbrowser.open(url)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
