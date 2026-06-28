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
    :root { color-scheme: dark; font-family: "Segoe UI", sans-serif; background: #101316; color: #eef3f4; }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; background: linear-gradient(135deg, #101316, #172023); }
    main { max-width: 980px; margin: 0 auto; padding: 32px 18px; }
    header { display: flex; justify-content: space-between; gap: 16px; align-items: end; margin-bottom: 22px; }
    h1 { margin: 0; font-size: 32px; letter-spacing: 0; }
    p { color: #b9c5c7; line-height: 1.55; }
    .panel { background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.12); border-radius: 8px; padding: 18px; margin: 14px 0; }
    .row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
    button, select, input { border-radius: 6px; border: 1px solid rgba(255,255,255,.18); background: #1f2a2d; color: #eef3f4; padding: 10px 12px; font: inherit; }
    button { cursor: pointer; background: #d9f99d; color: #111827; font-weight: 700; }
    button.secondary { background: #263236; color: #eef3f4; }
    button.danger { background: #fecaca; color: #7f1d1d; }
    button:disabled { opacity: .48; cursor: not-allowed; }
    .path { font-family: Consolas, monospace; color: #d8e4e6; overflow-wrap: anywhere; }
    .list { display: grid; gap: 8px; margin-top: 12px; }
    .item { padding: 10px; background: rgba(0,0,0,.18); border-radius: 6px; }
    progress { width: 100%; height: 18px; accent-color: #d9f99d; }
    pre { white-space: pre-wrap; max-height: 260px; overflow: auto; background: #0b0f10; padding: 12px; border-radius: 6px; color: #cbd5d7; }
    footer { margin-top: 24px; color: #8ea0a4; font-size: 13px; display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
    a { color: #d9f99d; }
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
      const [error, setError] = React.useState("");

      const refresh = () => api("/api/status").then(setStatus).catch(err => setError(err.message));
      React.useEffect(() => { refresh(); const id = setInterval(refresh, 1500); return () => clearInterval(id); }, []);

      const pickFolder = () => api("/api/pick-folder").then(d => d.path && setTargets([d.path])).catch(err => setError(err.message));
      const pickFiles = () => api("/api/pick-files").then(d => d.paths?.length && setTargets(d.paths)).catch(err => setError(err.message));
      const start = () => api("/api/start", {targets, profile, recursive}).then(setStatus).catch(err => setError(err.message));
      const cancel = () => api("/api/cancel", {}).then(setStatus).catch(err => setError(err.message));
      const pct = status.total ? Math.round(status.completed / status.total * 100) : 0;

      return e(React.Fragment, null,
        e("header", null,
          e("div", null, e("h1", null, "Fast Transcriber"), e("p", null, "Local folder and file transcription using the existing adaptive runner.")),
          e("button", {className:"secondary", onClick:refresh}, "Refresh")
        ),
        e("section", {className:"panel"},
          e("div", {className:"row"},
            e("button", {onClick:pickFolder, disabled:status.running}, "Select folder"),
            e("button", {onClick:pickFiles, disabled:status.running}, "Select files"),
            e("select", {value:profile, onChange:ev=>setProfile(ev.target.value), disabled:status.running},
              ["auto","high","balanced","low_gpu","cpu_fallback"].map(v => e("option", {key:v, value:v}, v))
            ),
            e("label", null, e("input", {type:"checkbox", checked:recursive, onChange:ev=>setRecursive(ev.target.checked), disabled:status.running}), " Recursive"),
            e("button", {onClick:start, disabled:status.running || !targets.length}, "Start"),
            e("button", {className:"danger", onClick:cancel, disabled:!status.running}, "Cancel")
          ),
          e("div", {className:"list"}, targets.map(path => e("div", {className:"item path", key:path}, path)))
        ),
        e("section", {className:"panel"},
          e("div", {className:"row"}, e("strong", null, status.state || "idle"), e("span", null, `${status.completed || 0}/${status.total || 0} files (${pct}%)`)),
          e("progress", {value:status.completed || 0, max:status.total || 1}),
          status.current ? e("p", {className:"path"}, status.current) : null
        ),
        error ? e("section", {className:"panel"}, e("strong", null, "Error"), e("p", null, error)) : null,
        e("section", {className:"panel"}, e("strong", null, "Log"), e("pre", null, status.log || "")),
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


def pick_folder() -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        return filedialog.askdirectory(title="Select folder to transcribe")
    finally:
        root.destroy()


def pick_files() -> list[str]:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        paths = filedialog.askopenfilenames(
            title="Select videos to transcribe",
            filetypes=[("Video files", "*.mp4 *.mkv *.avi *.mov *.wmv *.m4v *.webm"), ("All files", "*.*")],
        )
        return list(paths)
    finally:
        root.destroy()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_html(HTML)
        elif path == "/api/status":
            self._send_json(STATE.snapshot())
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
