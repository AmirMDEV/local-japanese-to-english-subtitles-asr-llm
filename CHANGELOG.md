# Changelog

## 2026-06-25

- Fixed the web Jobs panel so completed jobs say saved, stale jobs load without `Failed to fetch`, and jobs can be deleted from the queue list without deleting exported subtitle files.
- Added drag-and-drop `.srt` import in the web preview editor so existing subtitles can be uploaded into a new job or attached to the selected job.
- Added a plain-language workflow guide and clearer review buttons so the web UI explains what to do at each stage.
- Made the web UI more guided: model settings now use dropdowns and picker buttons, existing-subtitle import has clearer labels, and subtitle preview supports multi-line selection for scene context and selected-range retranslation.
- Renamed the web UI to Fast Multilanguage Transcriber and added queue, job preview, line editing, notes, rebuild, import, health, and model-settings controls that mirror the Python UI workflows.
- Added model storage visibility for Ollama host/root, selected English model sizes, and the Japanese model cache folder.
- Reworked the local web UI into a responsive two-panel layout with stable scroll regions and resize-friendly controls.
- Added Ollama model dropdowns, refresh, and a one-click download/select path for `fredrezones55/Gemma-4-Uncensored-HauhauCS-Aggressive:e2b`.
- Updated the default Kotoba Japanese ASR model to `kotoba-tech/kotoba-whisper-v2.2`.
- Fixed adaptive batch transcription so per-video outputs honor the requested output folder.
- Added a bounded low-memory wait timeout for adaptive transcription.
- Added subtitle quality review flags for dense lines, long lines, Japanese leakage, repeated output, unchanged text, and glossary misses.

## 2026-04-28

- Bumped the app version to `0.3.0` for the renamed quality-first release line.
- Added a Japanese ASR model-evaluation note and an in-code candidate registry that ranks models by Japanese CER first.
- Added an opt-in ReazonSpeech k2 ASR engine that uses short CPU chunks and records benchmark evidence in job checkpoints.
- Renamed the visible Windows app and release bundle to `Japanese to English Subtitler` while keeping the older CLI aliases working.
- Added quality-first Japanese subtitle settings with Kotoba as the default ASR path and faster-whisper large-v3 as the fast local fallback.
- Added adaptive faster-whisper transcription, benchmark tooling, and a resumable quality-job runner for long Japanese videos.
- Added translation retry handling, review artifacts, cue timing cleanup, and tests for non-overlapping SRT exports.
