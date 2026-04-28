# Changelog

## 2026-04-28

- Bumped the app version to `0.3.0` for the renamed quality-first release line.
- Added a Japanese ASR model-evaluation note and an in-code candidate registry that ranks models by Japanese CER first.
- Added an opt-in ReazonSpeech k2 ASR engine that uses short CPU chunks and records benchmark evidence in job checkpoints.
- Renamed the visible Windows app and release bundle to `Japanese to English Subtitler` while keeping the older CLI aliases working.
- Added quality-first Japanese subtitle settings with Kotoba as the default ASR path and faster-whisper large-v3 as the fast local fallback.
- Added adaptive faster-whisper transcription, benchmark tooling, and a resumable quality-job runner for long Japanese videos.
- Added translation retry handling, review artifacts, cue timing cleanup, and tests for non-overlapping SRT exports.
