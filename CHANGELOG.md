# Changelog

## 2026-04-28

- Added quality-first Japanese subtitle settings with Kotoba as the default ASR path and faster-whisper large-v3 as the fast local fallback.
- Added adaptive faster-whisper transcription, benchmark tooling, and a resumable quality-job runner for long Japanese videos.
- Added translation retry handling, review artifacts, cue timing cleanup, and tests for non-overlapping SRT exports.
