# Changelog

## 2026-06-25

- Applied an Amir UI Revamp repair pass to make completed, paused, and model status labels more honest, reduce oversized text metrics, stop the header floating over long web UI workflows, and keep queue picker buttons readable inside the narrow input panel.
- Fixed adaptive GPU detection on newer Windows `nvidia-smi` output by falling back to parsing the visible memory table when `memory.free,memory.total` query fields are unavailable.
- Changed built-in translation defaults to Gemma 4 Uncensored for both direct and context-applied English so reset/new configs do not fall back to Qwen.
- Added an always-visible progress bar to running job cards so live progress still appears when another completed job is selected.
- Added live worker CPU, RAM, GPU, and VRAM readouts beside the frozen-progress hints in the web UI.
- Added live progress-age hints in the web job list and running progress card, showing likely active, watch, or possible stuck based on last saved progress.
- Added `Force stop now` in the web UI for long-running jobs, terminating only the owned worker process tree and pausing the job immediately.
- Added a web `Resume this job` / `Resume stuck job` control that requeues paused, failed, queued, or stuck working jobs and restarts processing.
- Made web line edits auto-save into the preview and subtitle files, and added the context-applied English column to the preview grid.
- Combined Health and Redo log into one collapsible web diagnostics panel with side-by-side desktop columns and a single mobile stack.
- Fixed the web model defaults panel order so `Transcription and translation models` stays at the top of the review workflow before jobs and subtitle review panels.
- Added per-job stop controls in the web UI so a queued job pauses before starting and a running job stops after the next safe checkpoint without pausing the whole queue.
- Made every web UI section collapsible with saved browser state so long workflow panels can be hidden without losing the current job page.
- Fixed wrapped web panel headers so long titles and action buttons do not overlap in narrow columns.
- Fixed web job deletion on Windows by hiding jobs from the queue list instead of deleting locked job folders or partial files.
- Moved `Overall video context` above the subtitle preview/editor in the web review column so context is set before reviewing or rerunning lines.
- Added visible feedback for `Restore before` in the web second-pass changes list, showing restoring, restored, or failed states per line.
- Clarified the web line editor so Direct English translation is read-only reference text, while Context-applied English is the editable final subtitle and auto-saves after typing.
- Fixed web second-pass change rows so clicking a changed line selects it, highlights the matching subtitle preview row, and opens the editable line fields.
- Changed web model settings to auto-save immediately when dropdowns or picker buttons change, removing the manual `Save settings` button.
- Added live web progress bars for running steps, including second-pass coherence review progress by subtitle group.
- Added an `X` button beside each web job so individual completed, failed, or paused jobs can be removed directly from the Jobs list.
- Tightened the web model settings layout so long model names fit better, and removed the redundant `Use Gemma e2b` and `Download Gemma` buttons now that model dropdowns handle selection.
- Added an `Unsaved changes` indicator to model settings so dropdown changes clearly need Save settings before new jobs use them.
- Fixed web model settings polling so changing the Japanese listening model to Qwen3-ASR no longer jumps back to the saved Kotoba value before Save settings.
- Clarified the preview line editor boxes with visible labels and short explanations for Japanese source subtitles, direct English translation, context-applied English, and reference subtitles.
- Added an optional second-pass coherence review for context-applied English subtitles, using saved overall/time-range context plus neighboring subtitle context and showing before/after changed lines in the web UI.
- Fixed Qwen3-ASR 1.7B long-sample coverage by using 30-second Qwen chunks and more generation headroom; live DANDY 90s proof now reaches 91.78s with monotonic forced-aligned cues.
- Fixed web subtitle preview rows so time, Japanese, and direct English text stay inside row boxes, wrap cleanly, and selected rows no longer overlap nearby subtitle lines.
- Added Qwen3-ASR 0.6B and 1.7B Japanese listening choices, routed them through the optional `qwen-asr` package with `Qwen/Qwen3-ForcedAligner-0.6B` timestamps, and recorded speaker separation as not enabled yet.
- Added surrounding Japanese subtitle context to selected-range English retranslation prompts, while still replacing only the selected subtitle lines.
- Added source-side resume state JSON beside exported subtitles and made the web preview poll the selected job so partial Japanese subtitles appear while transcription is still running.
- Audited the full web UI scroll layout on desktop and mobile, then tightened the workflow order: compact mobile step cards, two-column desktop model settings, existing-subtitle import beside input selection, jobs beside review, and automatic first-job selection on reload.
- Rebuilt the web subtitle preview as a side-by-side timing grid with Japanese subtitles and direct English translation in separate wrapping columns, preserving multi-select and preventing long translated lines from overlapping nearby rows.
- Added job-list clearing controls and rebuilt subtitle preview rows so time stays in a left column, subtitle text wraps on the right, selected rows do not overlap, and preview time can switch between seconds and `HH:MM:SS.mmm`.
- Renamed the existing-subtitle import panel to `Load subtitle files into preview`, clarified that it creates/attaches SRT files for the Preview and line editor, and made attach actions reload the selected job preview.
- Moved model configuration to the top of the web workflow column, fixed dark dropdown option contrast, and clarified that the Japanese model cache stores Hugging Face ASR/listening files while Gemma/Ollama English models stay in Ollama storage.
- Expanded the web Models panel so Direct English translation and Context-applied English show their exact Ollama storage root, manifest file, and blob file paths.
- Split web translation context into overall video context and time-range context, with Shift-click range selection, automatic start/end time fill, editable context range cards, and selected-range retranslation wording.
- Renamed user-facing translation labels across web, desktop, CLI help, and docs: direct output is now `direct English translation`, and adapted output is now `context-applied English`.
- Clarified the web context panel so overall context is described as English-translation prompt context, renamed batch label to series/project name, and made scene-note wording time-range specific.
- Fixed web worker status after restarting the browser UI, auto-selecting newly loaded jobs, closing stale web UI instances on launch, and showing translation progress as soon as a subtitle group starts.
- Split the selected-job controls so job actions and Subtitle Edit open actions are visibly separate, with every Subtitle Edit button labeled by destination.
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
