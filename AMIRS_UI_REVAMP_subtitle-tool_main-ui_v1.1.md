# Amir's UI Revamp Plan

## Run Metadata
- Project: subtitle-tool
- Target: main web UI
- Mode: Repair current UI
- Platform/framework: Plain Python HTTP server, React via CDN, inline CSS
- Plan version: v1.1
- Created: 2026-06-30
- Based on previous plan: `AMIRS_UI_REVAMP_subtitle-tool_main-ui_v1.0.md`
- User style direction: implement all audit fixes while preserving workflow
- Density target: balanced operational UI
- Screen-size expectations: current in-app Browser desktop viewport plus narrow mobile-style viewport

## User Intent Summary
Implement the reported web UI logic fixes so the app behaves predictably when jobs are running, model defaults are changed, time ranges are typed manually, settings are reset, and subtitle files are dropped repeatedly.

## Repo/UI Audit
Primary surface remains `src/local_subtitle_stack/web_ui.py`. Existing static web tests live in `tests/test_web_ui.py`. Prior solved-problem note confirms this app should preserve existing queue and worker behavior while adding focused guards.

## Real Feature Inventory
Queue inputs, model settings, Defaults reset, ASR folder picker, overall context redo, second-pass coherence review, selected time-range retranslation, time-range context cards, drag/drop SRT import, backend redo endpoints.

## Information Architecture Audit
No new visible sections needed. The main repair is interaction honesty: disable controls when background work makes the action unsafe, confirm destructive/default-reset actions, and show clear validation errors for invalid context ranges.

## Current UI Problems or New UI Requirements
- Queue controls used `status.running`, which the status API does not return.
- Redo/coherence/range buttons could launch while a worker or redo task held the queue lock.
- Local ASR folder picker could leave Qwen/Reazon selected and silently ignore the folder.
- Invalid time ranges could be added or sent to backend redo.
- Defaults reset app-wide model settings without confirmation.
- Dropped subtitle filenames could collide when numbered files already existed.

## Clutter Audit
No visual expansion needed. Keep current controls in place and add only minimal validation/disabled behavior.

## Design System Contract
Reuse current dark practical app styling, existing button variants, and existing error panel. No new tokens or dependencies.

## Component Contract
Changed controls: queue input buttons, ASR folder picker, Defaults button, redo/coherence/range buttons, time-range inputs/cards, drag/drop upload path.

## Structure and Layout Contract
Preserve current workbench shell, current panel order, diagnostics panel, model settings placement, and preview layout.

## Interaction and Feedback Contract
Controls that would conflict with a live worker or redo task must be disabled. Invalid time ranges must show clear error text before backend work starts. Defaults reset needs explicit confirmation.

## Interactive Component State Matrix
| Component | Parent | Interaction | States Checked | Bounding-Box Result | Evidence | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Queue manual/picker controls | Queue inputs panel | worker/redo busy state | enabled, disabled | no text overflow at desktop or 390px | focused tests, browser bounding boxes | passed |
| ASR folder picker | Model settings panel | pick folder path | auto-save with `asr_engine: "kotoba"` | no text overflow at desktop or 390px | focused tests, browser bounding boxes | passed |
| Defaults button | Model settings panel | click | confirm prompt | no text overflow at desktop or 390px | focused tests | passed |
| Redo/coherence/range buttons | Context/time-range panels | worker/redo busy state | enabled, disabled | no text overflow at desktop or 390px | focused tests, browser bounding boxes | passed |
| Time-range context inputs | Time-range panel | invalid manual input | valid, invalid, end-before-start | no text overflow; error shown live | focused tests, browser validation check | passed |
| Dropped subtitle upload | Preview editor drop zone | repeated filename upload | unique filename | not visual | focused tests | passed |

## Responsive Contract
Desktop and narrow checks must confirm no accidental horizontal overflow and no text clipping in changed controls.

## Platform-Specific Rules
Plain React and Python only. No dependency changes. Do not use browser physical mouse or keyboard input.

## Optional Imagegen Brief
Not applicable for this repair run.

## Goal Objective
Make web UI job actions and model/default/time-range behavior honest and collision-safe without redesigning the workflow.

## Stage-by-Stage Execution Plan
- Stage 1: confirm contract, route, and solved-problem note. Status: passed.
- Stage 2: patch web UI/backend guards and validation. Status: passed.
- Stage 3: add focused tests for every reported fix class. Status: passed.
- Stage 4: run focused and full test suites. Status: passed.
- Stage 5: verify live web UI at desktop and narrow viewport. Status: passed.
- Stage 6: commit and push after live verification. Status: pending.

## Verification Matrix
- `pytest -q tests/test_web_ui.py`: passed.
- `pytest -q`: passed.
- Live `http://127.0.0.1:8876/` desktop viewport 2202x1272: passed, no document horizontal overflow.
- Live narrow viewport 390x844: passed, no document horizontal overflow.
- Component text-fit checks for changed buttons: passed, no button overflow in checked controls.
- `/api/status` confirmed `worker_running=false`, `rebuild_running=false`, `pause_requested=false` before restarting only the web UI.
- Invalid time range check: passed, browser showed `End time must be after start time.` and `/api/status` remained idle.

## Screenshot/Visual Critique Log
Captured desktop and narrow in-app Browser screenshots after server reload. Desktop showed the model settings panel with `Local Kotoba/Hugging Face ASR folder`, readable queue buttons, and jobs list. Narrow screenshot showed queue controls stacked with no clipped button text.

## Attempt Log
- Attempt 1: applied focused fixes and tests; focused/full tests passed.
- Attempt 2: Browser docs call timed out, so the lightweight Browser attach probe was used. Tab control and CDP screenshot capture worked; Browser visibility API still reported false.

## Blockers and Risks
Browser visibility API reported false even after `visibility.set(true)`, but tab DOM, screenshots, viewport checks, and CDP capture worked. No worker was running during web server reload.

## Final Verification Report
Focused tests passed. Full test suite passed. Live desktop/narrow browser checks passed. Commit and push pending.
