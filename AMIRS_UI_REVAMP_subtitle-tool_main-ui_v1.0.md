# Amir's UI Revamp Plan

## Run Metadata
- Project: subtitle-tool
- Target: main web UI
- Mode: Repair and declutter/polish current UI
- Platform/framework: Plain Python HTTP server, React via CDN, inline CSS
- Plan version: v1.0
- Created: 2026-06-30
- Based on previous plan: none found
- User style direction: make the current UI better while preserving workflow
- Density target: balanced, still operational
- Screen-size expectations: 390px mobile, 1440px desktop

## User Intent Summary
Apply `amirs-ui-revamp` to improve the local subtitle web UI without changing core job behavior.

## Repo/UI Audit
Main target file is `src/local_subtitle_stack/web_ui.py`. Status data comes from `src/local_subtitle_stack/service.py`. Existing tests cover web markup and status rows.

## Real Feature Inventory
Queue inputs, subtitle import, jobs list, selected job controls, preview editor, context notes, time-range context, models, diagnostics, redo log, model settings.

## Information Architecture Audit
Primary flow remains queue -> model defaults -> job list -> selected job -> preview/context. Rare export actions stay in the Subtitle Edit subsection.

## Current UI Problems or New UI Requirements
Paused jobs read like active jobs. Completed jobs can show stale progress. Model labels can blur ASR and translation models. Queue picker buttons can clip text inside the real narrow input panel even when whole-page screenshots look acceptable.

## Clutter Audit
Keep visible actions stable, but make status text more honest so users do not need extra diagnostics to interpret state.

## Design System Contract
Use existing dark practical app tokens. Keep 8px max panel radius, yellow accent only for progress/primary actions, red only for destructive actions.

## Component Contract
Reuse existing panel, metric, progress, button, and job row patterns. No new component framework.

## Structure and Layout Contract
Preserve the current workbench shell and two-column desktop layout. Avoid introducing extra nested panels.

## Interaction and Feedback Contract
Status wording must match actual job state. Disabled actions must remain visually disabled. No job action should run during visual checks.

## Responsive Contract
Verify at desktop and 390px. No horizontal overflow. Also verify constrained parent panels directly with cropped screenshots and button text-fit checks.

## Platform-Specific Rules
Plain React/CSS only. No dependency changes.

## Optional Imagegen Brief
Not applicable for this run.

## Goal Objective
Make the UI more truthful and easier to scan by repairing stale/misleading status labels.

## Stage-by-Stage Execution Plan
- Stage 1: inspect live UI and status API. Status: passed.
- Stage 2: patch paused/completed/model label wording and queue picker button fit. Status: passed.
- Stage 3: verify web markup, status API, desktop/mobile smoke, constrained queue panel, and tests. Status: passed.

## Verification Matrix
- `pytest -q tests/test_web_ui.py tests/test_service.py::test_status_rows_include_stage_and_overall_progress`
- full `pytest -q`
- Playwright desktop and mobile smoke against `http://127.0.0.1:8876/`
- cropped `Queue inputs` panel screenshots at real/narrow parent widths
- DOM text-fit checks for visible buttons in `Queue inputs`
- live `/api/status`

## Screenshot/Visual Critique Log
Initial smoke: no desktop/mobile overflow, but paused selected job used active wording and completed job displayed stale progress.
Second pass: sticky topbar floated mid-screenshot after scroll, and selected-job text metrics were too large for long model/stage names.
Final smoke before user screenshot: desktop and mobile had no horizontal overflow, no console errors, no failed requests, completed jobs showed saved/100%, paused job showed paused/resume wording, and model metric said `Translation model`.
User screenshot regression: the first pass missed clipped queue picker buttons because the check used broad screenshots rather than cropped real parent-panel/text-fit evidence.
Final constrained-panel smoke: in-app Browser `Queue inputs` crop at the real desktop parent width showed `Select folder` and `Select files` as half-width buttons with full labels visible and `Add path` on its own full-width row. 390px viewport crop showed stacked picker buttons, wrapped long action buttons, no button text overflow, and no document horizontal overflow.

## Attempt Log
- Attempt 1: focused status-label repair passed.
- Attempt 2: compacted selected-job text metrics and removed sticky topbar after screenshot critique.
- Attempt 3: changed queue picker grid spans, allowed button text to wrap, and added component-level text-fit verification.

## Blockers and Risks
Server sometimes exits when no managed session remains. Treat as app lifecycle issue, not part of this visual wording pass.

## Final Verification Report
Focused tests passed. Full test suite passed. In-app Browser desktop and 390px constrained-panel checks passed with no `Queue inputs` button text overflow and no narrow viewport horizontal overflow.
