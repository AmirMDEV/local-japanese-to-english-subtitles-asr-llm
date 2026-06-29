# Amir Goal Plan

## Objective
Make the web UI identify as Fast Multilanguage Transcriber and bring it to practical feature parity with the Python UI by reusing existing service methods.

## Follow-up Objective
Make the parity UI understandable: dropdown model/settings controls, clearer existing-subtitle import labels, multi-line subtitle selection for scene context, and selected-range retranslate workflow.

## Constraints
- Preserve: existing worker/service behavior, model config, queue layout, Subtitle Edit integration, tests.
- Do not change: core transcription/rebuild algorithms unless web parity needs a service wrapper.
- Allowed scope: `src/local_subtitle_stack/web_ui.py`, narrow tests, changelog.
- Not allowed: new frontend build stack, new dependencies, separate database, rewritten worker.

## Verification Surface
| Check | Command/artifact | Pass condition | Required? |
|---|---|---|---|
| Unit suite | `.venv311\Scripts\python.exe -m pytest` | all tests pass | yes |
| Web API smoke | HTTP requests to local preview | status/settings/jobs endpoints return JSON | yes |
| Visual desktop | Playwright screenshot 1440x1000 | no overflow/clipping, title correct | yes |
| Visual mobile | Playwright screenshot 390x844 | no overflow/clipping, title correct | yes |
| Git | `git status --short --branch` | clean and pushed | yes |

## Stages
| Stage | Status | Objective | Depends on | Verification | Attempts | Notes |
|---|---|---|---|---|---|---|
| 1 | passed | Audit Python-vs-web feature gap and rename app | none | title visible in HTML/screenshots | 1/3 | Python UI feature map gathered from `ui.py` and service methods |
| 2 | passed | Switch web backend from standalone CLI batch runner to shared `WorkerService` API endpoints | 1 | API smoke | 1/3 | `/api/status`, `/api/models`, `/api/health` smoke passed |
| 3 | passed | Add web controls for queue, worker, import, health, model settings | 2 | API smoke + tests | 1/3 | Controls render in screenshot and route to service endpoints |
| 4 | passed | Add selected job preview, line edit, notes, rebuild/open actions | 2 | API smoke + tests | 1/3 | Controls render in screenshot and route to service endpoints |
| 5 | passed | Responsive UI polish and visual verification | 3,4 | desktop/mobile screenshots | 1/3 | Desktop/mobile no overflow/clipping |
| 6 | passed | Final tests, changelog, commit, push | 5 | pytest + git clean | 1/3 | `88 passed`; commit `3393886` pushed |

## Subagent Plan
| Subagent | Purpose | Scope | Return format | Status |
|---|---|---|---|---|
| none | Main thread owns live UI and shared backend edits | n/a | n/a | skipped |

## Execution Log
- Stage 1 attempt 1: Loaded `amir-pursue-goal`, audited Tk button commands, service methods, current web routes. CodeGraph unavailable because project index missing.
- Stage 2 attempt 1: Chosen route is shared `WorkerService`, not duplicate web worker logic.
- Stages 2-5 attempt 1: Added service-backed web endpoints and React panels for queue, jobs, preview/editor, notes/rebuilds, models, settings, imports, health. API smoke passed. Desktop/mobile screenshots inspected; no overflow/clipping.
- Stage 6 attempt 1: `pytest` passed, changes committed and pushed.
- Follow-up stage active: replacing raw settings/import controls with guided controls and range selection.
- Follow-up stage passed: dropdown model/settings controls, path picker buttons, clearer existing-subtitle wording, multi-line subtitle selection, selected-range context/retranslation controls, tests, desktop/mobile screenshots.

## Blockers
| Stage | Blocker | Evidence | Tried | What would unblock |
|---|---|---|---|---|

## Definition of Done
- Web UI title and header say Fast Multilanguage Transcriber.
- Web UI exposes Python UI workflows through existing service methods where possible.
- Missing workflows are documented only if blocked by browser security or platform limitations.
- Tests pass.
- Desktop and mobile screenshots show no horizontal overflow or clipped controls.
- Changes committed and pushed.
