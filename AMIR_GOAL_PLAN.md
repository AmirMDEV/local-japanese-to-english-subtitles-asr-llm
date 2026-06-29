# Amir Goal Plan

## Objective
Add an optional second-pass subtitle coherence workflow that rewrites context-applied English using whole-video and time-range context, then shows before/after changed lines for review.

## Constraints
- Preserve: existing ASR, direct English translation, context-applied English, queue worker, saved context ranges.
- Do not change: internal output keys `direct` and `easy`.
- Allowed scope: prompt builder, service workflow, CLI/web endpoint, web UI, tests, docs.
- Not allowed: frontend framework rewrite, new dependency, destructive overwrite without review.

## Verification Surface
| Check | Command/artifact | Pass condition | Required? |
|---|---|---|---|
| Focused tests | `.venv311\Scripts\python.exe -m pytest -q tests/test_service.py::test_coherence_pass_updates_adapted_lines_and_records_before_after tests/test_web_ui.py` | pass | yes |
| Full suite | `.venv311\Scripts\python.exe -m pytest -q` | pass | yes |
| Browser desktop | Playwright screenshot/verifier | second-pass controls and before/after review visible without overflow | yes |
| Git | `git status --short --branch` | clean after push | yes |

## Stages
| Stage | Status | Objective | Depends on | Verification | Attempts | Notes |
|---|---|---|---|---|---|---|
| 1 | passed | Audit existing rebuild and context flow | none | code inspection | 1/3 | existing rebuild already saves notes and uses surrounding context |
| 2 | passed | Add second-pass backend | 1 | focused service test | 1/3 | writes `coherence-review.json` before/after rows |
| 3 | passed | Add web controls and review list | 2 | static UI test | 1/3 | includes restore-before button |
| 4 | passed | Full verification, docs, commit, push | 3 | full suite, browser, git | 1/3 | full suite and browser verifier passed |

## Subagent Plan
| Subagent | Purpose | Scope | Return format | Status |
|---|---|---|---|---|
| none | Single repo, sequential backend/UI edits | n/a | n/a | skipped |

## Execution Log
- Added `build_coherence_pass_prompt` for second-pass subtitle coherence with previous final lines, target lines, next context, overall context, and time-range context.
- Added service workflow that rewrites context-applied English, exports updated SRT, and records changed rows in `coherence-review.json`.
- Added CLI command `coherence-pass` and web endpoint `/api/job/coherence-pass`.
- Added web button `Run second-pass coherence review` and `Second-pass changes` before/after review list with per-line `Restore before`.
- Focused tests passed.
- Full suite passed.
- Browser verifier passed for second-pass controls and layout.

## Blockers
| Stage | Blocker | Evidence | Tried | What would unblock |
|---|---|---|---|---|

## Definition of Done
- User can run optional second-pass coherence review from the web UI.
- Review pass uses saved overall context and time-range context.
- Changed context-applied English lines are visible as before/after rows.
- User can restore one changed line to the previous text.
- Tests and browser verification pass.
