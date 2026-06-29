# Amir Goal Plan

## Objective
Make web UI time-range context clear, multi-range, editable, and tied to subtitle selection.

## Constraints
- Preserve: existing translation pipeline, queue worker, saved scene-context prompt injection.
- Do not change: internal output keys `direct` and `easy`.
- Allowed scope: web UI, tests, changelog, solved-problem note.
- Not allowed: backend rewrite or new frontend dependency.

## Verification Surface
| Check | Command/artifact | Pass condition | Required? |
|---|---|---|---|
| Unit suite | `.venv311\Scripts\python.exe -m pytest` | all tests pass | yes |
| Browser desktop | Playwright screenshot | no overlap, selection auto-fills range, note card appears | yes |
| Browser mobile | Playwright screenshot | no horizontal overflow, controls readable | yes |
| Git | `git status --short --branch` | clean after push | yes |

## Stages
| Stage | Status | Objective | Depends on | Verification | Attempts | Notes |
|---|---|---|---|---|---|---|
| 1 | passed | Audit current UI and prompt flow | none | code inspection | 1/3 | backend already supports range rebuild |
| 2 | passed | Patch selection and context UI | 1 | tests + browser | 1/3 | no dependency added |
| 3 | passed | Verify live preview | 2 | screenshots | 1/3 | port 8876 |
| 4 | active | Docs, commit, push | 3 | git clean | 1/3 | update solved note |

## Subagent Plan
| Subagent | Purpose | Scope | Return format | Status |
|---|---|---|---|---|
| none | Main path is single-file sequential patch | n/a | n/a | skipped |

## Execution Log
- Audited `web_ui.py`: line preview, selected rows, notes, rebuild endpoint already wired.
- Implemented split overall/time-range context panels, Shift range selection, auto-filled range times, editable range cards, clearer job-stage labels, and `Pick context-applied`.
- Verified `92 passed in 11.20s`; Playwright desktop/mobile time-range context verifier passed.

## Blockers
| Stage | Blocker | Evidence | Tried | What would unblock |
|---|---|---|---|---|

## Definition of Done
- Overall context and time-range context are separate sections.
- Selecting subtitle rows auto-fills start/end.
- Shift-click selects a continuous subtitle range.
- Multiple editable context cards can be saved.
- Retranslate controls clearly state whole job vs selected time range.
- Tests and browser screenshots pass.
