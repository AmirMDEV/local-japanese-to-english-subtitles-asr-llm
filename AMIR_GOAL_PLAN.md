# Amir Goal Plan

## Objective
Make Qwen3-ASR 1.7B complete the existing 90s sample with valid timings, and fix subtitle preview row overflow.

## Constraints
- Preserve: existing translation pipeline, queue worker, saved scene-context prompt injection.
- Do not change: internal output keys `direct` and `easy`.
- Allowed scope: Qwen ASR routing, preview layout CSS/rendering, tests, changelog, solved-problem note.
- Not allowed: backend rewrite, frontend framework migration, new dependency.

## Verification Surface
| Check | Command/artifact | Pass condition | Required? |
|---|---|---|---|
| Qwen live proof | `.venv311\Scripts\python.exe .codex-temp\run_qwen17_live_test.py` | 1.7B output reaches tail, cues monotonic, no bad cues | yes |
| Unit suite | `.venv311\Scripts\python.exe -m pytest` | all tests pass | yes |
| Browser desktop | Playwright screenshot/verifier | preview rows and selected row have no overlap or child overflow | yes |
| Git | `git status --short --branch` | clean after push | yes |

## Stages
| Stage | Status | Objective | Depends on | Verification | Attempts | Notes |
|---|---|---|---|---|---|---|
| 1 | passed | Run Qwen3-ASR 1.7B live sample proof | none | live verifier | 1/3 | first proof cut off at 66.863s |
| 2 | passed | Patch Qwen tail coverage | 1 | live verifier | 1/3 | 30-second Qwen chunks plus 4096 generation headroom |
| 3 | passed | Patch preview row overflow | none | Playwright layout verifier | 1/3 | row children stay inside row boxes |
| 4 | active | Docs, commit, push | 2,3 | git clean | 1/3 | update solved note |

## Subagent Plan
| Subagent | Purpose | Scope | Return format | Status |
|---|---|---|---|---|
| none | Main path is single-file sequential patch | n/a | n/a | skipped |

## Execution Log
- Live Qwen3-ASR 1.7B proof initially produced valid monotonic cues but stopped at 66.863s on the 90s DANDY sample.
- Added 30-second Qwen chunks and increased Qwen generation headroom to 4096.
- Live Qwen3-ASR 1.7B proof passed after patch: 4 chunks, 130 cues, last cue 91.78s, no bad cues, monotonic.
- Rebuilt preview row rendering/CSS so time, Japanese, and English columns wrap inside row boxes; Playwright verifier found 13 visible rows, no row overlap, no child overflow.
- Full test suite passed.

## Blockers
| Stage | Blocker | Evidence | Tried | What would unblock |
|---|---|---|---|---|

## Definition of Done
- Qwen3-ASR 1.7B transcribes the 90s already-transcribed sample through the clip tail with valid timings.
- Web subtitle preview rows wrap without row overlap, text overflow, or selected-row overflow.
- Tests and browser layout verification pass.
