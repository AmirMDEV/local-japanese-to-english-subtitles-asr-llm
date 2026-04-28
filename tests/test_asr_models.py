from __future__ import annotations

from local_subtitle_stack.asr_models import REAZON_K2_ENGINE, candidate_for_engine, ranked_asr_candidates


def test_asr_candidate_registry_prioritizes_japanese_cer() -> None:
    candidates = ranked_asr_candidates()

    assert candidates[0].key == "reazonspeech-k2-v2"
    assert candidates[0].engine == REAZON_K2_ENGINE
    assert any(metric.metric == "CER" and metric.value == 6.45 for metric in candidates[0].metrics)


def test_candidate_for_engine_matches_reazon_k2() -> None:
    candidate = candidate_for_engine(REAZON_K2_ENGINE)

    assert candidate is not None
    assert candidate.model_id == "reazon-research/reazonspeech-k2-v2"
