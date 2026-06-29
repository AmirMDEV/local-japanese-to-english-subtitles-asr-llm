from __future__ import annotations

from local_subtitle_stack.asr_models import (
    QWEN3_ASR_0_6B_MODEL_ID,
    QWEN3_ASR_1_7B_MODEL_ID,
    QWEN3_ASR_ENGINE,
    REAZON_K2_ENGINE,
    candidate_for_engine,
    ranked_asr_candidates,
)


def test_asr_candidate_registry_prioritizes_japanese_cer() -> None:
    candidates = ranked_asr_candidates()

    assert candidates[0].key == "reazonspeech-k2-v2"
    assert candidates[0].engine == REAZON_K2_ENGINE
    assert any(metric.metric == "CER" and metric.value == 6.45 for metric in candidates[0].metrics)


def test_candidate_for_engine_matches_reazon_k2() -> None:
    candidate = candidate_for_engine(REAZON_K2_ENGINE)

    assert candidate is not None
    assert candidate.model_id == "reazon-research/reazonspeech-k2-v2"


def test_qwen3_asr_candidates_include_0_6b_and_1_7b() -> None:
    candidates = ranked_asr_candidates()
    qwen_models = {candidate.model_id for candidate in candidates if candidate.engine == QWEN3_ASR_ENGINE}

    assert QWEN3_ASR_0_6B_MODEL_ID in qwen_models
    assert QWEN3_ASR_1_7B_MODEL_ID in qwen_models
