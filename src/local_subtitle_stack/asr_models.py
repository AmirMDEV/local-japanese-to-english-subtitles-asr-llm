from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ASRMetric:
    dataset: str
    metric: str
    value: float
    note: str = ""


@dataclass(frozen=True, slots=True)
class ASRCandidate:
    key: str
    label: str
    engine: str
    model_id: str
    status: str
    summary: str
    metrics: tuple[ASRMetric, ...]
    sources: tuple[str, ...]


REAZON_K2_ENGINE = "reazonspeech-k2"
REAZON_K2_MODEL_ID = "reazon-research/reazonspeech-k2-v2"
QWEN3_ASR_ENGINE = "qwen3-asr"
QWEN3_ASR_0_6B_MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
QWEN3_ASR_1_7B_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
QWEN3_FORCED_ALIGNER_MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B"


ASR_CANDIDATES: tuple[ASRCandidate, ...] = (
    ASRCandidate(
        key="reazonspeech-k2-v2",
        label="ReazonSpeech k2 Japanese ASR",
        engine=REAZON_K2_ENGINE,
        model_id=REAZON_K2_MODEL_ID,
        status="experimental",
        summary=(
            "Japanese-only ONNX/K2 model with strong reported Japanese CER and CPU-capable "
            "runtime. Best first experimental alternative to Kotoba for local subtitles."
        ),
        metrics=(
            ASRMetric("JSUT Basic5000", "CER", 6.45),
            ASRMetric("Common Voice v8 Japanese", "CER", 7.85),
            ASRMetric("TEDxJP-10K", "CER", 9.09),
        ),
        sources=(
            "https://research.reazon.jp/blog/2024-08-01-ReazonSpeech.html",
            "https://huggingface.co/reazon-research/reazonspeech-k2-v2",
        ),
    ),
    ASRCandidate(
        key="kotoba-whisper-v2",
        label="Kotoba-Whisper Japanese quality",
        engine="kotoba",
        model_id="kotoba-tech/kotoba-whisper-v2.0",
        status="stable",
        summary=(
            "Whisper-compatible Japanese model trained on ReazonSpeech. Good production "
            "default because timestamps fit the existing subtitle pipeline."
        ),
        metrics=(
            ASRMetric("Common Voice v8 Japanese", "CER", 9.2),
            ASRMetric("JSUT Basic5000", "CER", 8.4),
            ASRMetric("ReazonSpeech held-out", "CER", 11.6),
            ASRMetric("Common Voice v8 Japanese", "WER", 58.8, "Japanese WER is segmentation-sensitive."),
            ASRMetric("JSUT Basic5000", "WER", 63.7, "Japanese WER is segmentation-sensitive."),
            ASRMetric("ReazonSpeech held-out", "WER", 55.6, "Japanese WER is segmentation-sensitive."),
        ),
        sources=(
            "https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0",
            "https://huggingface.co/kotoba-tech/kotoba-whisper-v2.2",
        ),
    ),
    ASRCandidate(
        key="qwen3-asr-0.6b",
        label="Qwen3-ASR 0.6B with forced aligner",
        engine=QWEN3_ASR_ENGINE,
        model_id=QWEN3_ASR_0_6B_MODEL_ID,
        status="research",
        summary=(
            "Smaller Qwen3-ASR model with Japanese support and Qwen forced aligner timestamps. "
            "Good faster Qwen option before trying the 1.7B model."
        ),
        metrics=(
            ASRMetric("Open ASR leaderboard mean", "WER", 8.86),
            ASRMetric("Multilingual FLEURS 12-language average", "WER", 7.57, "Includes Japanese but is not Japanese-only."),
            ASRMetric("Multilingual Common Voice 13-language average", "WER", 12.75, "Includes Japanese but is not Japanese-only."),
        ),
        sources=(
            "https://huggingface.co/Qwen/Qwen3-ASR-0.6B",
            "https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B",
        ),
    ),
    ASRCandidate(
        key="qwen3-asr-1.7b",
        label="Qwen3-ASR 1.7B with forced aligner",
        engine=QWEN3_ASR_ENGINE,
        model_id=QWEN3_ASR_1_7B_MODEL_ID,
        status="research",
        summary=(
            "Strong open ASR leaderboard model with Japanese in its multilingual sets and "
            "Qwen forced aligner timestamps. Higher quality Qwen option, heavier than 0.6B."
        ),
        metrics=(
            ASRMetric("Open ASR leaderboard mean", "WER", 5.76),
            ASRMetric("Multilingual FLEURS 12-language average", "WER", 4.90, "Includes Japanese but is not Japanese-only."),
            ASRMetric("Multilingual Common Voice 13-language average", "WER", 9.18, "Includes Japanese but is not Japanese-only."),
        ),
        sources=(
            "https://huggingface.co/Qwen/Qwen3-ASR-1.7B",
            "https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B",
            "https://arxiv.org/abs/2601.21337",
        ),
    ),
    ASRCandidate(
        key="wav2vec2-xls-r-1b-japanese",
        label="wav2vec2 XLS-R Japanese",
        engine="ctc-experimental",
        model_id="vumichien/wav2vec2-xls-r-1b-japanese",
        status="research",
        summary=(
            "Very strong self-reported Common Voice Japanese CER with a language model, "
            "but CTC timing/alignment work is needed for high-quality subtitles."
        ),
        metrics=(
            ASRMetric("Common Voice v8 Japanese with 4-gram LM", "CER", 3.35),
            ASRMetric("Common Voice v8 Japanese with 4-gram LM", "WER", 7.88),
        ),
        sources=("https://huggingface.co/vumichien/wav2vec2-xls-r-1b-japanese",),
    ),
    ASRCandidate(
        key="vibevoice-asr",
        label="Microsoft VibeVoice ASR",
        engine="vibevoice-asr",
        model_id="microsoft/VibeVoice-ASR-HF",
        status="research",
        summary=(
            "Long-form ASR model with good Open ASR leaderboard WER, but no Japanese-only "
            "published score found in the checked primary sources."
        ),
        metrics=(ASRMetric("Open ASR leaderboard mean", "WER", 7.77),),
        sources=("https://huggingface.co/microsoft/VibeVoice-ASR-HF",),
    ),
)


def ranked_asr_candidates() -> tuple[ASRCandidate, ...]:
    return ASR_CANDIDATES


def candidate_for_engine(engine: str) -> ASRCandidate | None:
    normalized = engine.strip().lower()
    for candidate in ASR_CANDIDATES:
        if candidate.engine == normalized:
            return candidate
    return None
