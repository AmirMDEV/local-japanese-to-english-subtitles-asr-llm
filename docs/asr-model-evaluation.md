# ASR Model Evaluation

This app should judge Japanese transcription models mainly by Japanese CER, then by subtitle timing support, then by speed on local hardware.

WER is useful for English-like word-separated languages, but Japanese word segmentation makes WER noisy. CER is the better headline number for Japanese subtitles.

## Current Decision

Keep Kotoba as the stable default because it already works with the existing timestamped subtitle pipeline.

Add ReazonSpeech k2 as the first experimental alternative because it has stronger reported Japanese CER, is Japanese-specific, uses ONNX/K2, and can run on CPU. It is opt-in until local subtitle timing and long-video behavior are verified on the target machine.

## Ranked Candidates

| Rank | Model | App status | Reported evidence | Why it matters |
| --- | --- | --- | --- | --- |
| 1 | ReazonSpeech k2 v2 | Experimental engine | CER 6.45 JSUT, 7.85 Common Voice v8 Japanese, 9.09 TEDxJP-10K | Best Japanese-specific reported CER found; CPU-capable ONNX runtime; chunks must stay around 30 seconds. |
| 2 | Kotoba-Whisper v2.x | Stable default | v2.0 CER 9.2 Common Voice v8 Japanese, 8.4 JSUT, 11.6 ReazonSpeech held-out | Already fits the app's chunked timestamp pipeline; v2.2 adds punctuation/diarization stack but extra dependencies. |
| 3 | Qwen3-ASR 1.7B | Research candidate | Open ASR mean WER 5.76; multilingual FLEURS average WER 4.90 includes Japanese | Very strong broad ASR model; needs Qwen dependency stack and forced aligner integration before subtitle use. |
| 4 | wav2vec2 XLS-R Japanese | Research candidate | Self-reported Common Voice v8 Japanese CER 3.35 and WER 7.88 with 4-gram LM | Strong transcript score, but CTC needs reliable alignment before it can produce high-quality subtitles. |
| 5 | Microsoft VibeVoice ASR | Research candidate | Open ASR mean WER 7.77 | Good long-form model, but no Japanese-only published score found in primary sources. |
| 6 | NVIDIA Canary/Parakeet family | Not selected yet | Canary docs list English, German, French, Spanish; NIM supports several ASR families | Strong global ASR family, but no clear Japanese-first production path found for this app yet. |
| 7 | Meta SeamlessM4T/MMS | Not selected yet | SeamlessM4T-v2 reports 18.5 WER average over 77 FLEURS languages | Translation-oriented stack; not the strongest local Japanese subtitle target. |

## Sources Checked

- ReazonSpeech v2.1 announcement: https://research.reazon.jp/blog/2024-08-01-ReazonSpeech.html
- ReazonSpeech k2 v2 model card: https://huggingface.co/reazon-research/reazonspeech-k2-v2
- ReazonSpeech NeMo v2 model card: https://huggingface.co/reazon-research/reazonspeech-nemo-v2
- Kotoba-Whisper v2.0 model card: https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0
- Kotoba-Whisper v2.2 model card: https://huggingface.co/kotoba-tech/kotoba-whisper-v2.2
- Qwen3-ASR 1.7B model card: https://huggingface.co/Qwen/Qwen3-ASR-1.7B
- Qwen3-ASR technical report: https://arxiv.org/abs/2601.21337
- wav2vec2 XLS-R Japanese model card: https://huggingface.co/vumichien/wav2vec2-xls-r-1b-japanese
- Microsoft VibeVoice ASR HF model card: https://huggingface.co/microsoft/VibeVoice-ASR-HF
- NVIDIA NeMo ASR model docs: https://docs.nvidia.com/nemo-framework/user-guide/25.02/nemotoolkit/asr/models.html
- SeamlessM4T Nature paper: https://www.nature.com/articles/s41586-024-08359-z
