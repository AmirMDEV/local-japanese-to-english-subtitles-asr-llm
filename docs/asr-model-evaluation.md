# ASR Model Evaluation

This app should judge Japanese transcription models mainly by Japanese CER, then by subtitle timing support, then by speed on local hardware.

WER is useful for English-like word-separated languages, but Japanese word segmentation makes WER noisy. CER is the better headline number for Japanese subtitles.

## Current Decision

Keep Kotoba as the stable default because it already works with the existing timestamped subtitle pipeline.

Add ReazonSpeech k2 as the first experimental alternative because it has stronger reported Japanese CER, is Japanese-specific, uses ONNX/K2, and can run on CPU. It is opt-in until local subtitle timing and long-video behavior are verified on the target machine.

## Local 20 Second Timing Check

Sample: `DANDY-386.mp4`, audio from `00:01:21` to `00:01:41`, extracted as 16 kHz mono WAV.

Machine state: Windows laptop, NVIDIA GeForce RTX 3070 Laptop GPU, 8 GB VRAM, NVIDIA driver 591.86, CUDA 13.1 runtime. GPU already had about 3.3 GB in use from desktop apps before the run.

Environment: scratch Python 3.12 virtual environment at `scratch/asr-bench-venv`, CUDA Torch `2.11.0+cu130`.

| Model | Runtime path | Device | Cold setup/load | Cached model load | 20s inference | Total cached run | Output notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Kotoba-Whisper v2.2 | `transformers` ASR pipeline, `AutoModelForSpeechSeq2Seq`, `torch.float16` | `cuda:0` | 122.58s first load/download | 5.56s | 2.78s | 8.34s | Fastest inference. Good punctuation and chunk timestamps. Text started: `ちょっとこの面接の風景を...大丈夫ですか?` |
| Qwen3-ASR 1.7B raw | official `qwen-asr` package, transformers backend, `torch.bfloat16` | `cuda:0` | 349.34s first load/download | 8.09s | 7.95s | 16.05s | Fit in 8 GB VRAM for this short sample. No timestamps without the extra forced aligner. Text had broader context but added possible hallucinated tail text after the 20s sample. |
| ReazonSpeech k2 v2 initial | `reazonspeech.k2.asr`, ONNX/K2 | CPU | one-time model download already completed in previous smoke test | included in run | 9.20s cached smoke run | 9.20s | Good first sentence text, no punctuation, but missed later speech in the same 20s sample. Needs shorter chunks for subtitle use. |

Timing read: Kotoba v2.2 is fastest for this exact 20 second sample on this machine. Qwen3-ASR 1.7B is slower and needs the forced aligner model before it can produce subtitle timing, but it did fit on the GPU. Qwen should stay a research candidate until timestamp integration and hallucination checks pass on multiple samples.

## Fair Subtitle Timing Check

Same 20 second sample, but each model is configured closer to how it would be used for subtitle generation.

| Model | Subtitle setup | Device | Cached load | 20s inference | Total cached run | Timestamp output | Accuracy notes |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| Kotoba-Whisper v2.2 | `transformers` ASR pipeline with Whisper timestamps | `cuda:0` | 5.56s | 2.78s | 8.34s | 5 chunks | Fastest. Good punctuation. Missed filler `あの` near the start compared with the previous Kotoba v2.1 reference. |
| Qwen3-ASR 1.7B + Qwen3-ForcedAligner 0.6B | official `qwen-asr`, `return_time_stamps=True` | `cuda:0` | 13.39s | 6.99s | 20.38s | 51 aligned items | Fit in 8 GB VRAM, but used about 7.2 GB total GPU memory on this desktop state. Good first sentence, added extra tail text through `加奈さんかわいい。` inside the 20s sample. |
| Qwen3-ASR 0.6B + Qwen3-ForcedAligner 0.6B | official `qwen-asr`, `return_time_stamps=True` | `cuda:0` | 10.45s | 8.81s | 19.26s | 48 aligned items | Smaller ASR was not faster on this sample. More wording errors: `始めます`, `神楽川`, and extra `かわいいですね`. |
| ReazonSpeech k2 v2 | app wrapper, ONNX/K2, 6s chunks, 1s overlap, app cue merge | CPU | 5.19s | 7.14s | 12.33s | 4 merged cues | Captured more speech than the first long-chunk smoke, but still dropped or truncated some phrases. Better than 25s chunks, not reliable enough yet. |

Fair-test read: Kotoba v2.2 remains the best current production choice for speed plus subtitle timing. Qwen 1.7B with forced aligner is usable and properly timestamped, but slower and heavy on VRAM. Qwen 0.6B is not clearly useful here. ReazonSpeech k2 needs VAD-grade segmentation before it can compete as a subtitle engine.

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
