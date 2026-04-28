from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from local_subtitle_stack.config import (
    AppConfig,
    CachePaths,
    ModelConfig,
    ToolPaths,
    default_profiles,
    detect_subtitle_edit,
    detect_tool,
)
from local_subtitle_stack.integrations import FFmpegClient, OllamaClient, SubtitleEditClient
from local_subtitle_stack.pipeline import parse_srt
from local_subtitle_stack.queue import QueueStore
from local_subtitle_stack.service import WorkerService
from local_subtitle_stack.utils import no_window_creationflags, subtitle_output_dir


DEFAULT_SOURCE = Path(r"T:\Microsoft Softworks\General\Japanese\DANDY-386.mp4")
DEFAULT_ROOT = Path("scratch") / "dandy-model-benchmark"
SAMPLES = (
    ("early", 600, 90),
    ("middle", 5400, 90),
    ("late", 11400, 90),
)
ASR_CANDIDATES = (
    ("kotoba-v2.1", "kotoba", "kotoba-tech/kotoba-whisper-v2.1", "auto"),
    ("reazonspeech-k2-v2", "reazonspeech-k2", "reazon-research/reazonspeech-k2-v2", "auto"),
    ("kotoba-v1.1", "kotoba", "kotoba-tech/kotoba-whisper-v1.1", "auto"),
    ("faster-whisper-balanced", "faster-whisper", "kotoba-tech/kotoba-whisper-v2.1", "balanced"),
)
TRANSLATION_CANDIDATES = (
    ("plamo", "mitmul/plamo-2-translate:latest"),
    ("qwen3", "qwen3:4b-q8_0"),
)


def run(args: list[str]) -> None:
    subprocess.run(args, check=True, creationflags=no_window_creationflags())


def extract_samples(source: Path, samples_dir: Path, ffmpeg_path: str) -> list[Path]:
    samples_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for label, start, duration in SAMPLES:
        target = samples_dir / f"{source.stem}.{label}.{start}s.{duration}s.mp4"
        if not target.exists() or target.stat().st_size < 1024:
            run(
                [
                    ffmpeg_path,
                    "-y",
                    "-ss",
                    str(start),
                    "-t",
                    str(duration),
                    "-i",
                    str(source),
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-sn",
                    "-dn",
                    "-ac",
                    "2",
                    "-ar",
                    "48000",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "160k",
                    str(target),
                ]
            )
        paths.append(target)
    return paths


def build_service(root: Path, models: ModelConfig) -> WorkerService:
    tools = ToolPaths(
        ffmpeg=detect_tool("ffmpeg"),
        ffprobe=detect_tool("ffprobe"),
        ollama=detect_tool("ollama"),
        subtitle_edit=detect_subtitle_edit(),
        python311="",
    )
    config = AppConfig(
        config_path=str(root / "config.toml"),
        queue_root=str(root / "queue"),
        tools=tools,
        cache_paths=CachePaths(hf_hub_cache=str(root / "model-cache")),
        models=models,
        profiles=default_profiles(),
    )
    return WorkerService(
        config=config,
        store=QueueStore(config),
        ffmpeg=FFmpegClient(config.tools.ffmpeg, config.tools.ffprobe),
        subtitle_edit=SubtitleEditClient(config.tools.subtitle_edit),
        ollama=OllamaClient(executable_path=config.tools.ollama),
    )


def srt_metrics(path: Path, duration_seconds: float) -> dict[str, object]:
    cues = parse_srt(path) if path.exists() else []
    if not cues:
        return {
            "exists": path.exists(),
            "cue_count": 0,
            "last_end": 0.0,
            "coverage_percent": 0.0,
            "chars": 0,
            "sample": "",
        }
    gaps = [
        max(cues[index].start - cues[index - 1].end, 0.0)
        for index in range(1, len(cues))
    ]
    sample_text = " / ".join(cue.text.replace("\n", " ") for cue in cues[:5])
    return {
        "exists": True,
        "cue_count": len(cues),
        "first_start": cues[0].start,
        "last_end": cues[-1].end,
        "coverage_percent": round((cues[-1].end / duration_seconds) * 100, 2),
        "max_gap_seconds": round(max(gaps or [0.0]), 2),
        "chars": sum(len(cue.text) for cue in cues),
        "sample": sample_text[:500],
    }


def run_candidate(
    *,
    root: Path,
    sample_paths: list[Path],
    asr_label: str,
    asr_engine: str,
    asr_model: str,
    faster_profile: str,
    translation_label: str,
    translation_model: str,
) -> dict[str, object]:
    run_root = root / "runs" / f"{asr_label}__{translation_label}"
    if run_root.exists():
        shutil.rmtree(run_root)
    models = ModelConfig(
        asr_engine=asr_engine,
        asr=asr_model,
        faster_whisper_profile=faster_profile,
        literal_translation=translation_model,
        adapted_translation="qwen3:4b-q8_0",
    )
    service = build_service(run_root, models)
    start = time.monotonic()
    errors: list[str] = []
    job_results: list[dict[str, object]] = []
    for sample in sample_paths:
        try:
            manifest = service.enqueue(
                sample,
                profile="conservative",
                include_adapted_english=False,
                prefer_fast_translation=False,
            )
        except Exception as exc:
            errors.append(f"{sample.name}: enqueue failed: {type(exc).__name__}: {exc}")
            continue
    try:
        service.run_until_empty()
    except Exception as exc:
        errors.append(f"worker failed: {type(exc).__name__}: {exc}")

    for _job_dir, manifest, state in service.store.list_jobs():
        export_dir = Path(manifest.export_dir or subtitle_output_dir(Path(manifest.source_path)))
        job_results.append(
            {
                "source": manifest.source_name,
                "state": state,
                "status": manifest.status,
                "error": manifest.error,
                "asr_details": manifest.checkpoint("transcribe").details,
                "japanese": srt_metrics(export_dir / manifest.artifacts["ja_srt"], 90.0),
                "english": srt_metrics(export_dir / manifest.artifacts["literal_srt"], 90.0),
            }
        )

    elapsed = time.monotonic() - start
    return {
        "asr_label": asr_label,
        "translation_label": translation_label,
        "translation_model": translation_model,
        "elapsed_seconds": round(elapsed, 2),
        "errors": errors,
        "jobs": job_results,
    }


def write_report(root: Path, report: dict[str, object]) -> None:
    report_path = root / "benchmark-report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# DANDY-386 Model Benchmark", ""]
    lines.append(f"Source: `{report['source']}`")
    lines.append("")
    for result in report["results"]:  # type: ignore[index]
        lines.append(
            f"## {result['asr_label']} + {result['translation_label']} "
            f"({result['elapsed_seconds']}s)"
        )
        if result["errors"]:
            lines.append("Errors:")
            for error in result["errors"]:
                lines.append(f"- {error}")
        for job in result["jobs"]:
            ja = job["japanese"]
            en = job["english"]
            lines.append(
                f"- {job['source']}: {job['status']} | "
                f"JA cues {ja['cue_count']} coverage {ja['coverage_percent']}% | "
                f"EN cues {en['cue_count']} coverage {en['coverage_percent']}%"
            )
            if ja["sample"]:
                lines.append(f"  - JA sample: {ja['sample']}")
            if en["sample"]:
                lines.append(f"  - EN sample: {en['sample']}")
        lines.append("")
    (root / "benchmark-report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--quick", action="store_true", help="Run only early sample and two strongest candidates.")
    parser.add_argument(
        "--translations",
        default="plamo,qwen3",
        help="Comma-separated translation candidate labels: plamo,qwen3",
    )
    parser.add_argument(
        "--asr",
        default="kotoba-v2.1,kotoba-v1.1,faster-whisper-balanced",
        help="Comma-separated ASR candidate labels.",
    )
    args = parser.parse_args()

    source = Path(args.source)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    ffmpeg = detect_tool("ffmpeg") or "ffmpeg"
    sample_paths = extract_samples(source, root / "samples", ffmpeg)
    if args.quick:
        sample_paths = sample_paths[:1]
        quick_asr_labels = set(args.asr.split(","))
        asr_candidates = [item for item in ASR_CANDIDATES if item[0] in quick_asr_labels]
        if not asr_candidates:
            asr_candidates = ASR_CANDIDATES[:2]
        translation_labels = set(args.translations.split(","))
        translation_candidates = [item for item in TRANSLATION_CANDIDATES if item[0] in translation_labels]
        if not translation_candidates:
            translation_candidates = TRANSLATION_CANDIDATES
    else:
        asr_labels = set(args.asr.split(","))
        translation_labels = set(args.translations.split(","))
        asr_candidates = [item for item in ASR_CANDIDATES if item[0] in asr_labels]
        translation_candidates = [item for item in TRANSLATION_CANDIDATES if item[0] in translation_labels]

    results: list[dict[str, object]] = []
    for asr_label, asr_engine, asr_model, faster_profile in asr_candidates:
        for translation_label, translation_model in translation_candidates:
            results.append(
                run_candidate(
                    root=root,
                    sample_paths=sample_paths,
                    asr_label=asr_label,
                    asr_engine=asr_engine,
                    asr_model=asr_model,
                    faster_profile=faster_profile,
                    translation_label=translation_label,
                    translation_model=translation_model,
                )
            )

    report = {
        "source": str(source),
        "samples": [str(path) for path in sample_paths],
        "sample_plan": [asdict_sample for asdict_sample in SAMPLES],
        "results": results,
    }
    write_report(root, report)
    print(root / "benchmark-report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
