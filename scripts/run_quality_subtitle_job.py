from __future__ import annotations

import argparse
import shutil
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
from local_subtitle_stack.queue import QueueStore
from local_subtitle_stack.service import WorkerService


DEFAULT_SOURCE = Path(r"T:\Microsoft Softworks\General\Japanese\DANDY-386.mp4")
DEFAULT_QUEUE_ROOT = Path("scratch") / "dandy-quality-run" / "queue"
DEFAULT_MODEL_CACHE = (
    Path("scratch")
    / "dandy-model-benchmark"
    / "runs"
    / "kotoba-v2.1__qwen3"
    / "model-cache"
)


def build_service(args: argparse.Namespace) -> WorkerService:
    tools = ToolPaths(
        ffmpeg=detect_tool("ffmpeg"),
        ffprobe=detect_tool("ffprobe"),
        ollama=detect_tool("ollama"),
        subtitle_edit=detect_subtitle_edit(),
        python311="",
    )
    config = AppConfig(
        config_path=str(Path(args.queue_root).parent / "config.toml"),
        queue_root=str(Path(args.queue_root)),
        tools=tools,
        cache_paths=CachePaths(hf_hub_cache=str(Path(args.cache_dir)) if args.cache_dir else ""),
        models=ModelConfig(
            asr_engine=args.asr_engine,
            asr=args.asr_model,
            faster_whisper_profile=args.faster_whisper_profile,
            literal_translation=args.literal_model,
            adapted_translation=args.adapted_model,
        ),
        profiles=default_profiles(),
    )
    return WorkerService(
        config=config,
        store=QueueStore(config),
        ffmpeg=FFmpegClient(config.tools.ffmpeg, config.tools.ffprobe),
        subtitle_edit=SubtitleEditClient(config.tools.subtitle_edit),
        ollama=OllamaClient(executable_path=config.tools.ollama),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--queue-root", default=str(DEFAULT_QUEUE_ROOT))
    parser.add_argument("--cache-dir", default=str(DEFAULT_MODEL_CACHE))
    parser.add_argument("--profile", default="conservative")
    parser.add_argument("--asr-engine", default="kotoba", choices=["kotoba", "faster-whisper"])
    parser.add_argument("--asr-model", default="kotoba-tech/kotoba-whisper-v2.1")
    parser.add_argument(
        "--faster-whisper-profile",
        default="balanced",
        choices=["auto", "high", "balanced", "low_gpu", "cpu_fallback"],
    )
    parser.add_argument("--literal-model", default="qwen3:4b-q8_0")
    parser.add_argument("--adapted-model", default="qwen3:4b-q8_0")
    parser.add_argument("--no-adapted", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--resume-only", action="store_true")
    args = parser.parse_args()

    queue_root = Path(args.queue_root)
    if args.fresh and queue_root.exists():
        resolved = queue_root.resolve()
        if "scratch" not in resolved.parts:
            raise RuntimeError(f"Refusing to clear non-scratch queue root: {resolved}")
        shutil.rmtree(queue_root)

    service = build_service(args)
    if args.resume_only:
        service.run_until_empty()
        for _job_dir, manifest, _state in service.store.list_jobs():
            print(f"{manifest.job_id}: {manifest.status} -> {manifest.export_dir}")
            if manifest.error:
                print(manifest.error)
        return 0

    source = Path(args.source)
    manifest = service.enqueue(
        source,
        profile=args.profile,
        include_adapted_english=not args.no_adapted,
        prefer_fast_translation=False,
    )
    print(f"Queued {manifest.job_id}")
    print(f"Output folder: {manifest.export_dir}")
    service.run_until_empty()
    _job_dir, completed = service.store.find_job(manifest.job_id)
    print(f"Finished {completed.job_id}: {completed.status}")
    print(f"Output folder: {completed.export_dir}")
    if completed.error:
        print(completed.error)
    return 0 if completed.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
