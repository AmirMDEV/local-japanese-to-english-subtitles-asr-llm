from __future__ import annotations

import argparse
from pathlib import Path

from .app import build_service
from .queue import QueueError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="subtitle-tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    enqueue = subparsers.add_parser("enqueue", help="Queue one or more source videos.")
    enqueue.add_argument("sources", nargs="+")
    enqueue.add_argument("--profile", default="conservative")
    enqueue.add_argument("--glossary")
    enqueue.add_argument("--series")
    enqueue.add_argument("--context")
    enqueue.add_argument("--recursive", action="store_true")
    enqueue.add_argument("--no-easy-english", action="store_true")

    subparsers.add_parser("worker", help="Process queued jobs until the queue is empty.")
    subparsers.add_parser("status", help="Show queue status.")

    resume = subparsers.add_parser("resume", help="Resume a queued or failed job.")
    resume.add_argument("job_id")

    import_existing = subparsers.add_parser(
        "import-existing",
        help="Import an existing subtitle set for editing or reprocessing.",
    )
    import_existing.add_argument("--profile", default="conservative")
    import_existing.add_argument("--video")
    import_existing.add_argument("--primary-subtitle")
    import_existing.add_argument("--ja")
    import_existing.add_argument("--direct")
    import_existing.add_argument("--easy")
    import_existing.add_argument("--reference")
    import_existing.add_argument("--series")
    import_existing.add_argument("--context")
    import_existing.add_argument("--no-easy-english", action="store_true")

    rebuild = subparsers.add_parser("rebuild-english", help="Rebuild English subtitle outputs for one job.")
    rebuild.add_argument("job_id")

    review = subparsers.add_parser("open-review", help="Open completed subtitle outputs in Subtitle Edit.")
    review.add_argument("job_id", nargs="?")

    output = subparsers.add_parser("open-output", help="Open the exported subtitle folder in Explorer.")
    output.add_argument("job_id", nargs="?")

    subparsers.add_parser("pause", help="Pause the queue after the current safe checkpoint.")
    subparsers.add_parser("unpause", help="Clear the pause flag.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    service = build_service()

    try:
        if args.command == "enqueue":
            glossary = Path(args.glossary) if args.glossary else None
            for source in args.sources:
                source_path = Path(source)
                if source_path.is_dir():
                    manifests, skipped = service.enqueue_folder(
                        folder=source_path,
                        profile=args.profile,
                        glossary=glossary,
                        series=args.series,
                        context=args.context,
                        recursive=args.recursive,
                        include_adapted_english=not args.no_easy_english,
                    )
                    print(f"Queued {len(manifests)} videos from {source_path}")
                    if skipped:
                        print(f"Skipped {len(skipped)} already queued videos.")
                    for manifest in manifests:
                        print(f"Queued {manifest.source_name} as {manifest.job_id}")
                    continue
                manifest = service.enqueue(
                    source=source_path,
                    profile=args.profile,
                    glossary=glossary,
                    series=args.series,
                    context=args.context,
                    include_adapted_english=not args.no_easy_english,
                )
                print(f"Queued {manifest.source_name} as {manifest.job_id}")
            return 0

        if args.command == "worker":
            service.run_until_empty()
            print("Queue complete.")
            return 0

        if args.command == "status":
            rows = service.status_rows()
            if not rows:
                print("No jobs found.")
                return 0
            for row in rows:
                print(
                    f"{row['job_id']}  {row['status']:<9}  {row['stage']:<18}  "
                    f"{row['state_dir']:<7}  {row['source']}"
                )
            return 0

        if args.command == "resume":
            manifest = service.resume(args.job_id)
            print(f"Resumed {manifest.job_id}")
            return 0

        if args.command == "import-existing":
            manifest = service.import_existing(
                profile=args.profile,
                video=Path(args.video) if args.video else None,
                primary_subtitle=Path(args.primary_subtitle) if args.primary_subtitle else None,
                japanese=Path(args.ja) if args.ja else None,
                direct=Path(args.direct) if args.direct else None,
                easy=Path(args.easy) if args.easy else None,
                reference=Path(args.reference) if args.reference else None,
                series=args.series,
                context=args.context,
                include_adapted_english=not args.no_easy_english,
            )
            print(f"Imported existing subtitles into {manifest.job_id}")
            return 0

        if args.command == "rebuild-english":
            manifest = service.rebuild_english_from_saved_notes(args.job_id)
            print(f"Rebuilt English for {manifest.job_id}")
            return 0

        if args.command == "open-review":
            paths = service.open_review(args.job_id)
            for path in paths:
                print(path)
            return 0

        if args.command == "open-output":
            print(service.open_output_folder(args.job_id))
            return 0

        if args.command == "pause":
            service.store.set_pause(True)
            print("Pause requested.")
            return 0

        if args.command == "unpause":
            service.store.set_pause(False)
            print("Pause cleared.")
            return 0
    except QueueError as exc:
        print(str(exc))
        return 1

    raise QueueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
