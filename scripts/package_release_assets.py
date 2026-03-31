from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


MAX_SOURCE_PART_BYTES = 600 * 1024 * 1024


def add_files(zip_path: Path, base_dir: Path, relative_paths: list[Path]) -> None:
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED, compresslevel=1) as archive:
        for relative_path in relative_paths:
            source_path = base_dir / relative_path
            archive.write(source_path, Path(base_dir.name) / relative_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-root", required=True)
    parser.add_argument("--dist-name", default="SubtitleTool")
    args = parser.parse_args()

    dist_root = Path(args.dist_root).resolve()
    if not dist_root.exists():
        raise SystemExit(f"Distribution folder not found: {dist_root}")

    output_dir = dist_root.parent
    dist_name = args.dist_name

    all_files = [path for path in dist_root.rglob("*") if path.is_file()]
    relative_files = [path.relative_to(dist_root) for path in all_files]

    app_files: list[Path] = []
    torch_lib_files: list[Path] = []
    torch_extra_files: list[Path] = []

    for relative_path in relative_files:
        relative_text = relative_path.as_posix()
        if relative_text.startswith("_internal/torch/lib/"):
            torch_lib_files.append(relative_path)
        elif relative_text.startswith("_internal/torch/") or relative_text.startswith("_internal/torch-"):
            torch_extra_files.append(relative_path)
        else:
            app_files.append(relative_path)

    torch_lib_parts: list[list[Path]] = [[]]
    running_size = 0
    for relative_path in sorted(torch_lib_files):
        source_size = (dist_root / relative_path).stat().st_size
        current_part = torch_lib_parts[-1]
        if current_part and running_size + source_size > MAX_SOURCE_PART_BYTES:
            torch_lib_parts.append([])
            current_part = torch_lib_parts[-1]
            running_size = 0
        current_part.append(relative_path)
        running_size += source_size

    if torch_lib_parts and not torch_lib_parts[-1]:
        torch_lib_parts.pop()

    assets = [(output_dir / f"{dist_name}-windows-x64-app.zip", sorted(app_files))]
    for index, part_files in enumerate(torch_lib_parts, start=1):
        assets.append((output_dir / f"{dist_name}-windows-x64-torch-lib-part{index}.zip", part_files))
    assets.append((output_dir / f"{dist_name}-windows-x64-torch-extra.zip", sorted(torch_extra_files)))

    for zip_path, files in assets:
        if zip_path.exists():
            zip_path.unlink()
        add_files(zip_path, dist_root, files)
        print(f"Created {zip_path} ({zip_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
