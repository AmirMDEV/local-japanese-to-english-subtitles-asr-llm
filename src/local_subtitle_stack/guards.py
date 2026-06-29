from __future__ import annotations

import csv
import re
import subprocess
from dataclasses import dataclass

import psutil

from .utils import no_window_creationflags


class ResourceGuardError(RuntimeError):
    pass


@dataclass(slots=True)
class ResourceSnapshot:
    free_ram_mb: int
    process_rss_mb: int
    gpu_free_mb: int = 0
    gpu_total_mb: int = 0

    @property
    def gpu_used_mb(self) -> int:
        if self.gpu_total_mb <= 0:
            return 0
        return max(self.gpu_total_mb - self.gpu_free_mb, 0)


def _parse_nvidia_smi_table_memory(output: str) -> tuple[int, int]:
    match = re.search(r"(\d+)MiB\s*/\s*(\d+)MiB", output)
    if not match:
        return 0, 0
    used = int(match.group(1))
    total = int(match.group(2))
    return max(total - used, 0), total


def capture_snapshot() -> ResourceSnapshot:
    memory = psutil.virtual_memory()
    process = psutil.Process()
    gpu_free = 0
    gpu_total = 0

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            creationflags=no_window_creationflags(),
        )
        rows = list(csv.reader(line for line in completed.stdout.splitlines() if line.strip()))
        if rows:
            gpu_free = int(rows[0][0].strip())
            gpu_total = int(rows[0][1].strip())
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError, ValueError):
        try:
            completed = subprocess.run(
                ["nvidia-smi"],
                check=True,
                capture_output=True,
                text=True,
                creationflags=no_window_creationflags(),
            )
            gpu_free, gpu_total = _parse_nvidia_smi_table_memory(completed.stdout)
        except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
            pass

    return ResourceSnapshot(
        free_ram_mb=int(memory.available / 1024 / 1024),
        process_rss_mb=int(process.memory_info().rss / 1024 / 1024),
        gpu_free_mb=gpu_free,
        gpu_total_mb=gpu_total,
    )


def choose_device(min_free_vram_mb: int) -> str:
    snapshot = capture_snapshot()
    if snapshot.gpu_total_mb > 0 and snapshot.gpu_free_mb >= min_free_vram_mb:
        return "cuda"
    return "cpu"


def ensure_safe_to_start_job(min_free_ram_mb: int, max_rss_mb: int) -> ResourceSnapshot:
    snapshot = capture_snapshot()
    if snapshot.free_ram_mb < min_free_ram_mb:
        raise ResourceGuardError(
            f"Free RAM is too low to start a job ({snapshot.free_ram_mb} MB < {min_free_ram_mb} MB)."
        )
    if snapshot.process_rss_mb > max_rss_mb:
        raise ResourceGuardError(
            f"Worker RSS is too high ({snapshot.process_rss_mb} MB > {max_rss_mb} MB)."
        )
    return snapshot


def ensure_safe_to_start_gpu_phase(min_free_ram_mb: int, min_free_vram_mb: int, max_rss_mb: int) -> ResourceSnapshot:
    snapshot = ensure_safe_to_start_job(min_free_ram_mb=min_free_ram_mb, max_rss_mb=max_rss_mb)
    if snapshot.gpu_total_mb > 0 and snapshot.gpu_free_mb < min_free_vram_mb:
        raise ResourceGuardError(
            f"Free GPU VRAM is too low ({snapshot.gpu_free_mb} MB < {min_free_vram_mb} MB)."
        )
    return snapshot
