from __future__ import annotations

import logging
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import psutil

from app.paths import APP_DIR

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HardwareReport:
    operating_system: str
    python_version: str
    cpu: str
    logical_cores: int
    ram_total_gb: float
    ram_available_gb: float
    gpu: str
    vram_total_mb: int | None
    vram_free_mb: int | None
    cuda_available: bool
    torch_version: str
    disk_free_gb: float
    backend: str

    def summary(self) -> str:
        gpu = self.gpu
        if self.vram_total_mb is not None:
            gpu += f" ({self.vram_free_mb or 0}/{self.vram_total_mb} MB free)"
        return (
            f"{self.backend} | {gpu} | "
            f"{self.ram_available_gb:.1f} GB RAM available"
        )


def _nvidia_device() -> tuple[str, int | None, int | None]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return "No NVIDIA GPU detected", None, None
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        first_line = result.stdout.splitlines()[0]
        name, total, free = (part.strip() for part in first_line.split(",", 2))
        return name, int(total), int(free)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        logger.warning("Could not read NVIDIA GPU details", exc_info=True)
        return "NVIDIA GPU detected; details unavailable", None, None


def detect_hardware() -> HardwareReport:
    gpu, vram_total, vram_free = _nvidia_device()
    torch_version = "not installed"
    cuda_available = False
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_total = int(props.total_memory / (1024 * 1024))
            vram_free = int(torch.cuda.mem_get_info(0)[0] / (1024 * 1024))
    except ImportError:
        pass
    except (RuntimeError, OSError):
        cuda_available = False
        logger.warning("CUDA probe failed; CPU inference will be used", exc_info=True)

    free_disk = shutil.disk_usage(APP_DIR if APP_DIR.exists() else Path.home()).free
    report = HardwareReport(
        operating_system=f"{platform.system()} {platform.release()}",
        python_version=platform.python_version(),
        cpu=platform.processor() or platform.machine(),
        logical_cores=psutil.cpu_count(logical=True) or 1,
        ram_total_gb=psutil.virtual_memory().total / (1024**3),
        ram_available_gb=psutil.virtual_memory().available / (1024**3),
        gpu=gpu,
        vram_total_mb=vram_total,
        vram_free_mb=vram_free,
        cuda_available=cuda_available,
        torch_version=torch_version,
        disk_free_gb=free_disk / (1024**3),
        backend="CUDA" if cuda_available else "CPU",
    )
    logger.info(
        "Hardware: OS=%s, Python=%s, CPU=%s (%s logical cores), RAM=%.1f GB, "
        "GPU=%s, CUDA=%s, torch=%s, disk_free=%.1f GB",
        report.operating_system,
        report.python_version,
        report.cpu,
        report.logical_cores,
        report.ram_total_gb,
        report.gpu,
        report.cuda_available,
        report.torch_version,
        report.disk_free_gb,
    )
    return report
