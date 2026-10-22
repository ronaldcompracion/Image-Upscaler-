from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from app.paths import MODEL_DIR

logger = logging.getLogger(__name__)

MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.1.0/RealESRGAN_x4plus.pth"
)
MODEL_FILENAME = "RealESRGAN_x4plus.pth"
MODEL_SIZE_BYTES = 67_040_989
MODEL_SHA256: str | None = None


@dataclass(frozen=True)
class ModelInfo:
    name: str
    filename: str
    description: str
    installed: bool
    size_bytes: int | None


def model_path() -> Path:
    return MODEL_DIR / MODEL_FILENAME


def is_model_installed() -> bool:
    path = model_path()
    if not path.is_file() or path.stat().st_size != MODEL_SIZE_BYTES:
        return False
    digest_path = path.with_suffix(path.suffix + ".sha256")
    if not digest_path.is_file():
        return False
    expected = digest_path.read_text(encoding="ascii").strip()
    return len(expected) == 64 and _sha256(path) == expected


def get_model_info() -> ModelInfo:
    path = model_path()
    installed = is_model_installed()
    return ModelInfo(
        name="Real-ESRGAN x4plus",
        filename=MODEL_FILENAME,
        description="General-purpose photo and artwork restoration; native 4x SR.",
        installed=installed,
        size_bytes=path.stat().st_size if installed else MODEL_SIZE_BYTES,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_model(progress_callback=None) -> Path:
    parsed = urlparse(MODEL_URL)
    if parsed.scheme != "https" or parsed.hostname != "github.com":
        raise ValueError("The model source is not an approved HTTPS GitHub URL.")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    destination = model_path()
    if is_model_installed():
        return destination

    temporary_path: Path | None = None
    digest = hashlib.sha256()
    received = 0
    try:
        request = urllib.request.Request(
            MODEL_URL,
            headers={"User-Agent": "LocalAIImageEnhancer/1.0"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            final_url = urlparse(response.geturl())
            if final_url.scheme != "https" or final_url.hostname not in {
                "github.com",
                "objects.githubusercontent.com",
                "release-assets.githubusercontent.com",
            }:
                raise ValueError("The model download redirected to an untrusted host.")
            with tempfile.NamedTemporaryFile(
                prefix=".model-", suffix=".download", dir=MODEL_DIR, delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    received += len(block)
                    if received > MODEL_SIZE_BYTES:
                        raise ValueError("Downloaded model exceeds the expected size.")
                    temporary.write(block)
                    digest.update(block)
                    if progress_callback:
                        progress_callback(received, MODEL_SIZE_BYTES)
        if received != MODEL_SIZE_BYTES:
            raise ValueError(
                f"Model size mismatch: expected {MODEL_SIZE_BYTES:,} bytes, "
                f"received {received:,}."
            )
        checksum = digest.hexdigest()
        if MODEL_SHA256 is not None and checksum != MODEL_SHA256:
            raise ValueError("Model SHA-256 does not match the trusted checksum.")
        os.replace(temporary_path, destination)
        temporary_path = None
        destination.with_suffix(destination.suffix + ".sha256").write_text(
            checksum, encoding="ascii"
        )
        logger.info("Installed model %s; sha256=%s", MODEL_FILENAME, checksum)
        return destination
    except (OSError, urllib.error.URLError, ValueError):
        logger.exception("Model download failed")
        raise
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
