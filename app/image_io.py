from __future__ import annotations

import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}
OUTPUT_FORMATS = {
    "PNG": ("PNG", ".png"),
    "JPEG": ("JPEG", ".jpg"),
    "WEBP": ("WEBP", ".webp"),
    "TIFF": ("TIFF", ".tiff"),
}


@dataclass(frozen=True)
class LoadedImage:
    image: Image.Image
    icc_profile: bytes | None
    source_format: str


def load_image(path: Path) -> LoadedImage:
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {path.suffix or '(no extension)'}")
    try:
        with Image.open(path) as source:
            source.load()
            oriented = ImageOps.exif_transpose(source)
            has_alpha = "A" in oriented.getbands() or "transparency" in oriented.info
            if has_alpha:
                image = oriented.convert("RGBA")
            elif oriented.mode == "L":
                image = oriented.copy()
            else:
                image = oriented.convert("RGB")
            return LoadedImage(
                image=image,
                icc_profile=source.info.get("icc_profile"),
                source_format=source.format or path.suffix.lstrip(".").upper(),
            )
    except (OSError, ValueError) as error:
        logger.warning("Failed to load image %s: %s", path.name, error)
        raise ValueError(f"Could not read image '{path.name}': {error}") from error


def _unique_destination(directory: Path, stem: str, extension: str) -> Path:
    candidate = directory / f"{stem}{extension}"
    suffix = 2
    while candidate.exists():
        candidate = directory / f"{stem}_{suffix}{extension}"
        suffix += 1
    return candidate


def save_image(
    image: Image.Image,
    output_dir: Path,
    source_path: Path,
    suffix: str,
    output_format: str,
    quality: int,
    icc_profile: bytes | None = None,
) -> Path:
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"Unsupported output format: {output_format}")
    if len(suffix) > 80 or re.fullmatch(r"[A-Za-z0-9._-]*", suffix) is None:
        raise ValueError("Filename suffix may contain only letters, digits, '.', '_' or '-'.")
    output_dir.mkdir(parents=True, exist_ok=True)
    pil_format, extension = OUTPUT_FORMATS[output_format]
    if output_format == "JPEG" and image.mode in ("RGBA", "LA"):
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image.convert("RGB"), mask=image.getchannel("A"))
        image = background
    elif output_format == "JPEG" and image.mode != "RGB":
        image = image.convert("RGB")

    options: dict[str, object] = {}
    if output_format in {"JPEG", "WEBP"}:
        options["quality"] = max(1, min(100, quality))
    if icc_profile:
        options["icc_profile"] = icc_profile

    stem = f"{source_path.stem}{suffix}"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".enhanced-", suffix=".tmp", dir=output_dir, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        image.save(temporary_path, format=pil_format, **options)
        destination = _unique_destination(output_dir, stem, extension)
        collision_suffix = 2
        while True:
            try:
                os.rename(temporary_path, destination)
                break
            except FileExistsError:
                destination = output_dir / f"{stem}_{collision_suffix}{extension}"
                collision_suffix += 1
        return destination
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
