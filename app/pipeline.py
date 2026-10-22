from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import psutil

from app.engine import InferenceResult, RealESRGANEngine
from app.image_io import load_image, save_image

logger = logging.getLogger(__name__)

PROFILES: dict[str, float] = {
    "Natural": 0.72,
    "Photo Restoration": 0.90,
    "Maximum Detail": 1.0,
    "Portrait": 0.75,
    "Artwork": 0.95,
    "Text & Documents": 0.60,
    "Custom": 0.85,
}


@dataclass(frozen=True)
class ProcessingOptions:
    output_dir: Path
    suffix: str
    output_format: str
    quality: int
    scale: int
    profile: str
    strength: float | None
    processing_mode: str
    cpu_threads: int
    use_cuda: bool


@dataclass(frozen=True)
class ProcessingResult:
    source: Path
    destination: Path
    width: int
    height: int
    backend: str
    elapsed_seconds: float
    warnings: tuple[str, ...]
    preview: Image.Image


def process_image(
    source_path: Path,
    options: ProcessingOptions,
    cancel_requested=None,
    engine: RealESRGANEngine | None = None,
) -> ProcessingResult:
    import time

    started = time.monotonic()
    if cancel_requested and cancel_requested():
        raise InterruptedError("Processing was cancelled.")
    if options.scale not in {1, 2, 4}:
        raise ValueError("Scale must be 1x, 2x, or 4x.")
    loaded = load_image(source_path)
    original = loaded.image
    logger.info(
        "Input validated: path=%s format=%s dimensions=%sx%s mode=%s channels=%s",
        source_path,
        loaded.source_format,
        original.width,
        original.height,
        original.mode,
        len(original.getbands()),
    )
    if original.width <= 0 or original.height <= 0:
        raise ValueError("The input image has invalid dimensions.")
    if original.width * original.height > 200_000_000:
        raise ValueError("Images larger than 200 megapixels are not supported.")
    output_pixels = original.width * original.height * options.scale**2
    estimated_working_bytes = (
        original.width * original.height * 8 + output_pixels * 10
    )
    available_bytes = psutil.virtual_memory().available
    if estimated_working_bytes > available_bytes * 0.8:
        raise ValueError(
            "Estimated image memory exceeds 80% of currently available RAM. "
            "Choose a smaller scale or free memory before retrying."
        )
    logger.info(
        "Estimated working memory for %s: %.1f MB",
        source_path.name,
        estimated_working_bytes / (1024**2),
    )

    strength = (
        options.strength
        if options.strength is not None
        else PROFILES.get(options.profile, PROFILES["Natural"])
    )
    if engine is None:
        engine = RealESRGANEngine(
            "cuda" if options.use_cuda else "cpu",
            options.processing_mode,
            options.cpu_threads,
        )

    grayscale = original.mode == "L"
    if "A" in original.getbands():
        alpha = original.getchannel("A")
        color = original.convert("RGB")
    else:
        alpha = None
        color = original.convert("RGB")
    try:
        inference: InferenceResult = engine.enhance(color, options.scale, strength)
    except Exception:
        try:
            engine.recover_after_failure()
        except Exception:
            logger.exception("Failed to reset inference state after an error")
        raise
    logger.info(
        "Inference completed: model=Real-ESRGAN x4plus backend=%s tile=%s",
        inference.backend,
        inference.tile_size,
    )
    if cancel_requested and cancel_requested():
        raise InterruptedError("Processing was cancelled.")
    output = inference.image
    if grayscale:
        output = output.convert("L")
    if alpha is not None:
        resized_alpha = alpha.resize(output.size, Image.Resampling.LANCZOS)
        output.putalpha(resized_alpha)

    destination = save_image(
        output,
        options.output_dir,
        source_path,
        options.suffix,
        options.output_format,
        options.quality,
        loaded.icc_profile,
    )
    expected_size = (original.width * options.scale, original.height * options.scale)
    try:
        with Image.open(destination) as saved:
            saved.load()
            if saved.size != expected_size:
                raise ValueError(
                    f"Saved output dimensions {saved.size} do not match "
                    f"expected dimensions {expected_size}."
                )
            if saved.width <= 0 or saved.height <= 0:
                raise ValueError("Saved output has invalid dimensions.")
            if destination.stat().st_size <= 0:
                raise ValueError("Saved output is empty.")
    except Exception:
        try:
            destination.unlink(missing_ok=True)
        except OSError:
            logger.exception("Could not remove invalid output %s", destination)
        logger.exception("Output validation failed for %s", destination)
        raise
    logger.info(
        "Output validated: path=%s dimensions=%sx%s bytes=%s",
        destination,
        expected_size[0],
        expected_size[1],
        destination.stat().st_size,
    )
    elapsed = time.monotonic() - started
    logger.info(
        "Processed %s to %s (%sx%s, scale=%sx, backend=%s, %.2fs)",
        source_path.name,
        destination.name,
        output.width,
        output.height,
        options.scale,
        inference.backend,
        elapsed,
    )
    preview = output.copy()
    preview.thumbnail((1400, 900), Image.Resampling.LANCZOS)
    return ProcessingResult(
        source=source_path,
        destination=destination,
        width=output.width,
        height=output.height,
        backend=inference.backend,
        elapsed_seconds=elapsed,
        warnings=inference.warnings,
        preview=preview,
    )
