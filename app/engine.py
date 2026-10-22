from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from app.model_registry import model_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceResult:
    image: Image.Image
    backend: str
    tile_size: int
    warnings: tuple[str, ...]


class RealESRGANEngine:
    def __init__(self, device: str, processing_mode: str, cpu_threads: int):
        self.device = device
        self.processing_mode = processing_mode
        self.cpu_threads = max(1, cpu_threads)
        self.tile_size = self._tile_size(device)
        self._upscaler = None
        self._torch = None

    def _tile_size(self, device: str) -> int:
        if device == "cuda":
            if self.processing_mode == "Maximum quality":
                return 256
            if self.processing_mode == "Automatic":
                try:
                    import torch

                    free_memory, _ = torch.cuda.mem_get_info()
                    return 128 if free_memory >= 2 * (1024**3) else 64
                except (RuntimeError, ImportError):
                    return 64
            return 128
        if self.processing_mode == "Maximum quality":
            return 128
        if self.processing_mode == "Automatic":
            import psutil

            return 128 if psutil.virtual_memory().available >= 8 * (1024**3) else 64
        return 96

    def _create_upscaler(self, device: str, tile_size: int):
        try:
            import torch
            import torchvision.transforms.functional as torchvision_functional

            # BasicSR 1.4.2 imports this path, which newer torchvision removed.
            import sys

            sys.modules.setdefault(
                "torchvision.transforms.functional_tensor", torchvision_functional
            )
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError as error:
            raise RuntimeError(
                "AI dependencies are missing. Run install.ps1 to install the "
                "selected PyTorch build and AI packages, "
                "then restart the application."
            ) from error
        if not model_path().is_file():
            raise FileNotFoundError("The Real-ESRGAN x4plus model is not installed.")
        if device == "cpu":
            torch.set_num_threads(self.cpu_threads)
        logger.info(
            "Loading %s on %s with %s px tiles",
            model_path().name,
            device.upper(),
            tile_size,
        )
        network = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        upscaler = RealESRGANer(
            scale=4,
            model_path=str(model_path()),
            model=network,
            tile=tile_size,
            tile_pad=16,
            pre_pad=10,
            half=device == "cuda",
            device=torch.device(device),
        )
        self._torch = torch
        return upscaler

    def close(self) -> None:
        self._upscaler = None
        if self.device == "cuda" and self._torch is not None:
            self._torch.cuda.empty_cache()
        logger.info("Inference model released")

    def recover_after_failure(self) -> None:
        self._upscaler = None
        if self.device == "cuda" and self._torch is not None:
            self._torch.cuda.empty_cache()
        logger.warning("Discarded cached inference state after a failed job")

    @staticmethod
    def _is_out_of_memory(error: RuntimeError) -> bool:
        return "out of memory" in str(error).lower() or "cuda error: memory" in str(
            error
        ).lower()

    def enhance(self, image: Image.Image, scale: int, strength: float) -> InferenceResult:
        if scale not in {1, 2, 4}:
            raise ValueError("Scale must be 1x, 2x, or 4x.")
        if not 0.0 <= strength <= 1.0:
            raise ValueError("Enhancement strength must be between 0 and 1.")
        import torch

        device = self.device
        warnings: list[str] = []
        tile_size = self.tile_size
        source = image.convert("RGB")
        if self._upscaler is None:
            try:
                self._upscaler = self._create_upscaler(device, tile_size)
            except RuntimeError as error:
                if device != "cuda" or (
                    not self._is_out_of_memory(error)
                    and "cuda" not in str(error).lower()
                ):
                    raise
                warnings.append("GPU initialization failed; using CPU inference.")
                logger.warning("Falling back to CPU after CUDA initialization failure")
                torch.cuda.empty_cache()
                device = "cpu"
                tile_size = self._tile_size(device)
                self._upscaler = self._create_upscaler(device, tile_size)
        import numpy as np

        while True:
            self._upscaler.tile_size = tile_size
            try:
                output, _ = self._upscaler.enhance(
                    np.asarray(source)[:, :, ::-1].copy(), outscale=4
                )
                break
            except RuntimeError as error:
                is_oom = self._is_out_of_memory(error)
                is_cuda_failure = device == "cuda" and "cuda" in str(error).lower()
                if not is_oom and not is_cuda_failure:
                    raise
                self._upscaler = None
                if device == "cuda":
                    torch.cuda.empty_cache()
                if is_cuda_failure and not is_oom:
                    device = "cpu"
                    tile_size = self._tile_size(device)
                    warnings.append("GPU inference failed; retried on CPU.")
                    logger.warning("Falling back to CPU after CUDA inference failure")
                    self._upscaler = self._create_upscaler(device, tile_size)
                    continue
                if tile_size > 64:
                    tile_size = max(64, tile_size // 2)
                    warnings.append(
                        f"Memory limit reached; retried with {tile_size}px tiles."
                    )
                    logger.warning("Retrying inference with %s pixel tiles", tile_size)
                    self._upscaler = self._create_upscaler(device, tile_size)
                    continue
                if device == "cuda":
                    device = "cpu"
                    tile_size = self._tile_size(device)
                    warnings.append("GPU memory was insufficient; retried on CPU.")
                    logger.warning("Falling back to CPU after CUDA out-of-memory")
                    torch.cuda.empty_cache()
                    self._upscaler = self._create_upscaler(device, tile_size)
                    continue
                self.device = device
                self.tile_size = tile_size
                raise RuntimeError(
                    "Inference ran out of memory even with the smallest CPU tiles."
                ) from error
        self.device = device
        self.tile_size = tile_size
        result = Image.fromarray(output[:, :, ::-1].copy())
        target_size = (image.width * scale, image.height * scale)
        if result.size != target_size:
            result = result.resize(target_size, Image.Resampling.LANCZOS)
        if strength < 1.0:
            faithful_base = source.resize(target_size, Image.Resampling.LANCZOS)
            result = Image.blend(faithful_base, result, strength)
        return InferenceResult(
            image=result,
            backend=device.upper(),
            tile_size=tile_size,
            warnings=tuple(warnings),
        )
