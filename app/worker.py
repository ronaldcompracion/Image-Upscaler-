from __future__ import annotations

import logging
import threading
import traceback
import uuid
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from app.engine import RealESRGANEngine
from app.job_context import reset_job_id, set_job_id
from app.model_registry import download_model
from app.pipeline import ProcessingOptions, process_image

logger = logging.getLogger(__name__)


def _to_qimage(image):
    from PySide6.QtGui import QImage

    if image.mode == "RGBA":
        pixel_format = QImage.Format.Format_RGBA8888
    else:
        image = image.convert("RGB")
        pixel_format = QImage.Format.Format_RGB888
    return QImage(
        image.tobytes(),
        image.width,
        image.height,
        image.width * len(image.getbands()),
        pixel_format,
    ).copy()


class ProcessingWorker(QObject):
    progress = Signal(int, int, str)
    item_finished = Signal(str, object)
    item_failed = Signal(str, str)
    item_cancelled = Signal(str)
    finished = Signal()

    def __init__(self, files: list[Path], options: ProcessingOptions):
        super().__init__()
        self.files = files
        self.options = options
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def _mark_cancelled(self, paths: list[Path]) -> None:
        for path in paths:
            token = set_job_id(uuid.uuid4().hex[:12])
            logger.info("Job cancelled before processing: input=%s", path)
            self.item_cancelled.emit(str(path))
            reset_job_id(token)

    @Slot()
    def run(self) -> None:
        engine: RealESRGANEngine | None = None
        try:
            batch_id = uuid.uuid4().hex[:12]
            batch_token = set_job_id(batch_id)
            try:
                logger.info("Batch worker started for %s input(s)", len(self.files))
                if self._cancel.is_set():
                    self._mark_cancelled(self.files)
                    return
                engine = RealESRGANEngine(
                    "cuda" if self.options.use_cuda else "cpu",
                    self.options.processing_mode,
                    self.options.cpu_threads,
                )
            except Exception as error:
                for path in self.files:
                    job_id = uuid.uuid4().hex[:12]
                    token = set_job_id(job_id)
                    logger.exception(
                        "Job failed during model/device initialization: %s", path
                    )
                    self.item_failed.emit(
                        str(path), f"{error}\n\n{traceback.format_exc()}"
                    )
                    reset_job_id(token)
                return
            finally:
                reset_job_id(batch_token)

            for index, path in enumerate(self.files, start=1):
                if self._cancel.is_set():
                    self._mark_cancelled(self.files[index - 1 :])
                    break
                job_id = uuid.uuid4().hex[:12]
                token = set_job_id(job_id)
                logger.info("Job started: input=%s", path)
                self.progress.emit(index, len(self.files), str(path))
                try:
                    result = process_image(
                        path,
                        self.options,
                        self._cancel.is_set,
                        engine,
                    )
                    self.item_finished.emit(str(path), result)
                    logger.info(
                        "Job succeeded: output=%s duration=%.2fs",
                        result.destination,
                        result.elapsed_seconds,
                    )
                except InterruptedError:
                    logger.info("Job cancelled: input=%s", path)
                    self.item_cancelled.emit(str(path))
                    self._mark_cancelled(self.files[index:])
                    break
                except Exception as error:
                    details = f"{error}\n\n{traceback.format_exc()}"
                    logger.exception("Job failed: input=%s", path)
                    self.item_failed.emit(str(path), details)
                finally:
                    logger.info("Job cleanup complete: input=%s", path)
                    reset_job_id(token)
        finally:
            try:
                if engine is not None:
                    try:
                        engine.close()
                    except Exception:
                        logger.exception("Failed to release inference resources")
            finally:
                self.finished.emit()


class ModelDownloadWorker(QObject):
    progress = Signal(int, int)
    finished = Signal(bool, str)

    @Slot()
    def run(self) -> None:
        try:
            path = download_model(
                lambda received, total: self.progress.emit(received, total)
            )
            self.finished.emit(True, str(path))
        except Exception as error:
            logger.exception("Model download failed")
            self.finished.emit(False, str(error))
