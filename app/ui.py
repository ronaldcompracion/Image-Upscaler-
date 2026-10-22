from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QSettings, Qt, QThread, Slot
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.hardware import detect_hardware
from app.image_io import SUPPORTED_EXTENSIONS, load_image
from app.model_registry import get_model_info, is_model_installed
from app.paths import APP_DIR, DEFAULT_OUTPUT_DIR
from app.pipeline import ProcessingOptions, ProcessingResult, PROFILES
from app.worker import ModelDownloadWorker, ProcessingWorker

logger = logging.getLogger(__name__)


def _pil_to_qimage(image: Image.Image) -> QImage:
    image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
    image_format = (
        QImage.Format.Format_RGBA8888
        if image.mode == "RGBA"
        else QImage.Format.Format_RGB888
    )
    return QImage(
        image.tobytes(),
        image.width,
        image.height,
        image.width * len(image.getbands()),
        image_format,
    ).copy()


class SettingsDialog(QDialog):
    def __init__(self, parent: QWidget, hardware_summary: str):
        super().__init__(parent)
        self.setWindowTitle("Settings and diagnostics")
        layout = QVBoxLayout(self)
        form = QFormLayout()
        model = get_model_info()
        form.addRow("Hardware", QLabel(hardware_summary))
        form.addRow("Model", QLabel(f"{model.name}: {'Installed' if model.installed else 'Not installed'}"))
        form.addRow("Model location", QLabel(str(APP_DIR / "models")))
        form.addRow("Logs", QLabel(str(APP_DIR / "logs")))
        form.addRow(
            "Face restoration",
            QLabel("Not integrated; no face-changing processing is applied."),
        )
        form.addRow(
            "Network activity",
            QLabel("Only the explicit model download button accesses the network."),
        )
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("LocalAIImageEnhancer", "LocalAIImageEnhancer")
        self.hardware = detect_hardware()
        self.queue: dict[str, str] = {}
        self._processing_thread: QThread | None = None
        self._processing_worker: ProcessingWorker | None = None
        self._download_thread: QThread | None = None
        self._download_worker: ModelDownloadWorker | None = None
        self.setWindowTitle("Local AI Image Enhancer")
        self.setMinimumSize(1060, 700)
        self.setAcceptDrops(True)
        self._build_ui()
        self._refresh_model_status()
        self.hardware_label.setText(self.hardware.summary())

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        self.setCentralWidget(central)

        top = QHBoxLayout()
        title = QLabel("Local AI Image Enhancer")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        top.addWidget(title)
        top.addStretch(1)
        self.hardware_label = QLabel()
        top.addWidget(self.hardware_label)
        settings_button = QPushButton("Settings")
        settings_button.clicked.connect(self._show_settings)
        top.addWidget(settings_button)
        root.addLayout(top)

        self.model_label = QLabel()
        root.addWidget(self.model_label)

        body = QSplitter(Qt.Orientation.Horizontal)
        self.queue_list = QListWidget()
        self.queue_list.currentItemChanged.connect(self._show_selected)
        body.addWidget(self.queue_list)

        preview_container = QWidget()
        previews = QHBoxLayout(preview_container)
        self.original_preview = self._preview_panel("Original")
        self.enhanced_preview = self._preview_panel("Enhanced")
        previews.addWidget(self.original_preview[0])
        previews.addWidget(self.enhanced_preview[0])
        body.addWidget(preview_container)
        body.setStretchFactor(0, 1)
        body.setStretchFactor(1, 3)
        root.addWidget(body, 1)

        options_group = QGroupBox("Enhancement and export")
        options = QFormLayout(options_group)
        saved_profile = self.settings.value("profile", "Natural", type=str)
        if saved_profile not in PROFILES:
            saved_profile = "Natural"
        saved_scale = self.settings.value("scale", "4x", type=str)
        if saved_scale not in {"1x", "2x", "4x"}:
            saved_scale = "4x"
        saved_mode = self.settings.value("processing_mode", "Automatic", type=str)
        if saved_mode not in {"Automatic", "Balanced", "Maximum quality"}:
            saved_mode = "Automatic"
        saved_format = self.settings.value("output_format", "PNG", type=str)
        if saved_format not in {"PNG", "JPEG", "WEBP", "TIFF"}:
            saved_format = "PNG"
        row = QHBoxLayout()
        self.profile_combo = QComboBox()
        self.profile_combo.addItems(PROFILES)
        self.profile_combo.setCurrentText(saved_profile)
        self.profile_combo.currentTextChanged.connect(self._update_strength)
        row.addWidget(self.profile_combo)
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["1x", "2x", "4x"])
        self.scale_combo.setCurrentText(saved_scale)
        row.addWidget(self.scale_combo)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Automatic", "Balanced", "Maximum quality"])
        self.mode_combo.setCurrentText(saved_mode)
        row.addWidget(self.mode_combo)
        options.addRow("Profile / scale / processing mode", row)

        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0, 100)
        default_strength = round(PROFILES[self.profile_combo.currentText()] * 100)
        saved_strength = self.settings.value("strength", default_strength, type=int)
        self.strength_slider.setValue(max(0, min(100, saved_strength)))
        self.strength_label = QLabel("72%")
        strength_row = QHBoxLayout()
        strength_row.addWidget(self.strength_slider, 1)
        strength_row.addWidget(self.strength_label)
        self.strength_slider.valueChanged.connect(
            lambda value: self.strength_label.setText(f"{value}%")
        )
        options.addRow("Enhancement strength", strength_row)

        output_row = QHBoxLayout()
        self.output_dir = QLineEdit(
            self.settings.value("output_dir", str(DEFAULT_OUTPUT_DIR), type=str)
        )
        browse_output = QPushButton("Browse…")
        browse_output.clicked.connect(self._choose_output_directory)
        output_row.addWidget(self.output_dir, 1)
        output_row.addWidget(browse_output)
        options.addRow("Output folder", output_row)

        export_row = QHBoxLayout()
        self.suffix_edit = QLineEdit(self.settings.value("suffix", "_enhanced", type=str))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG", "JPEG", "WEBP", "TIFF"])
        self.format_combo.setCurrentText(saved_format)
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(1, 100)
        saved_quality = self.settings.value("quality", 95, type=int)
        self.quality_spin.setValue(max(1, min(100, saved_quality)))
        export_row.addWidget(QLabel("Filename suffix"))
        export_row.addWidget(self.suffix_edit)
        export_row.addWidget(QLabel("Format"))
        export_row.addWidget(self.format_combo)
        export_row.addWidget(QLabel("Quality"))
        export_row.addWidget(self.quality_spin)
        options.addRow("Output options", export_row)
        root.addWidget(options_group)

        controls = QHBoxLayout()
        self.add_button = QPushButton("Add images")
        self.add_button.clicked.connect(self._add_files)
        self.remove_button = QPushButton("Remove selected")
        self.remove_button.clicked.connect(self._remove_selected)
        self.download_button = QPushButton("Download AI model")
        self.download_button.clicked.connect(self._download_model)
        self.start_button = QPushButton("Start batch")
        self.start_button.clicked.connect(self._start_batch)
        self.cancel_button = QPushButton("Cancel after current inference")
        self.cancel_button.clicked.connect(self._cancel_batch)
        self.cancel_button.setEnabled(False)
        self.retry_button = QPushButton("Retry failed")
        self.retry_button.clicked.connect(self._retry_failed)
        self.retry_button.setEnabled(False)
        self.open_folder_button = QPushButton("Open output folder")
        self.open_folder_button.clicked.connect(self._open_output_folder)
        for button in (
            self.add_button,
            self.remove_button,
            self.download_button,
            self.start_button,
            self.cancel_button,
            self.retry_button,
            self.open_folder_button,
        ):
            controls.addWidget(button)
        root.addLayout(controls)

        self.progress = QProgressBar()
        self.progress.setFormat("Ready")
        root.addWidget(self.progress)
        self.status_label = QLabel("Add images or drop them anywhere in this window.")
        root.addWidget(self.status_label)

    @staticmethod
    def _preview_panel(title: str) -> tuple[QGroupBox, QLabel]:
        panel = QGroupBox(title)
        layout = QVBoxLayout(panel)
        image = QLabel("No preview")
        image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image.setMinimumSize(300, 300)
        image.setStyleSheet("background: #202124; color: #bbb;")
        layout.addWidget(image)
        return panel, image

    @Slot()
    def _refresh_model_status(self) -> None:
        info = get_model_info()
        status = "Installed" if info.installed else "Not installed"
        self.model_label.setText(
            f"Model: {info.name} — {status} | "
            f"{info.size_bytes / (1024**2):.1f} MB | "
            "Official Real-ESRGAN release; general image restoration"
        )
        self.download_button.setEnabled(
            not info.installed and self._download_thread is None
        )
        self.start_button.setEnabled(
            info.installed and self._processing_thread is None
        )
        self.retry_button.setEnabled(
            self._processing_thread is None
            and any(status.startswith("Failed:") for status in self.queue.values())
        )

    @Slot()
    def _add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select images",
            "",
            "Images (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff)",
        )
        self._add_paths([Path(path) for path in files])

    def _add_paths(self, paths: list[Path]) -> None:
        for path in paths:
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS or not path.is_file():
                continue
            key = str(path.resolve())
            if key in self.queue:
                continue
            self.queue[key] = "Queued"
            item = QListWidgetItem(f"{path.name} — Queued")
            item.setData(Qt.ItemDataRole.UserRole, key)
            try:
                loaded = load_image(path)
                thumbnail = loaded.image.copy()
                thumbnail.thumbnail((64, 64), Image.Resampling.LANCZOS)
                item.setIcon(QPixmap.fromImage(_pil_to_qimage(thumbnail)))
            except ValueError as error:
                item.setText(f"{path.name} — Invalid: {error}")
                self.queue[key] = f"Failed: {error}"
            self.queue_list.addItem(item)
        if self.queue_list.count() and self.queue_list.currentRow() < 0:
            self.queue_list.setCurrentRow(0)

    @Slot()
    def _remove_selected(self) -> None:
        if self._processing_thread is not None:
            return
        for item in self.queue_list.selectedItems():
            key = item.data(Qt.ItemDataRole.UserRole)
            self.queue.pop(key, None)
            self.queue_list.takeItem(self.queue_list.row(item))
        self.original_preview[1].setText("No preview")

    @Slot(QListWidgetItem, QListWidgetItem)
    def _show_selected(self, current, _previous) -> None:
        if current is None:
            return
        path = Path(current.data(Qt.ItemDataRole.UserRole))
        try:
            loaded = load_image(path)
            preview = loaded.image.copy()
            preview.thumbnail((1200, 900), Image.Resampling.LANCZOS)
            self._set_preview(self.original_preview[1], _pil_to_qimage(preview))
        except (OSError, ValueError) as error:
            self.original_preview[1].setText(str(error))

    @staticmethod
    def _set_preview(label: QLabel, image: QImage) -> None:
        label.setPixmap(
            QPixmap.fromImage(image).scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    @Slot(str)
    def _update_strength(self, profile: str) -> None:
        self.strength_slider.setValue(round(PROFILES.get(profile, 0.72) * 100))

    @Slot()
    def _choose_output_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self, "Choose output folder", self.output_dir.text()
        )
        if selected:
            self.output_dir.setText(selected)

    def _options(self) -> ProcessingOptions:
        scale = int(self.scale_combo.currentText().removesuffix("x"))
        mode = self.mode_combo.currentText()
        return ProcessingOptions(
            output_dir=Path(self.output_dir.text()).expanduser(),
            suffix=self.suffix_edit.text() or "_enhanced",
            output_format=self.format_combo.currentText(),
            quality=self.quality_spin.value(),
            scale=scale,
            profile=self.profile_combo.currentText(),
            strength=self.strength_slider.value() / 100,
            processing_mode=mode,
            cpu_threads=max(1, (self.hardware.logical_cores or 2) - 1),
            use_cuda=self.hardware.cuda_available,
        )

    @Slot()
    def _start_batch(self) -> None:
        pending = [
            Path(key)
            for key, status in self.queue.items()
            if status in {"Queued", "Cancelled"} or status.startswith("Failed:")
        ]
        if not pending:
            QMessageBox.information(self, "No pending images", "Add or retry images first.")
            return
        if not is_model_installed():
            QMessageBox.warning(
                self,
                "AI model required",
                "Download the Real-ESRGAN model before processing images.",
            )
            return
        try:
            output = Path(self.output_dir.text()).expanduser()
            output.mkdir(parents=True, exist_ok=True)
            if not output.is_dir():
                raise OSError("The selected output path is not a directory.")
            self.settings.setValue("output_dir", str(output))
            self.settings.setValue("suffix", self.suffix_edit.text())
            self.settings.setValue("profile", self.profile_combo.currentText())
            self.settings.setValue("scale", self.scale_combo.currentText())
            self.settings.setValue("processing_mode", self.mode_combo.currentText())
            self.settings.setValue("strength", self.strength_slider.value())
            self.settings.setValue("output_format", self.format_combo.currentText())
            self.settings.setValue("quality", self.quality_spin.value())
            self._launch_processing(pending)
        except OSError as error:
            QMessageBox.critical(self, "Output folder error", str(error))

    @Slot()
    def _retry_failed(self) -> None:
        for path, status in tuple(self.queue.items()):
            if status.startswith("Failed:"):
                self.queue[path] = "Queued"
                self._update_queue_item(path, "Queued")
        self._start_batch()

    def _launch_processing(self, files: list[Path]) -> None:
        thread = QThread(self)
        worker = ProcessingWorker(files, self._options())
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.item_finished.connect(self._on_item_finished)
        worker.item_failed.connect(self._on_item_failed)
        worker.item_cancelled.connect(self._on_item_cancelled)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._processing_done)
        self._processing_thread = thread
        self._processing_worker = worker
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self._refresh_model_status()
        thread.start()

    @Slot(int, int, str)
    def _on_progress(self, index: int, total: int, filename: str) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(index - 1)
        name = Path(filename).name
        self.progress.setFormat(f"Image {index}/{total}: {name}")
        for row in range(self.queue_list.count()):
            item = self.queue_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == filename:
                self.queue[filename] = "Processing"
                item.setText(f"{name} — Processing")
                break

    @Slot(str, object)
    def _on_item_finished(self, path: str, result: ProcessingResult) -> None:
        self.queue[path] = f"Done: {result.destination}"
        self._update_queue_item(path, f"Done — {result.destination.name}")
        self._set_preview(self.enhanced_preview[1], _pil_to_qimage(result.preview))
        if result.warnings:
            self.status_label.setText(
                f"Saved {result.destination} | " + " ".join(result.warnings)
            )
        else:
            self.status_label.setText(
                f"Saved {result.width}×{result.height} via {result.backend} "
                f"in {result.elapsed_seconds:.1f}s: {result.destination}"
            )

    @Slot(str, str)
    def _on_item_failed(self, path: str, details: str) -> None:
        message = details.splitlines()[0] if details else "Unknown error"
        self.queue[path] = f"Failed: {message}"
        self._update_queue_item(path, f"Failed — {message}")
        self.status_label.setText(f"{Path(path).name}: {message}")
        logger.error("Image processing failed for %s\n%s", path, details)

    @Slot(str)
    def _on_item_cancelled(self, path: str) -> None:
        self.queue[path] = "Cancelled"
        self._update_queue_item(path, "Cancelled")
        self.status_label.setText(
            f"Cancelled {Path(path).name}; select Start batch to resume it."
        )

    def _update_queue_item(self, path: str, text: str) -> None:
        for row in range(self.queue_list.count()):
            item = self.queue_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == path:
                item.setText(f"{Path(path).name} — {text}")
                break

    @Slot()
    def _cancel_batch(self) -> None:
        if self._processing_worker is not None:
            self._processing_worker.cancel()
            self.cancel_button.setEnabled(False)
            self.status_label.setText(
                "Cancellation requested; current inference will finish before stopping."
            )

    @Slot()
    def _processing_done(self) -> None:
        self._processing_thread = None
        self._processing_worker = None
        self.cancel_button.setEnabled(False)
        self.progress.setValue(self.progress.maximum())
        self.progress.setFormat("Batch complete")
        if any(status == "Cancelled" for status in self.queue.values()):
            self.status_label.setText(
                "Batch stopped; cancelled items are ready to resume."
            )
        elif any(status.startswith("Failed:") for status in self.queue.values()):
            self.status_label.setText(
                "Batch finished with failures. Use Retry failed after correcting the cause."
            )
        else:
            self.status_label.setText("Batch processing finished.")
        self._refresh_model_status()

    @Slot()
    def _download_model(self) -> None:
        if self.hardware.disk_free_gb < 1:
            QMessageBox.warning(
                self, "Low disk space", "Less than 1 GB is available on the model drive."
            )
            return
        self._download_thread = QThread(self)
        self._download_worker = ModelDownloadWorker()
        self._download_worker.moveToThread(self._download_thread)
        self._download_thread.started.connect(self._download_worker.run)
        self._download_worker.progress.connect(self._on_download_progress)
        self._download_worker.finished.connect(self._on_download_finished)
        self._download_worker.finished.connect(self._download_thread.quit)
        self._download_worker.finished.connect(self._download_worker.deleteLater)
        self._download_thread.finished.connect(self._download_thread.deleteLater)
        self._download_thread.finished.connect(self._download_done)
        self.download_button.setEnabled(False)
        self.progress.setRange(0, 100)
        self.progress.setFormat("Downloading model…")
        self._download_thread.start()

    @Slot(int, int)
    def _on_download_progress(self, received: int, total: int) -> None:
        percent = round(received * 100 / total) if total else 0
        self.progress.setValue(percent)
        self.progress.setFormat(f"Downloading model… {percent}%")

    @Slot(bool, str)
    def _on_download_finished(self, success: bool, message: str) -> None:
        if success:
            QMessageBox.information(
                self,
                "Model installed",
                "Real-ESRGAN x4plus is installed locally. Inference can run offline.",
            )
            self.status_label.setText(f"Model installed: {message}")
        else:
            QMessageBox.critical(self, "Model download failed", message)
            self.status_label.setText("Model download failed. See the application log.")

    @Slot()
    def _download_done(self) -> None:
        self._download_thread = None
        self._download_worker = None
        self.progress.setFormat("Ready")
        self._refresh_model_status()

    @Slot()
    def _show_settings(self) -> None:
        SettingsDialog(self, self.hardware.summary()).exec()

    @Slot()
    def _open_output_folder(self) -> None:
        folder = Path(self.output_dir.text()).expanduser()
        import os

        try:
            folder.mkdir(parents=True, exist_ok=True)
            os.startfile(folder)
        except OSError as error:
            QMessageBox.critical(self, "Could not open folder", str(error))

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = [
            Path(url.toLocalFile())
            for url in event.mimeData().urls()
            if url.isLocalFile()
        ]
        self._add_paths(paths)
        event.acceptProposedAction()

    def closeEvent(self, event) -> None:
        if self._processing_thread is not None:
            answer = QMessageBox.question(
                self,
                "Processing in progress",
                "Request cancellation and close after the current inference finishes?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            if self._processing_worker is not None:
                self._processing_worker.cancel()
            self._processing_thread.quit()
            self._processing_thread.wait()
        if self._download_thread is not None:
            QMessageBox.warning(
                self,
                "Model download in progress",
                "Wait for the model download to finish before closing the application.",
            )
            event.ignore()
            return
        event.accept()
