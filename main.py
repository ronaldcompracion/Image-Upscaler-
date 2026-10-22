from __future__ import annotations

import logging
import sys

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QMessageBox

from app.logging_config import configure_logging
from app.paths import ensure_app_directories


def main() -> int:
    ensure_app_directories()
    configure_logging()
    logger = logging.getLogger(__name__)
    logger.info("Application starting")
    application = QApplication(sys.argv)
    application.setApplicationName("Local AI Image Enhancer")
    try:
        from app.ui import MainWindow

        window = MainWindow()
        window.show()
        result = application.exec()
        logger.info("Application stopped with exit code %s", result)
        window.close()
        window.deleteLater()
        application.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        application.processEvents()
        return result
    except Exception as error:
        logger.exception("Application startup failed")
        QMessageBox.critical(
            None,
            "Application startup failed",
            f"{error}\n\nSee the application log under the LocalAIImageEnhancer folder.",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
