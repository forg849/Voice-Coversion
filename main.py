

import sys
import os
import traceback
from pathlib import Path

# Если проект запускают системным Python, используем зависимости локального
# окружения проекта с той же версией интерпретатора.
_project_venv_site = Path(__file__).resolve().parent.parent / ".venv" / "Lib" / "site-packages"
if _project_venv_site.is_dir() and str(_project_venv_site) not in sys.path:
    sys.path.insert(0, str(_project_venv_site))

from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt


def _resolve_base_dir() -> Path:
    """
    Определяет базовую директорию программы. Если приложение собрано через
    PyInstaller (sys.frozen), данные хранятся рядом с .exe, а не во
    временной распаковочной папке _MEIPASS.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent

    script_path = globals().get("__file__")
    if script_path:
        return Path(script_path).resolve().parent

    return Path(sys.argv[0]).resolve().parent if sys.argv[0] else Path.cwd()


def _install_exception_hook(app: QApplication):
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        error_text = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        sys.stderr.write(error_text)

        try:
            log_path = _resolve_base_dir() / "data" / "last_error.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(error_text + "\n" + ("-" * 60) + "\n")
        except OSError:
            pass

        box = QMessageBox()
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("Voice Conversion Studio - Ошибка")
        box.setText(
            "Произошла непредвиденная ошибка. Программа попытается продолжить работу.\n"
            "Подробности сохранены в data/last_error.log."
        )
        box.setDetailedText(error_text)
        box.exec()

    sys.excepthook = handle_exception


def main():
    base_dir = _resolve_base_dir()
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    
    sys.path.insert(0, str(base_dir))

    app = QApplication(sys.argv)
    app.setApplicationName("Voice Conversion Studio")
    app.setQuitOnLastWindowClosed(False)  

    _install_exception_hook(app)

    from config.config_manager import ConfigManager
    from gui.main_window import MainWindow

    config_manager = ConfigManager(str(data_dir))
    window = MainWindow(config_manager, str(base_dir))

    if config_manager.get("interface", "start_minimized", False):
        window.hide()
    else:
        window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
