"""
voice_profile_window.py

Диалоговые окна, связанные с голосовыми профилями:
    - ImportVoiceDialog: импорт аудиофайла -> анализ -> создание профиля
    - AttachModelDialog: прикрепление обученной RVC-модели (.pth/.index) к профилю
    - TestVoiceDialog: запись короткой фразы, прослушивание оригинала и
      результата конвертации (раздел 10 ТЗ)
"""

import os
import tempfile

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QLineEdit, QTextEdit, QMessageBox, QProgressBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from app.audio_engine import analyze_audio, save_processed_audio, AudioEngineError


class _AnalyzeWorker(QThread):
    finished_ok = pyqtSignal(dict)
    finished_error = pyqtSignal(str)

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            result = analyze_audio(self.file_path)
            self.finished_ok.emit(result)
        except AudioEngineError as e:
            self.finished_error.emit(str(e))
        except Exception as e:
            self.finished_error.emit(f"Непредвиденная ошибка анализа: {e}")


class ImportVoiceDialog(QDialog):
    """
    Реализует раздел 4 ТЗ: импорт, предобработка и честный отчёт о
    качестве записи ДО создания профиля.
    """

    def __init__(self, profile_manager, parent=None):
        super().__init__(parent)
        self.profile_manager = profile_manager
        self.setWindowTitle("Import Voice")
        self.setMinimumWidth(440)

        self._selected_path = None
        self._analysis_result = None
        self._worker = None
        self.created_profile = None

        layout = QVBoxLayout(self)

        warning = QLabel(
            "⚠ Используйте только записи, на которые у вас есть право (собственный "
            "голос, голос с явного согласия человека, либо синтетический/вымышленный "
            "голос). Не используйте программу для выдачи себя за реального человека "
            "без его согласия."
        )
        warning.setWordWrap(True)
        warning.setObjectName("warningLabel")
        layout.addWidget(warning)

        file_row = QHBoxLayout()
        self.file_label = QLineEdit()
        self.file_label.setReadOnly(True)
        self.file_label.setPlaceholderText("Файл не выбран (WAV / MP3 / FLAC / OGG)")
        browse_btn = QPushButton("Обзор...")
        browse_btn.clicked.connect(self._on_browse)
        file_row.addWidget(self.file_label, stretch=1)
        file_row.addWidget(browse_btn)
        layout.addLayout(file_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # неопределённый прогресс на время анализа
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.report_box = QTextEdit()
        self.report_box.setReadOnly(True)
        self.report_box.setFixedHeight(140)
        layout.addWidget(self.report_box)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Имя профиля:"))
        self.name_edit = QLineEdit()
        name_row.addWidget(self.name_edit, stretch=1)
        layout.addLayout(name_row)

        buttons_row = QHBoxLayout()
        self.create_btn = QPushButton("Создать профиль")
        self.create_btn.setEnabled(False)
        self.create_btn.clicked.connect(self._on_create_profile)
        cancel_btn = QPushButton("Отмена")
        cancel_btn.clicked.connect(self.reject)
        buttons_row.addStretch(1)
        buttons_row.addWidget(cancel_btn)
        buttons_row.addWidget(self.create_btn)
        layout.addLayout(buttons_row)

    # ------------------------------------------------------------------ #

    def _on_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите аудиозапись голоса", "",
            "Аудио (*.wav *.mp3 *.flac *.ogg)"
        )
        if not path:
            return
        self._selected_path = path
        self.file_label.setText(path)
        base_name = os.path.splitext(os.path.basename(path))[0]
        if not self.name_edit.text():
            self.name_edit.setText(base_name.replace("_", " ").title())

        self.report_box.setPlainText("Анализ записи...")
        self.progress.setVisible(True)
        self.create_btn.setEnabled(False)

        self._worker = _AnalyzeWorker(path)
        self._worker.finished_ok.connect(self._on_analysis_ok)
        self._worker.finished_error.connect(self._on_analysis_error)
        self._worker.start()

    def _on_analysis_ok(self, result: dict):
        self.progress.setVisible(False)
        self._analysis_result = result

        lines = [
            f"Duration: {result['duration_sec']:.0f} sec",
            f"Sample Rate: {result['sample_rate']} Hz",
            f"Channels: {'Mono' if result['channels_original'] == 1 else 'Stereo -> Mono'}",
            f"Quality: {result['quality']}",
            f"Voice detected: {'Yes' if result['voice_detected'] else 'No'}",
        ]
        if result["warnings"]:
            lines.append("")
            lines.append("Предупреждения:")
            for w in result["warnings"]:
                lines.append(f"  • {w}")

        self.report_box.setPlainText("\n".join(lines))
        self.create_btn.setEnabled(True)

    def _on_analysis_error(self, message: str):
        self.progress.setVisible(False)
        self.report_box.setPlainText(f"Ошибка: {message}")
        self.create_btn.setEnabled(False)
        self._analysis_result = None

    def _on_create_profile(self):
        if not self._analysis_result or not self._selected_path:
            return
        display_name = self.name_edit.text().strip() or "New Voice"

        tmp_path = os.path.join(tempfile.gettempdir(), "vcs_import_tmp.wav")
        save_processed_audio(
            self._analysis_result["processed_audio"],
            self._analysis_result["processed_sample_rate"],
            tmp_path,
        )

        analysis_meta = {k: v for k, v in self._analysis_result.items() if k != "processed_audio"}

        try:
            self.created_profile = self.profile_manager.create_profile(
                display_name=display_name,
                source_audio_path=tmp_path,
                analysis_info=analysis_meta,
                model_type="none",
            )
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось создать профиль: {e}")
            return
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        self.accept()


class AttachModelDialog(QDialog):
    """
    Прикрепление уже обученной RVC-модели к профилю (Custom Voice Mode).
    Само обучение выполняется вне этой программы - см. README.md, раздел
    "Custom Voice Mode: как обучить модель".
    """

    def __init__(self, profile_manager, profile_folder: str, parent=None):
        super().__init__(parent)
        self.profile_manager = profile_manager
        self.profile_folder = profile_folder
        self.setWindowTitle("Attach RVC Model")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)

        info = QLabel(
            "Выберите файл обученной RVC-модели (.pth) и, опционально, файл "
            "индекса (.index), полученные во внешнем тренировочном пайплайне "
            "RVC. Для backend so-vits-svc положите рядом с .pth одноимённый "
            ".json-конфиг. Обучение модели не выполняется в этом окне — см. README.md."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        pth_row = QHBoxLayout()
        self.pth_edit = QLineEdit()
        self.pth_edit.setReadOnly(True)
        pth_btn = QPushButton("Выбрать .pth")
        pth_btn.clicked.connect(self._on_browse_pth)
        pth_row.addWidget(self.pth_edit, stretch=1)
        pth_row.addWidget(pth_btn)
        layout.addLayout(pth_row)

        index_row = QHBoxLayout()
        self.index_edit = QLineEdit()
        self.index_edit.setReadOnly(True)
        index_btn = QPushButton("Выбрать .index (опционально)")
        index_btn.clicked.connect(self._on_browse_index)
        index_row.addWidget(self.index_edit, stretch=1)
        index_row.addWidget(index_btn)
        layout.addLayout(index_row)

        buttons_row = QHBoxLayout()
        attach_btn = QPushButton("Прикрепить")
        attach_btn.clicked.connect(self._on_attach)
        cancel_btn = QPushButton("Отмена")
        cancel_btn.clicked.connect(self.reject)
        buttons_row.addStretch(1)
        buttons_row.addWidget(cancel_btn)
        buttons_row.addWidget(attach_btn)
        layout.addLayout(buttons_row)

    def _on_browse_pth(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите модель", "", "RVC модель (*.pth)")
        if path:
            self.pth_edit.setText(path)

    def _on_browse_index(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите индекс", "", "FAISS индекс (*.index)")
        if path:
            self.index_edit.setText(path)

    def _on_attach(self):
        if not self.pth_edit.text():
            QMessageBox.warning(self, "Voice Conversion Studio", "Сначала выберите .pth файл модели.")
            return
        try:
            self.profile_manager.attach_model(
                self.profile_folder,
                self.pth_edit.text(),
                self.index_edit.text() or None,
            )
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось прикрепить модель: {e}")
            return
        self.accept()
