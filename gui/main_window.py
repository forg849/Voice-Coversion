"""
main_window.py

Главное окно программы (раздел 2 ТЗ): список voice-профилей слева,
детали/слайдеры выбранного профиля и Test Voice справа, блок управления
real-time конвертацией (input/output устройства, level meter, Start/Stop,
latency) снизу.

Тяжёлые операции (анализ аудио, загрузка модели, запись Test Voice) не
блокируют GUI-поток - см. использование QThread в voice_profile_window.py
и _TestRecordWorker/_TestConvertWorker ниже.
"""

import os
import json
import tempfile
import numpy as np
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QPushButton, QComboBox, QMessageBox, QSystemTrayIcon, QMenu,
    QGroupBox, QInputDialog, QFileDialog, QSplitter
)
from PyQt6.QtGui import QIcon, QAction, QPixmap, QPainter, QColor
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal

from app.audio_engine import RealtimeAudioPipeline, AudioEngineError
from app.device_manager import DeviceManager, DeviceManagerError
from app.effects import EffectsChain, pitch_shift
from app.model_manager import ModelManager
from app.voice_converter import VoiceConverter, VoiceConverterError
from profiles.profile_manager import ProfileManager, ProfileError

from gui.widgets import LabeledSlider, LevelMeter, LatencyIndicator
from gui.voice_profile_window import ImportVoiceDialog, AttachModelDialog
from gui.settings_window import SettingsWindow


def _make_tray_icon() -> QIcon:
    """Генерирует простую иконку программно, чтобы не требовать внешний .ico файл."""
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setBrush(QColor("#7C4DFF"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(4, 4, 56, 56)
    painter.setBrush(QColor("#FFFFFF"))
    painter.drawRoundedRect(26, 16, 12, 24, 6, 6)
    painter.drawRect(30, 40, 4, 10)
    painter.end()
    return QIcon(pixmap)


class _TestRecordWorker(QThread):
    finished_ok = pyqtSignal(str)  # путь к записанному wav
    finished_error = pyqtSignal(str)

    def __init__(self, input_device, sample_rate: int, duration_sec: float = 4.0):
        super().__init__()
        self.input_device = input_device
        self.sample_rate = sample_rate
        self.duration_sec = duration_sec

    def run(self):
        try:
            import sounddevice as sd
            import soundfile as sf
            recording = sd.rec(
                int(self.duration_sec * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=self.input_device,
            )
            sd.wait()
            out_path = os.path.join(tempfile.gettempdir(), "vcs_test_original.wav")
            sf.write(out_path, recording, self.sample_rate)
            self.finished_ok.emit(out_path)
        except Exception as e:
            self.finished_error.emit(f"Не удалось записать тестовую фразу: {e}")


class _TestConvertWorker(QThread):
    finished_ok = pyqtSignal(str)
    finished_error = pyqtSignal(str)

    def __init__(self, voice_converter, effects_chain, wav_path: str, sample_rate: int,
                 pitch: float, conversion_strength: float):
        super().__init__()
        self.voice_converter = voice_converter
        self.effects_chain = effects_chain
        self.wav_path = wav_path
        self.sample_rate = sample_rate
        self.pitch = pitch
        self.conversion_strength = conversion_strength

    def run(self):
        try:
            import soundfile as sf
            audio, sr = sf.read(self.wav_path, dtype="float32")
            if self.voice_converter.has_loaded_model():
                converted = self.voice_converter.convert(
                    audio, sr,
                    pitch_shift_semitones=self.pitch,
                    conversion_strength=self.conversion_strength,
                )
            else:
                converted = pitch_shift(audio, sr, self.pitch)
            converted = self.effects_chain.process(converted, sr)
            energy_gain = 10 ** (self.energy_slider.value() / 20.0)
            converted = np.clip(converted * energy_gain, -1.0, 1.0).astype(np.float32)
            out_path = os.path.join(tempfile.gettempdir(), "vcs_test_converted.wav")
            sf.write(out_path, converted, sr)
            self.finished_ok.emit(out_path)
        except VoiceConverterError as e:
            self.finished_error.emit(str(e))
        except Exception as e:
            self.finished_error.emit(f"Ошибка тестовой конвертации: {e}")


DARK_STYLESHEET = """
QMainWindow, QDialog { background-color: #1E1F26; color: #E6E6E6; }
QWidget { background-color: #1E1F26; color: #E6E6E6; font-size: 13px; }
QListWidget { background-color: #262832; border: 1px solid #35374A; border-radius: 6px; }
QListWidget::item { padding: 8px; }
QListWidget::item:selected { background-color: #7C4DFF; color: white; border-radius: 4px; }
QGroupBox { border: 1px solid #35374A; border-radius: 6px; margin-top: 10px; padding-top: 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #B8B8D1; }
QPushButton { background-color: #35374A; border: 1px solid #454761; border-radius: 6px; padding: 6px 14px; }
QPushButton:hover { background-color: #454761; }
QPushButton#startButton { background-color: #4CAF50; font-weight: 600; }
QPushButton#startButton:hover { background-color: #5CC760; }
QPushButton#stopButton { background-color: #F44336; font-weight: 600; }
QPushButton#stopButton:hover { background-color: #FF5A4D; }
QComboBox, QLineEdit, QTextEdit, QSpinBox { background-color: #262832; border: 1px solid #35374A; border-radius: 4px; padding: 4px; }
QLabel#warningLabel { color: #FFC107; padding: 6px; background-color: #332B12; border-radius: 6px; }
QProgressBar#levelMeter { background-color: #262832; border: 1px solid #35374A; border-radius: 6px; }
QProgressBar#levelMeter::chunk { background-color: #7C4DFF; border-radius: 6px; }
QLabel#latencyIndicator { font-size: 13px; }
"""


class MainWindow(QMainWindow):
    def __init__(self, config_manager, base_dir: str):
        super().__init__()
        self.config_manager = config_manager
        self.base_dir = Path(base_dir)

        self.profile_manager = ProfileManager(str(self.base_dir / "data" / "voices"))
        self.model_manager = ModelManager(str(self.base_dir / "data" / "models"))
        self.effects_chain = EffectsChain()

        device_mode = self.config_manager.get("processing", "device_mode", "auto")
        self.voice_converter = VoiceConverter(self.model_manager, device_mode=device_mode)

        self.pipeline = None
        self.current_profile = None
        self._test_original_path = None

        self.setWindowTitle("Voice Conversion Studio")
        self.setMinimumSize(880, 560)
        self.setStyleSheet(DARK_STYLESHEET)

        self._build_ui()
        self._build_tray_icon()
        self._reload_profile_list()
        self._refresh_device_combos()

        self._level_timer = QTimer(self)
        self._level_timer.timeout.connect(self._on_level_timer)
        self._level_timer.start(80)
        self._latest_level = 0.0

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        title = QLabel("VOICE CONVERSION STUDIO")
        title.setStyleSheet("font-size: 18px; font-weight: 700; padding: 4px;")
        root_layout.addWidget(title)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter, stretch=1)

        # -- Левая колонка: список профилей -- #
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.addWidget(QLabel("VOICE PROFILES"))

        self.profile_list = QListWidget()
        self.profile_list.currentItemChanged.connect(self._on_profile_selected)
        left_layout.addWidget(self.profile_list, stretch=1)

        profile_buttons_row1 = QHBoxLayout()
        import_btn = QPushButton("+ Import")
        import_btn.clicked.connect(self._on_import_voice)
        rename_btn = QPushButton("Rename")
        rename_btn.clicked.connect(self._on_rename_profile)
        profile_buttons_row1.addWidget(import_btn)
        profile_buttons_row1.addWidget(rename_btn)
        left_layout.addLayout(profile_buttons_row1)

        profile_buttons_row2 = QHBoxLayout()
        duplicate_btn = QPushButton("Duplicate")
        duplicate_btn.clicked.connect(self._on_duplicate_profile)
        delete_btn = QPushButton("Delete")
        delete_btn.clicked.connect(self._on_delete_profile)
        profile_buttons_row2.addWidget(duplicate_btn)
        profile_buttons_row2.addWidget(delete_btn)
        left_layout.addLayout(profile_buttons_row2)

        profile_buttons_row3 = QHBoxLayout()
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self._on_export_profile)
        import_profile_btn = QPushButton("Import Profile")
        import_profile_btn.clicked.connect(self._on_import_profile_archive)
        profile_buttons_row3.addWidget(export_btn)
        profile_buttons_row3.addWidget(import_profile_btn)
        left_layout.addLayout(profile_buttons_row3)

        attach_model_btn = QPushButton("Attach Optional Model...")
        attach_model_btn.clicked.connect(self._on_attach_model)
        left_layout.addWidget(attach_model_btn)

        splitter.addWidget(left_widget)

        # -- Правая колонка: детали профиля -- #
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        self.profile_title_label = QLabel("Выберите или импортируйте профиль")
        self.profile_title_label.setStyleSheet("font-size: 15px; font-weight: 600;")
        right_layout.addWidget(self.profile_title_label)

        self.profile_source_label = QLabel("")
        self.profile_samples_label = QLabel("")
        self.profile_status_label = QLabel("")
        right_layout.addWidget(self.profile_source_label)
        right_layout.addWidget(self.profile_samples_label)
        right_layout.addWidget(self.profile_status_label)

        sliders_group = QGroupBox("Voice Parameters")
        sliders_layout = QVBoxLayout(sliders_group)
        self.pitch_slider = LabeledSlider("Pitch", -12, 12, 0, step=1, suffix=" st")
        self.energy_slider = LabeledSlider("Energy", -10, 10, 0, step=1)
        self.conversion_slider = LabeledSlider("Conversion", 0, 100, 75, step=1, suffix="%")
        self.pitch_slider.valueChangedFloat.connect(self._on_voice_params_changed)
        self.energy_slider.valueChangedFloat.connect(self._on_voice_params_changed)
        self.conversion_slider.valueChangedFloat.connect(self._on_voice_params_changed)
        sliders_layout.addWidget(self.pitch_slider)
        sliders_layout.addWidget(self.energy_slider)
        sliders_layout.addWidget(self.conversion_slider)
        right_layout.addWidget(sliders_group)

        effects_group = QGroupBox("Voice Controls and Audio Effects")
        effects_layout = QVBoxLayout(effects_group)
        self._build_effects_controls(effects_layout)
        right_layout.addWidget(effects_group)

        test_group = QGroupBox("Test Voice")
        test_layout = QVBoxLayout(test_group)
        test_buttons_row = QHBoxLayout()
        self.record_test_btn = QPushButton("● RECORD (4 сек)")
        self.record_test_btn.clicked.connect(self._on_record_test)
        self.play_original_btn = QPushButton("▶ Play Original")
        self.play_original_btn.setEnabled(False)
        self.play_original_btn.clicked.connect(lambda: self._play_wav(self._test_original_path))
        self.play_converted_btn = QPushButton("▶ Play Converted")
        self.play_converted_btn.setEnabled(False)
        self.play_converted_btn.clicked.connect(lambda: self._play_wav(self._test_converted_path))
        test_buttons_row.addWidget(self.record_test_btn)
        test_buttons_row.addWidget(self.play_original_btn)
        test_buttons_row.addWidget(self.play_converted_btn)
        test_layout.addLayout(test_buttons_row)
        self.test_status_label = QLabel("")
        test_layout.addWidget(self.test_status_label)
        right_layout.addWidget(test_group)

        right_layout.addStretch(1)
        splitter.addWidget(right_widget)
        splitter.setSizes([260, 560])

        # -- Нижняя панель: устройства + запуск -- #
        bottom_group = QGroupBox("Real-time Conversion")
        bottom_layout = QVBoxLayout(bottom_group)

        devices_row = QHBoxLayout()
        devices_row.addWidget(QLabel("Input:"))
        self.input_device_combo = QComboBox()
        devices_row.addWidget(self.input_device_combo, stretch=1)
        devices_row.addWidget(QLabel("Output:"))
        self.output_device_combo = QComboBox()
        devices_row.addWidget(self.output_device_combo, stretch=1)
        settings_btn = QPushButton("Settings")
        settings_btn.clicked.connect(self._on_open_settings)
        devices_row.addWidget(settings_btn)
        bottom_layout.addLayout(devices_row)

        level_row = QHBoxLayout()
        level_row.addWidget(QLabel("Input Level"))
        self.level_meter = LevelMeter()
        level_row.addWidget(self.level_meter, stretch=1)
        bottom_layout.addLayout(level_row)

        controls_row = QHBoxLayout()
        self.start_btn = QPushButton("START")
        self.start_btn.setObjectName("startButton")
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setObjectName("stopButton")
        self.stop_btn.clicked.connect(self._on_stop)
        self.stop_btn.setEnabled(False)
        self.latency_label = LatencyIndicator()
        self.gpu_status_label = QLabel("")
        controls_row.addWidget(self.start_btn)
        controls_row.addWidget(self.stop_btn)
        controls_row.addStretch(1)
        controls_row.addWidget(self.gpu_status_label)
        controls_row.addWidget(self.latency_label)
        bottom_layout.addLayout(controls_row)

        root_layout.addWidget(bottom_group)

        self._test_converted_path = None
        self._update_gpu_status_label()

    def _build_effects_controls(self, layout: QVBoxLayout):
        from PyQt6.QtWidgets import QCheckBox

        def add_simple_toggle(effect_key, title):
            cb = QCheckBox(title)
            cb.toggled.connect(lambda checked, k=effect_key: self.effects_chain.set_enabled(k, checked))
            layout.addWidget(cb)
            return cb

        self.effect_checkboxes = {
            "noise_reduction": add_simple_toggle("noise_reduction", "Noise Reduction"),
            "reverb": add_simple_toggle("reverb", "Reverb"),
            "echo": add_simple_toggle("echo", "Echo"),
            "distortion": add_simple_toggle("distortion", "Distortion"),
            "equalizer": add_simple_toggle("equalizer", "Equalizer"),
            "compressor": add_simple_toggle("compressor", "Compressor"),
        }
        note = QLabel("Pitch/Formant регулируются слайдерами выше (Voice Parameters).")
        note.setWordWrap(True)
        layout.addWidget(note)

    def _build_tray_icon(self):
        self.tray_icon = QSystemTrayIcon(_make_tray_icon(), self)
        self.tray_icon.setToolTip("Voice Conversion Studio")

        menu = QMenu()
        show_action = QAction("Show", self)
        show_action.triggered.connect(self._on_tray_show)
        start_action = QAction("Start Voice Conversion", self)
        start_action.triggered.connect(self._on_start)
        stop_action = QAction("Stop Voice Conversion", self)
        stop_action.triggered.connect(self._on_stop)
        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self._on_open_settings)
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self._on_tray_exit)

        menu.addAction(show_action)
        menu.addAction(start_action)
        menu.addAction(stop_action)
        menu.addSeparator()
        menu.addAction(settings_action)
        menu.addSeparator()
        menu.addAction(exit_action)

        self.tray_icon.setContextMenu(menu)
        self.tray_icon.activated.connect(self._on_tray_activated)
        self.tray_icon.show()

    # ------------------------------------------------------------------ #
    # Profiles
    # ------------------------------------------------------------------ #

    def _reload_profile_list(self):
        self.profile_list.clear()
        for meta in self.profile_manager.list_profiles():
            item = QListWidgetItem(self._profile_display_text(meta))
            item.setData(Qt.ItemDataRole.UserRole, meta["folder"])
            self.profile_list.addItem(item)

    @staticmethod
    def _profile_display_text(meta: dict) -> str:
        icon = "🎙" if meta.get("model_type") == "none" else ("🤖" if meta.get("model_type") == "builtin" else "👤")
        return f"{icon} {meta.get('display_name', meta.get('folder'))}"

    def _selected_profile_folder(self):
        item = self.profile_list.currentItem()
        if item is None:
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    def _on_profile_selected(self, current, previous):
        folder = current.data(Qt.ItemDataRole.UserRole) if current else None
        if not folder:
            return
        try:
            meta = self.profile_manager.get_profile(folder)
        except ProfileError as e:
            QMessageBox.warning(self, "Voice Conversion Studio", str(e))
            return

        self.current_profile = meta
        self.profile_title_label.setText(meta.get("display_name", folder))
        self.profile_source_label.setText(f"Source: {meta.get('source_filename', '—')}")
        analysis = meta.get("analysis", {})
        self.profile_samples_label.setText(f"Duration: {analysis.get('duration_sec', '—')} sec")

        if meta.get("model_type") == "none":
            status = "Status: режим эффектов готов, модель не требуется"
        else:
            status = "Status: Ready"
        self.profile_status_label.setText(status)

        settings = meta.get("settings", {})
        self.pitch_slider.set_value(settings.get("pitch", 0))
        self.energy_slider.set_value(settings.get("energy", 0))
        self.conversion_slider.set_value(settings.get("conversion_strength", 0.75) * 100)

        self.voice_converter.unload_model()
        self._try_load_model_for_profile(meta)

    def _try_load_model_for_profile(self, meta: dict):
        folder = meta["folder"]
        if meta.get("model_type") != "rvc":
            return
        model_path = self.profile_manager.profile_model_path(folder)
        index_path = self.profile_manager.profile_index_path(folder)
        try:
            self.voice_converter.load_model(
                str(model_path),
                str(index_path) if index_path.exists() else None,
            )
        except VoiceConverterError as e:
            QMessageBox.warning(self, "Voice Conversion Studio", f"Модель не загружена:\n{e}")

    def _on_voice_params_changed(self, *_):
        if not self.current_profile:
            return
        folder = self.current_profile["folder"]
        settings = {
            "pitch": self.pitch_slider.value(),
            "energy": self.energy_slider.value(),
            "conversion_strength": self.conversion_slider.value() / 100.0,
        }
        try:
            self.current_profile = self.profile_manager.update_settings(folder, settings)
        except ProfileError:
            pass

    def _on_import_voice(self):
        dialog = ImportVoiceDialog(self.profile_manager, self)
        if dialog.exec():
            self._reload_profile_list()

    def _on_rename_profile(self):
        folder = self._selected_profile_folder()
        if not folder:
            return
        current_name = self.current_profile.get("display_name", "") if self.current_profile else ""
        new_name, ok = QInputDialog.getText(self, "Rename Voice", "Новое имя:", text=current_name)
        if ok and new_name.strip():
            self.profile_manager.rename_profile(folder, new_name.strip())
            self._reload_profile_list()

    def _on_duplicate_profile(self):
        folder = self._selected_profile_folder()
        if not folder:
            return
        current_name = self.current_profile.get("display_name", folder) if self.current_profile else folder
        new_name, ok = QInputDialog.getText(self, "Duplicate Voice", "Имя копии:", text=f"{current_name} Copy")
        if ok and new_name.strip():
            try:
                self.profile_manager.duplicate_profile(folder, new_name.strip())
            except ProfileError as e:
                QMessageBox.warning(self, "Voice Conversion Studio", str(e))
                return
            self._reload_profile_list()

    def _on_delete_profile(self):
        folder = self._selected_profile_folder()
        if not folder:
            return
        confirm = QMessageBox.question(
            self, "Delete Voice",
            "Удалить выбранный голосовой профиль без возможности восстановления?",
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.profile_manager.delete_profile(folder)
            self.current_profile = None
            self._reload_profile_list()

    def _on_export_profile(self):
        folder = self._selected_profile_folder()
        if not folder:
            return
        dest, _ = QFileDialog.getSaveFileName(self, "Export Profile", f"{folder}.zip", "ZIP архив (*.zip)")
        if not dest:
            return
        try:
            self.profile_manager.export_profile(folder, dest)
            QMessageBox.information(self, "Voice Conversion Studio", "Профиль экспортирован.")
        except ProfileError as e:
            QMessageBox.warning(self, "Voice Conversion Studio", str(e))

    def _on_import_profile_archive(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Profile", "", "ZIP архив (*.zip)")
        if not path:
            return
        try:
            self.profile_manager.import_profile(path)
        except ProfileError as e:
            QMessageBox.warning(self, "Voice Conversion Studio", str(e))
            return
        self._reload_profile_list()

    def _on_attach_model(self):
        folder = self._selected_profile_folder()
        if not folder:
            QMessageBox.information(self, "Voice Conversion Studio", "Сначала выберите профиль слева.")
            return
        dialog = AttachModelDialog(self.profile_manager, folder, self)
        if dialog.exec():
            self._reload_profile_list()
            self._on_profile_selected(self.profile_list.currentItem(), None)

    # ------------------------------------------------------------------ #
    # Test Voice
    # ------------------------------------------------------------------ #

    def _on_record_test(self):
        input_device = self.input_device_combo.currentData()
        sample_rate = self.config_manager.get("audio", "sample_rate", 44100)
        self.test_status_label.setText("Запись... говорите в микрофон (4 сек)")
        self.record_test_btn.setEnabled(False)

        self._record_worker = _TestRecordWorker(input_device, sample_rate)
        self._record_worker.finished_ok.connect(self._on_test_recorded)
        self._record_worker.finished_error.connect(self._on_test_error)
        self._record_worker.start()

    def _on_test_recorded(self, wav_path: str):
        self._test_original_path = wav_path
        self.play_original_btn.setEnabled(True)
        self.record_test_btn.setEnabled(True)

        if not self.voice_converter.is_ready():
            self.test_status_label.setText(f"Записано. {self.voice_converter.readiness_report()}")
            return
        if self.current_profile is None or self.current_profile.get("model_type") != "rvc":
            self.test_status_label.setText(
                "Записано. Для этого профиля не прикреплена модель — конвертация недоступна "
                "(Attach RVC Model)."
            )
            return

        self.test_status_label.setText("Конвертация...")
        sample_rate = self.config_manager.get("audio", "sample_rate", 44100)
        self._convert_worker = _TestConvertWorker(
            self.voice_converter, self.effects_chain, wav_path, sample_rate,
            pitch=self.pitch_slider.value(),
            conversion_strength=self.conversion_slider.value() / 100.0,
        )
        self._convert_worker.finished_ok.connect(self._on_test_converted)
        self._convert_worker.finished_error.connect(self._on_test_error)
        self._convert_worker.start()

    def _on_test_converted(self, wav_path: str):
        self._test_converted_path = wav_path
        self.play_converted_btn.setEnabled(True)
        self.test_status_label.setText("Готово.")

    def _on_test_error(self, message: str):
        self.record_test_btn.setEnabled(True)
        self.test_status_label.setText(f"Ошибка: {message}")

    def _play_wav(self, path):
        if not path or not os.path.exists(path):
            return
        try:
            import sounddevice as sd
            import soundfile as sf
            data, sr = sf.read(path, dtype="float32")
            output_device = self.output_device_combo.currentData()
            sd.play(data, sr, device=output_device)
        except Exception as e:
            QMessageBox.warning(self, "Voice Conversion Studio", f"Не удалось воспроизвести: {e}")

    # ------------------------------------------------------------------ #
    # Real-time devices / pipeline
    # ------------------------------------------------------------------ #

    def _refresh_device_combos(self):
        try:
            dm = DeviceManager()
        except DeviceManagerError as e:
            QMessageBox.warning(self, "Voice Conversion Studio", str(e))
            return

        self.input_device_combo.clear()
        self.input_device_combo.addItem("По умолчанию", None)
        for dev in dm.list_input_devices():
            self.input_device_combo.addItem(dev.name, dev.index)
        saved_input = self.config_manager.get("audio", "input_device")
        idx = self.input_device_combo.findData(saved_input)
        if idx >= 0:
            self.input_device_combo.setCurrentIndex(idx)

        self.output_device_combo.clear()
        self.output_device_combo.addItem("По умолчанию", None)
        virtual_candidates = {d.index for d in dm.find_virtual_output_candidates()}
        for dev in dm.list_output_devices():
            label = dev.name + ("  [виртуальный кабель]" if dev.index in virtual_candidates else "")
            self.output_device_combo.addItem(label, dev.index)
        saved_output = self.config_manager.get("audio", "output_device")
        idx = self.output_device_combo.findData(saved_output)
        if idx >= 0:
            self.output_device_combo.setCurrentIndex(idx)

    def _update_gpu_status_label(self):
        gpu_info = self.model_manager.detect_gpu()
        if gpu_info["available"]:
            self.gpu_status_label.setText(f"GPU: {gpu_info['device_name']} ({gpu_info.get('vram_gb', '?')} GB)")
            self.gpu_status_label.setStyleSheet("color: #4CAF50;")
        else:
            self.gpu_status_label.setText("GPU acceleration unavailable. Using CPU mode.")
            self.gpu_status_label.setStyleSheet("color: #FFC107;")

    def _current_processing_fn(self):
        def process(chunk, sample_rate):
            pitch = self.pitch_slider.value()
            strength = self.conversion_slider.value() / 100.0
            if self.voice_converter.has_loaded_model():
                processed = self.voice_converter.convert(
                    chunk, sample_rate, pitch_shift_semitones=pitch, conversion_strength=strength
                )
            else:
                processed = pitch_shift(chunk, sample_rate, pitch)

            processed = self.effects_chain.process(processed, sample_rate)
            energy_gain = 10 ** (self.energy_slider.value() / 20.0)
            return np.clip(processed * energy_gain, -1.0, 1.0).astype(np.float32)
        return process

    def _on_start(self):
        if self.pipeline and self.pipeline.is_running:
            return
        sample_rate = self.config_manager.get("audio", "sample_rate", 44100)
        buffer_size = self.config_manager.get("audio", "buffer_size", 1024)
        chunk_size = max(buffer_size * 2, 2048)

        self.pipeline = RealtimeAudioPipeline(
            input_device=self.input_device_combo.currentData(),
            output_device=self.output_device_combo.currentData(),
            sample_rate=sample_rate,
            buffer_size=buffer_size,
            chunk_size=chunk_size,
            process_fn=self._current_processing_fn(),
            on_error=self._on_pipeline_error,
            on_level=self._on_pipeline_level,
        )
        try:
            self.pipeline.start()
        except AudioEngineError as e:
            QMessageBox.critical(self, "Voice Conversion Studio", str(e))
            self.pipeline = None
            return

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _on_stop(self):
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.latency_label.set_latency(0)

    def _on_pipeline_error(self, message: str):
        # Вызывается из рабочего потока - показываем в статусной строке, не блокируя поток модальным окном.
        self.test_status_label.setText(f"⚠ {message}")

    def _on_pipeline_level(self, rms: float):
        self._latest_level = rms

    def _on_level_timer(self):
        self.level_meter.set_level(self._latest_level)
        if self.pipeline and self.pipeline.is_running:
            self.latency_label.set_latency(self.pipeline.last_latency_ms)

    # ------------------------------------------------------------------ #
    # Settings / Tray / Window lifecycle
    # ------------------------------------------------------------------ #

    def _on_open_settings(self):
        dialog = SettingsWindow(self.config_manager, self)
        if dialog.exec():
            self._refresh_device_combos()
            self._update_gpu_status_label()

    def _on_tray_show(self):
        self.showNormal()
        self.activateWindow()

    def _on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self._on_tray_show()

    def _on_tray_exit(self):
        self._on_stop()
        self.tray_icon.hide()
        from PyQt6.QtWidgets import QApplication
        QApplication.quit()

    def closeEvent(self, event):
        if self.config_manager.get("interface", "minimize_to_tray", True):
            event.ignore()
            self.hide()
            self.tray_icon.showMessage(
                "Voice Conversion Studio",
                "Программа продолжает работать в трее.",
                QSystemTrayIcon.MessageIcon.Information,
                2000,
            )
        else:
            self._on_stop()
            event.accept()
