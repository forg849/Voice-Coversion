"""
settings_window.py

Окно настроек (раздел 12 ТЗ): Audio / Processing / Interface.
Все изменения сразу пишутся в ConfigManager и сохраняются на диск при
нажатии "Сохранить".
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QComboBox,
    QCheckBox, QPushButton, QSpinBox, QTabWidget, QWidget, QMessageBox
)

from app.device_manager import DeviceManager, DeviceManagerError


SAMPLE_RATES = [16000, 22050, 32000, 44100, 48000]
BUFFER_SIZES = [256, 512, 1024, 2048, 4096]


class SettingsWindow(QDialog):
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.setWindowTitle("Settings")
        self.setMinimumWidth(460)

        try:
            self.device_manager = DeviceManager()
            self._device_error = None
        except DeviceManagerError as e:
            self.device_manager = None
            self._device_error = str(e)

        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        tabs.addTab(self._build_audio_tab(), "Audio")
        tabs.addTab(self._build_processing_tab(), "Processing")
        tabs.addTab(self._build_interface_tab(), "Interface")

        buttons_row = QHBoxLayout()
        save_btn = QPushButton("Сохранить")
        save_btn.clicked.connect(self._on_save)
        cancel_btn = QPushButton("Отмена")
        cancel_btn.clicked.connect(self.reject)
        buttons_row.addStretch(1)
        buttons_row.addWidget(cancel_btn)
        buttons_row.addWidget(save_btn)
        layout.addLayout(buttons_row)

    # ------------------------------------------------------------------ #

    def _build_audio_tab(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)

        self.input_combo = QComboBox()
        self.output_combo = QComboBox()

        if self._device_error:
            form.addRow(QLabel(f"⚠ {self._device_error}"))
        else:
            self.input_combo.addItem("Системное устройство по умолчанию", None)
            for dev in self.device_manager.list_input_devices():
                self.input_combo.addItem(dev.name, dev.index)

            self.output_combo.addItem("Системное устройство по умолчанию", None)
            virtual_candidates = {d.index for d in self.device_manager.find_virtual_output_candidates()}
            for dev in self.device_manager.list_output_devices():
                label = dev.name + ("  [виртуальный кабель]" if dev.index in virtual_candidates else "")
                self.output_combo.addItem(label, dev.index)

            current_input = self.config_manager.get("audio", "input_device")
            current_output = self.config_manager.get("audio", "output_device")
            self._select_combo_by_data(self.input_combo, current_input)
            self._select_combo_by_data(self.output_combo, current_output)

            if not virtual_candidates:
                form.addRow(QLabel(
                    "⚠ Виртуальный audio-кабель не обнаружен. Чтобы выбрать обработанный "
                    "голос как микрофон в Discord/OBS, установите VB-CABLE "
                    "(https://vb-audio.com/Cable/) и выберите его здесь как Output."
                ))

        form.addRow("Input device:", self.input_combo)
        form.addRow("Output device:", self.output_combo)

        self.sample_rate_combo = QComboBox()
        for sr in SAMPLE_RATES:
            self.sample_rate_combo.addItem(f"{sr} Hz", sr)
        self._select_combo_by_data(self.sample_rate_combo, self.config_manager.get("audio", "sample_rate", 44100))
        form.addRow("Sample rate:", self.sample_rate_combo)

        self.buffer_size_combo = QComboBox()
        for bs in BUFFER_SIZES:
            self.buffer_size_combo.addItem(f"{bs} samples", bs)
        self._select_combo_by_data(self.buffer_size_combo, self.config_manager.get("audio", "buffer_size", 1024))
        form.addRow("Buffer size:", self.buffer_size_combo)

        return widget

    def _build_processing_tab(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)

        self.device_mode_combo = QComboBox()
        self.device_mode_combo.addItem("Auto (использовать GPU, если доступна)", "auto")
        self.device_mode_combo.addItem("CPU", "cpu")
        self.device_mode_combo.addItem("GPU (принудительно)", "gpu")
        self._select_combo_by_data(self.device_mode_combo, self.config_manager.get("processing", "device_mode", "auto"))
        form.addRow("Compute device:", self.device_mode_combo)

        self.quality_combo = QComboBox()
        self.quality_combo.addItem("Fast (меньше задержка, ниже качество)", "fast")
        self.quality_combo.addItem("Balanced", "balanced")
        self.quality_combo.addItem("High (выше качество, больше задержка)", "high")
        self._select_combo_by_data(self.quality_combo, self.config_manager.get("processing", "quality", "balanced"))
        form.addRow("Quality:", self.quality_combo)

        self.low_latency_checkbox = QCheckBox("Low latency mode (меньшие буферы, риск тресков на слабом ПК)")
        self.low_latency_checkbox.setChecked(bool(self.config_manager.get("processing", "low_latency_mode", False)))
        form.addRow(self.low_latency_checkbox)

        return widget

    def _build_interface_tab(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)

        self.dark_theme_checkbox = QCheckBox("Тёмная тема")
        self.dark_theme_checkbox.setChecked(bool(self.config_manager.get("interface", "dark_theme", True)))
        form.addRow(self.dark_theme_checkbox)

        self.start_minimized_checkbox = QCheckBox("Запускать свёрнутым")
        self.start_minimized_checkbox.setChecked(bool(self.config_manager.get("interface", "start_minimized", False)))
        form.addRow(self.start_minimized_checkbox)

        self.minimize_to_tray_checkbox = QCheckBox("Сворачивать в системный трей")
        self.minimize_to_tray_checkbox.setChecked(bool(self.config_manager.get("interface", "minimize_to_tray", True)))
        form.addRow(self.minimize_to_tray_checkbox)

        return widget

    # ------------------------------------------------------------------ #

    @staticmethod
    def _select_combo_by_data(combo: QComboBox, data_value):
        idx = combo.findData(data_value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _on_save(self):
        if not self._device_error:
            self.config_manager.set("audio", "input_device", self.input_combo.currentData())
            self.config_manager.set("audio", "output_device", self.output_combo.currentData())
        self.config_manager.set("audio", "sample_rate", self.sample_rate_combo.currentData())
        self.config_manager.set("audio", "buffer_size", self.buffer_size_combo.currentData())

        self.config_manager.set("processing", "device_mode", self.device_mode_combo.currentData())
        self.config_manager.set("processing", "quality", self.quality_combo.currentData())
        self.config_manager.set("processing", "low_latency_mode", self.low_latency_checkbox.isChecked())

        self.config_manager.set("interface", "dark_theme", self.dark_theme_checkbox.isChecked())
        self.config_manager.set("interface", "start_minimized", self.start_minimized_checkbox.isChecked())
        self.config_manager.set("interface", "minimize_to_tray", self.minimize_to_tray_checkbox.isChecked())

        try:
            self.config_manager.save()
        except OSError as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить настройки: {e}")
            return

        QMessageBox.information(self, "Voice Conversion Studio", "Настройки сохранены.\nНекоторые изменения применятся после перезапуска потока обработки (Stop -> Start).")
        self.accept()
