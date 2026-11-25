"""
widgets.py

Переиспользуемые виджеты для тёмного интерфейса Voice Conversion Studio:
    - LabeledSlider: слайдер с подписью и текущим значением (Pitch/Energy/Conversion и т.п.)
    - LevelMeter: горизонтальный индикатор уровня входного сигнала
    - LatencyIndicator: текстовый индикатор latency с цветовой индикацией
    - CollapsibleEffectRow: строка "чекбокс + слайдеры" для одного эффекта
"""

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSlider, QCheckBox, QProgressBar
)
from PyQt6.QtCore import Qt, pyqtSignal


class LabeledSlider(QWidget):
    valueChangedFloat = pyqtSignal(float)

    def __init__(self, label: str, min_value: float, max_value: float,
                 initial: float = 0.0, step: float = 1.0, suffix: str = "", parent=None):
        super().__init__(parent)
        self._min = min_value
        self._max = max_value
        self._step = step
        self._suffix = suffix
        self._scale = 1.0 / step if step < 1 else 1.0

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.name_label = QLabel(label)
        self.name_label.setFixedWidth(90)
        self.name_label.setObjectName("sliderName")

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(int(min_value * self._scale))
        self.slider.setMaximum(int(max_value * self._scale))
        self.slider.setValue(int(initial * self._scale))
        self.slider.valueChanged.connect(self._on_slider_changed)

        self.value_label = QLabel(self._format_value(initial))
        self.value_label.setFixedWidth(60)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(self.name_label)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.value_label)

    def _format_value(self, value: float) -> str:
        if self._step >= 1:
            return f"{int(round(value))}{self._suffix}"
        return f"{value:.2f}{self._suffix}"

    def _on_slider_changed(self, raw_int_value: int):
        real_value = raw_int_value / self._scale
        self.value_label.setText(self._format_value(real_value))
        self.valueChangedFloat.emit(real_value)

    def value(self) -> float:
        return self.slider.value() / self._scale

    def set_value(self, value: float):
        self.slider.blockSignals(True)
        self.slider.setValue(int(value * self._scale))
        self.value_label.setText(self._format_value(value))
        self.slider.blockSignals(False)


class LevelMeter(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(0, 100)
        self.setTextVisible(False)
        self.setFixedHeight(14)
        self.setObjectName("levelMeter")

    def set_level(self, rms: float):
        # RMS ~0.0-1.0 -> проценты, с лёгкой логарифмической шкалой для наглядности на тихой речи
        import math
        if rms <= 0:
            percent = 0
        else:
            db = 20 * math.log10(min(rms, 1.0) + 1e-9)
            percent = max(0, min(100, int((db + 50) * (100 / 50))))
        self.setValue(percent)


class LatencyIndicator(QLabel):
    def __init__(self, parent=None):
        super().__init__("Latency: -- ms", parent)
        self.setObjectName("latencyIndicator")

    def set_latency(self, ms: float):
        self.setText(f"Latency: {ms:.0f} ms")
        if ms < 60:
            color = "#4CAF50"
        elif ms < 150:
            color = "#FFC107"
        else:
            color = "#F44336"
        self.setStyleSheet(f"color: {color}; font-weight: 600;")


class CollapsibleEffectRow(QWidget):
    """Строка одного эффекта: [x] Название   [слайдеры параметров]."""

    toggled = pyqtSignal(bool)

    def __init__(self, effect_title: str, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(4)

        header = QHBoxLayout()
        self.checkbox = QCheckBox(effect_title)
        self.checkbox.toggled.connect(self.toggled.emit)
        header.addWidget(self.checkbox)
        header.addStretch(1)
        self._layout.addLayout(header)

        self.params_container = QVBoxLayout()
        self._layout.addLayout(self.params_container)

    def add_param_widget(self, widget: QWidget):
        self.params_container.addWidget(widget)

    def is_enabled(self) -> bool:
        return self.checkbox.isChecked()
