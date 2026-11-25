"""
device_manager.py

Работа с реальными аудио-устройствами через библиотеку sounddevice (PortAudio).

Отвечает за:
- перечисление доступных устройств ввода/вывода;
- определение, установлен ли виртуальный аудио-драйвер (например, VB-Audio
  Virtual Cable / VB-CABLE), в который можно направить обработанный звук,
  чтобы он появился как микрофон в Discord/Zoom/OBS и т.п.;
- проверку доступности устройства перед стартом (устройство могло быть
  отключено между запуском программы и нажатием Start).

ВАЖНО про "виртуальный микрофон":
Python и PortAudio НЕ МОГУТ сами создать новое системное аудио-устройство
в Windows. Для этого нужен отдельный виртуальный audio driver уровня ОС
(kernel-mode driver), например:

    - VB-CABLE (бесплатный, https://vb-audio.com/Cable/)
    - VoiceMeeter (тоже от VB-Audio, добавляет микшер)

Программа не может и не должна пытаться имитировать это программно —
вместо этого она просто ищет среди устройств вывода такие, которые похожи
на уже установленный виртуальный кабель, и подсказывает пользователю, что
нужно установить, если ничего подходящего не найдено.
"""

from dataclasses import dataclass


# Подстроки в названиях устройств, которые обычно означают, что это
# виртуальный audio cable, а не физическая звуковая карта.
_KNOWN_VIRTUAL_DEVICE_HINTS = [
    "cable",            # VB-CABLE ("CABLE Input", "CABLE Output")
    "voicemeeter",      # VoiceMeeter (VAIO, AUX, VAIO3)
    "virtual audio",    # общее название некоторых драйверов
    "virtual-audio",
    "vac",              # Virtual Audio Cable (E-MU / Muzychenko)
]


@dataclass
class AudioDevice:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    is_likely_virtual: bool


class DeviceManagerError(Exception):
    pass


class DeviceManager:
    def __init__(self):
        try:
            import sounddevice as sd
        except OSError as e:
            # Обычно означает, что не найдена системная библиотека PortAudio
            raise DeviceManagerError(
                "Не удалось инициализировать аудиоподсистему (PortAudio). "
                "Проверьте, что звуковые драйверы установлены корректно."
            ) from e
        self._sd = sd

    def _looks_virtual(self, name: str) -> bool:
        lowered = name.lower()
        return any(hint in lowered for hint in _KNOWN_VIRTUAL_DEVICE_HINTS)

    def list_input_devices(self) -> list:
        return self._list_devices(want_input=True)

    def list_output_devices(self) -> list:
        return self._list_devices(want_input=False)

    def _list_devices(self, want_input: bool) -> list:
        try:
            raw_devices = self._sd.query_devices()
        except Exception as e:
            raise DeviceManagerError(f"Не удалось получить список аудио-устройств: {e}") from e

        result = []
        for idx, dev in enumerate(raw_devices):
            channels = dev.get("max_input_channels", 0) if want_input else dev.get("max_output_channels", 0)
            if channels <= 0:
                continue
            result.append(
                AudioDevice(
                    index=idx,
                    name=dev.get("name", f"Device {idx}"),
                    max_input_channels=dev.get("max_input_channels", 0),
                    max_output_channels=dev.get("max_output_channels", 0),
                    default_samplerate=dev.get("default_samplerate", 44100.0),
                    is_likely_virtual=self._looks_virtual(dev.get("name", "")),
                )
            )
        return result

    def find_virtual_output_candidates(self) -> list:
        """Возвращает устройства вывода, похожие на установленный виртуальный кабель."""
        return [d for d in self.list_output_devices() if d.is_likely_virtual]

    def default_input_device(self):
        try:
            idx = self._sd.default.device[0]
            if idx is None or idx < 0:
                return None
            return idx
        except Exception:
            return None

    def default_output_device(self):
        try:
            idx = self._sd.default.device[1]
            if idx is None or idx < 0:
                return None
            return idx
        except Exception:
            return None

    def device_exists(self, index: int, want_input: bool) -> bool:
        try:
            dev = self._sd.query_devices(index)
        except Exception:
            return False
        channels = dev.get("max_input_channels", 0) if want_input else dev.get("max_output_channels", 0)
        return channels > 0

    def get_device_info(self, index: int) -> dict:
        try:
            return self._sd.query_devices(index)
        except Exception as e:
            raise DeviceManagerError(f"Устройство #{index} недоступно: {e}") from e
