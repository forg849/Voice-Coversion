"""
model_manager.py

Отвечает за технологию voice conversion, выбранную для этого проекта: RVC
(Retrieval-based Voice Conversion), через пакет `rvc-python`.

ПОЧЕМУ ИМЕННО RVC (кратко; подробный разбор альтернатив — в README.md):
    - Открытый, бесплатный, активно поддерживаемый проект с готовыми
      предобученными весами (content encoder HuBERT, pitch extractor RMVPE).
    - Официально поддерживает как inference на CPU, так и на CUDA (NVIDIA GPU).
    - Единственная из рассмотренных технологий, у которой есть готовые
      Python-обёртки (`rvc-python`) и реальные near-real-time community-реализации
      под Windows (используется в проектах вроде w-okada's Real-Time Voice
      Changer, который тоже построен поверх RVC/so-vits-svc).
    - Позволяет "Fast Mode" (готовая модель, скачанная один раз) и
      "Custom Voice Mode" (тренировка/использование модели под конкретный
      голос), в отличие, например, от чистых pitch/formant DSP-эффектов,
      которые вообще не переносят тембр голоса.

ЧЕСТНОЕ ОГРАНИЧЕНИЕ (важно, см. также README и раздел 5/21 ТЗ):
    RVC-модель голоса (.pth) НЕЛЬЗЯ мгновенно и качественно создать из одного
    MP3-файла длиной в десятки секунд. Официальный тренировочный pipeline RVC
    требует:
        - от ~10 минут чистой речи одного диктора (лучше 30-60 минут) для
          приемлемого качества;
        - десятков-сотен эпох обучения на GPU (обычно от 20-40 минут на
          RTX-видеокарте среднего уровня и выше — не секунды и не минуты);
        - отдельного шага извлечения признаков (HuBERT) и pitch-разметки
          перед тренировкой.
    Эта программа НЕ имитирует обучение фейковым прогресс-баром. Вместо
    этого:
        - "Fast Mode" использует уже готовую (заранее обученную и
          распространяемую с разрешения авторов, либо предоставленную
          пользователем) .pth модель, которую нужно просто положить в
          data/models/pretrained/;
        - "Custom Voice Mode" в этом проекте реализован как ИНТЕГРАЦИЯ
          с внешним, отдельно запускаемым тренировочным пайплайном RVC
          (RVC WebUI / rvc-python trainer CLI) — сама программа проверяет
          наличие обученной модели и подключает её к профилю
          (ProfileManager.attach_model), но не проводит обучение внутри
          своего GUI, потому что честная тренировка занимает от десятков
          минут до нескольких часов и требует значительно больше вычислений,
          чем разумно скрывать за один клик в десктоп-программе.
"""

import os
from pathlib import Path


class ModelManagerError(Exception):
    pass


# Файлы, обязательные для работы RVC pipeline (content encoder + pitch extractor).
# Эти веса распространяются проектом RVC отдельно от pip-пакета (лицензионно
# и по размеру - десятки/сотни МБ - их не кладут в PyPI wheel).
REQUIRED_SHARED_MODELS = {
    "hubert_base.pt": "Content encoder (HuBERT) - извлекает лингвистическое содержимое речи.",
    "rmvpe.pt": "Pitch extractor (RMVPE) - извлекает высоту основного тона для сохранения интонации.",
}


class ModelManager:
    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self.shared_dir = self.models_dir / "shared"
        self.pretrained_dir = self.models_dir / "pretrained"
        self.shared_dir.mkdir(parents=True, exist_ok=True)
        self.pretrained_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #

    def check_shared_models(self) -> dict:
        """
        Проверяет, скачаны ли обязательные общие веса (hubert_base.pt, rmvpe.pt).
        Возвращает {filename: bool_present}.
        """
        status = {}
        for filename in REQUIRED_SHARED_MODELS:
            status[filename] = (self.shared_dir / filename).exists()
        return status

    def missing_shared_models(self) -> list:
        status = self.check_shared_models()
        return [name for name, present in status.items() if not present]

    def shared_models_ready(self) -> bool:
        return len(self.missing_shared_models()) == 0

    def get_setup_instructions(self) -> str:
        missing = self.missing_shared_models()
        if not missing:
            return "Все необходимые базовые модели уже установлены."

        lines = [
            "Для работы Voice Conversion необходимо один раз скачать базовые модели RVC "
            "(они не входят в дистрибутив программы из-за размера и лицензии):",
            "",
        ]
        for name in missing:
            lines.append(f"  - {name}: {REQUIRED_SHARED_MODELS[name]}")
        lines.append("")
        lines.append(f"Поместите скачанные файлы в папку: {self.shared_dir}")
        lines.append(
            "Актуальные ссылки для загрузки указаны в README.md проекта "
            "(официальный репозиторий RVC-Project / Retrieval-based-Voice-Conversion-WebUI)."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------ #

    def list_pretrained_voice_models(self) -> list:
        """Список готовых (Fast Mode) .pth моделей, доступных всем профилям."""
        if not self.pretrained_dir.exists():
            return []
        return sorted(p.name for p in self.pretrained_dir.glob("*.pth"))

    def pretrained_model_path(self, filename: str) -> Path:
        path = self.pretrained_dir / filename
        if not path.exists():
            raise ModelManagerError(f"Модель '{filename}' не найдена в {self.pretrained_dir}")
        return path

    # ------------------------------------------------------------------ #

    def detect_gpu(self) -> dict:
        """
        Проверяет доступность CUDA через torch, если он установлен.
        Возвращает {available: bool, device_name: str|None, reason: str|None}.
        Никогда не бросает исключение - используется для честного отображения
        статуса GPU в интерфейсе (раздел 14 ТЗ).
        """
        try:
            import torch
        except ImportError:
            return {"available": False, "device_name": None, "reason": "Библиотека torch не установлена."}

        try:
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                return {
                    "available": True,
                    "device_name": name,
                    "vram_gb": round(total_vram_gb, 1),
                    "reason": None,
                }
            return {"available": False, "device_name": None, "reason": "CUDA-совместимая NVIDIA GPU не обнаружена."}
        except Exception as e:
            return {"available": False, "device_name": None, "reason": f"Ошибка проверки GPU: {e}"}
