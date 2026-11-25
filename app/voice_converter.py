"""
voice_converter.py

Обёртка над реальным voice conversion backend'ом (RVC, через пакет
`rvc-python`, см. подробное обоснование выбора в model_manager.py и README.md).

Класс VoiceConverter НЕ содержит никакой "фейковой" обработки — если backend
недоступен (не установлена библиотека, нет базовых весов, нет модели у
профиля), он явно и честно сообщает об этом через VoiceConverterError, и
main_window.py обязан показать пользователю понятное сообщение, а не молча
подсовывать необработанный или эффектами-замаскированный звук как "voice
conversion".

Режимы:
    Fast Mode      -> используется одна из готовых моделей из
                       data/models/pretrained/ (общая для всех, кто её выберет).
    Custom Voice    -> используется модель, прикреплённая к конкретному
                       профилю через ProfileManager.attach_model()
                       (обучена заранее внешним RVC-тренером).

Если для профиля модель ещё не прикреплена (model_type == "none"), метод
convert() бросает VoiceConverterError с понятным объяснением - это
соответствует требованию "не имитируй, а объясняй ограничения".
"""

import numpy as np
import sys
from pathlib import Path


class VoiceConverterError(Exception):
    pass


class VoiceConverter:
    def __init__(self, model_manager, device_mode: str = "auto"):
        self.model_manager = model_manager
        self.device_mode = device_mode  # "auto" | "cpu" | "gpu"
        self._rvc_pipeline = None
        self._backend_name = None
        self._pipeline_sample_rate = None
        self._loaded_model_path = None
        self._backend_available, self._backend_error = self._check_backend()

    # ------------------------------------------------------------------ #

    def _check_backend(self):
        try:
            import rvc_python  # noqa: F401
            self._backend_name = "rvc"
            return True, None
        except ImportError as rvc_error:
            try:
                import so_vits_svc_fork  # noqa: F401
                self._backend_name = "so-vits-svc"
                return True, None
            except ImportError as svc_error:
                missing = getattr(svc_error, "name", None) or getattr(rvc_error, "name", "unknown")
                return False, (
                    "Доступный backend voice conversion не найден. Для Python 3.14 "
                    "установите альтернативу командой 'python -m pip install "
                    "so-vits-svc-fork'. "
                    f"Сейчас не хватает зависимости '{missing}'. "
                    f"Интерпретатор: {sys.executable}. "
                    f"Подробность импорта: {svc_error}. "
                    "RVC-backend через rvc-python требует отдельное окружение Python 3.10/3.11."
                )

    def _resolve_torch_device(self) -> str:
        if self.device_mode == "cpu":
            return "cpu"
        gpu_info = self.model_manager.detect_gpu()
        if self.device_mode == "gpu":
            if not gpu_info["available"]:
                raise VoiceConverterError(
                    f"Режим GPU выбран принудительно, но GPU недоступна: {gpu_info['reason']}"
                )
            return "cuda:0"
        # auto
        return "cuda:0" if gpu_info["available"] else "cpu"

    # ------------------------------------------------------------------ #

    def is_ready(self) -> bool:
        return self._backend_available and (
            self._backend_name == "so-vits-svc" or self.model_manager.shared_models_ready()
        )

    def has_loaded_model(self) -> bool:
        return self._rvc_pipeline is not None

    def readiness_report(self) -> str:
        if not self._backend_available:
            return self._backend_error
        if self._backend_name != "so-vits-svc" and not self.model_manager.shared_models_ready():
            return self.model_manager.get_setup_instructions()
        if self._backend_name == "so-vits-svc":
            return "Backend so-vits-svc готов. Для модели нужен .pth и одноимённый .json-конфиг."
        return "Backend RVC готов к работе."

    # ------------------------------------------------------------------ #

    def load_model(self, model_pth_path: str, index_path: str = None):
        """
        Загружает конкретную .pth (+опционально .index) модель в память.
        Реальная загрузка выполняется через rvc_python; это тяжёлая операция
        (сотни МБ на GPU/CPU) - не вызывайте её из audio callback, только
        заранее, при выборе профиля пользователем.
        """
        if not self._backend_available:
            raise VoiceConverterError(self._backend_error)
        if self._backend_name != "so-vits-svc" and not self.model_manager.shared_models_ready():
            raise VoiceConverterError(self.model_manager.get_setup_instructions())

        if self._loaded_model_path == model_pth_path and self._rvc_pipeline is not None:
            return  # уже загружена именно эта модель

        torch_device = self._resolve_torch_device()

        if self._backend_name == "so-vits-svc":
            return self._load_so_vits_model(model_pth_path, torch_device)

        try:
            from rvc_python.infer import RVCInference
        except ImportError as e:
            raise VoiceConverterError(
                "Не удалось импортировать RVCInference из rvc_python. "
                "Для Python 3.14 используйте модель so-vits-svc с JSON-конфигурацией."
            ) from e

        try:
            pipeline = RVCInference(
                model_path=str(model_pth_path),
                index_path=str(index_path) if index_path else None,
                device=torch_device,
                hubert_path=str(self.model_manager.shared_dir / "hubert_base.pt"),
                rmvpe_path=str(self.model_manager.shared_dir / "rmvpe.pt"),
            )
        except TypeError:
            # У разных версий rvc-python отличается сигнатура конструктора -
            # пробуем более старый/иной вариант API вместо падения программы.
            pipeline = RVCInference(str(model_pth_path))
            if hasattr(pipeline, "load_model"):
                pipeline.load_model(str(model_pth_path))
        except Exception as e:
            raise VoiceConverterError(
                f"Не удалось загрузить модель '{model_pth_path}': {e}"
            ) from e

        self._rvc_pipeline = pipeline
        self._loaded_model_path = model_pth_path

    def _load_so_vits_model(self, model_pth_path: str, torch_device: str):
        config_path = Path(model_pth_path).with_suffix(".json")
        if not config_path.exists():
            raise VoiceConverterError(
                "Для so-vits-svc рядом с моделью нужен JSON-конфиг с таким же именем: "
                f"{config_path}"
            )
        try:
            from so_vits_svc_fork.inference.core import Svc
            pipeline = Svc(
                net_g_path=model_pth_path,
                config_path=str(config_path),
                device=torch_device,
                half=False,
            )
        except Exception as error:
            raise VoiceConverterError(
                f"Не удалось загрузить so-vits-svc модель '{model_pth_path}': {error}"
            ) from error
        self._rvc_pipeline = pipeline
        self._pipeline_sample_rate = pipeline.target_sample
        self._loaded_model_path = model_pth_path

    def unload_model(self):
        self._rvc_pipeline = None
        self._pipeline_sample_rate = None
        self._loaded_model_path = None

    def convert(self, audio_chunk: np.ndarray, sample_rate: int,
                pitch_shift_semitones: float = 0.0,
                conversion_strength: float = 0.75) -> np.ndarray:
        """Преобразует один аудиоблок через выбранный voice-conversion backend."""
        if self._rvc_pipeline is None:
            raise VoiceConverterError(
                "Для этого профиля не загружена voice-conversion модель. "
                "Импортируйте/прикрепите обученную модель к профилю."
            )

        try:
            if self._backend_name == "so-vits-svc":
                result, result_sample_rate = self._rvc_pipeline.infer(
                    0,
                    int(round(pitch_shift_semitones)),
                    audio_chunk,
                )
                result = result.detach().cpu().numpy()
            else:
                try:
                    result = self._rvc_pipeline.infer(
                        audio_chunk,
                        sample_rate,
                        f0_up_key=pitch_shift_semitones,
                        index_rate=conversion_strength,
                    )
                except TypeError:
                    result = self._rvc_pipeline.infer(audio_chunk)
                result_sample_rate = self._pipeline_sample_rate or sample_rate
        except Exception as error:
            raise VoiceConverterError(f"Ошибка во время voice conversion: {error}") from error

        result = np.asarray(result, dtype=np.float32).flatten()

        # RVC внутренне может работать на другом sample rate (обычно 40-48кГц) -
        # приводим обратно к sample rate real-time пайплайна, чтобы не рассинхронизировать
        # буферы вывода.
        if len(result) != len(audio_chunk) or result_sample_rate != sample_rate:
            import librosa
            result = librosa.resample(
                result, orig_sr=int(result_sample_rate),
                target_sr=sample_rate,
            )
            if len(result) > len(audio_chunk):
                result = result[: len(audio_chunk)]
            elif len(result) < len(audio_chunk):
                padded = np.zeros(len(audio_chunk), dtype=np.float32)
                padded[: len(result)] = result
                result = padded

        return np.clip(result, -1.0, 1.0)
