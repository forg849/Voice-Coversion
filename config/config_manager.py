"""
config_manager.py

Отвечает за загрузку, сохранение и валидацию настроек приложения.
Настройки хранятся в JSON-файле в папке data/ рядом с программой
(а не в system32/Program Files, чтобы не требовать прав администратора).

Хранит:
- выбранные аудио-устройства (вход/выход)
- sample rate, buffer size
- режим CPU/GPU, quality, low latency
- настройки интерфейса (тема, трей, старт свернутым)
"""

import json
import os
import copy
from pathlib import Path

DEFAULT_CONFIG = {
    "audio": {
        "input_device": None,      # None = системное устройство по умолчанию
        "output_device": None,
        "sample_rate": 44100,
        "buffer_size": 1024,       # frames per callback
        "chunk_size": 2048,        # frames, отправляемых на конвертацию за раз
    },
    "processing": {
        "device_mode": "auto",     # "auto" | "cpu" | "gpu"
        "quality": "balanced",     # "fast" | "balanced" | "high"
        "low_latency_mode": False,
    },
    "interface": {
        "dark_theme": True,
        "start_minimized": False,
        "minimize_to_tray": True,
    },
    "last_profile": None,
    "last_preset": None,
}


class ConfigManager:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.data_dir / "config.json"
        self._config = self._load()

    def _load(self) -> dict:
        if not self.config_path.exists():
            self._save_dict(DEFAULT_CONFIG)
            return copy.deepcopy(DEFAULT_CONFIG)
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
        except (json.JSONDecodeError, OSError):
            backup_path = self.config_path.with_suffix(".json.bak")
            try:
                os.replace(self.config_path, backup_path)
            except OSError:
                pass
            self._save_dict(DEFAULT_CONFIG)
            return copy.deepcopy(DEFAULT_CONFIG)

        merged = copy.deepcopy(DEFAULT_CONFIG)
        self._deep_update(merged, loaded)
        return merged

    @staticmethod
    def _deep_update(base: dict, updates: dict) -> None:
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                ConfigManager._deep_update(base[key], value)
            else:
                base[key] = value

    def _save_dict(self, data: dict) -> None:
        tmp_path = self.config_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self.config_path)

    def save(self) -> None:
        self._save_dict(self._config)

    def get(self, section: str, key: str, default=None):
        return self._config.get(section, {}).get(key, default)

    def set(self, section: str, key: str, value) -> None:
        self._config.setdefault(section, {})[key] = value

    def get_section(self, section: str) -> dict:
        return copy.deepcopy(self._config.get(section, {}))

    def set_section(self, section: str, values: dict) -> None:
        self._config.setdefault(section, {}).update(values)

    @property
    def raw(self) -> dict:
        return self._config
