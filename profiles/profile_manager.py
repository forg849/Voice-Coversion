"""
profile_manager.py

Управление голосовыми профилями (Voice Profiles).

Каждый профиль — это папка внутри data/voices/<ProfileName>/ следующего вида:

    voices/
        Female_Voice_01/
            profile.json        <- метаданные (имя, дата создания, параметры источника)
            source.wav          <- предобработанный исходный файл (16-бит PCM, mono)
            model.pth           <- RVC-модель (.pth), ЕСЛИ она была натренирована/импортирована
            model.index         <- faiss index для RVC retrieval (опционально)

Важно: сам по себе ProfileManager НЕ занимается обучением модели и НЕ занимается
голосовым преобразованием — это делает app/voice_converter.py и app/model_manager.py.
ProfileManager отвечает только за файловую систему и метаданные.
"""

import json
import shutil
import uuid
from pathlib import Path
from datetime import datetime, timezone


class ProfileError(Exception):
    pass


class ProfileManager:
    def __init__(self, voices_dir: str):
        self.voices_dir = Path(voices_dir)
        self.voices_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Служебные методы
    # ------------------------------------------------------------------ #

    def _safe_name(self, name: str) -> str:
        cleaned = "".join(c if (c.isalnum() or c in " _-") else "_" for c in name).strip()
        cleaned = cleaned.replace(" ", "_")
        if not cleaned:
            cleaned = f"Voice_{uuid.uuid4().hex[:8]}"
        return cleaned

    def _profile_dir(self, folder_name: str) -> Path:
        return self.voices_dir / folder_name

    def _unique_folder_name(self, base_name: str) -> str:
        candidate = base_name
        counter = 2
        while self._profile_dir(candidate).exists():
            candidate = f"{base_name}_{counter}"
            counter += 1
        return candidate

    # ------------------------------------------------------------------ #
    # Публичное API
    # ------------------------------------------------------------------ #

    def list_profiles(self) -> list:
        """Возвращает список метаданных всех профилей, отсортированных по имени."""
        profiles = []
        if not self.voices_dir.exists():
            return profiles
        for entry in sorted(self.voices_dir.iterdir()):
            meta_path = entry / "profile.json"
            if entry.is_dir() and meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    meta["folder"] = entry.name
                    profiles.append(meta)
                except (json.JSONDecodeError, OSError):
                    continue
        return profiles

    def get_profile(self, folder_name: str) -> dict:
        meta_path = self._profile_dir(folder_name) / "profile.json"
        if not meta_path.exists():
            raise ProfileError(f"Профиль '{folder_name}' не найден.")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["folder"] = folder_name
        return meta

    def create_profile(
        self,
        display_name: str,
        source_audio_path: str,
        analysis_info: dict,
        model_type: str = "none",
    ) -> dict:
        """
        Создаёт новую папку профиля и копирует в неё предобработанный source-файл.

        analysis_info — результат audio_engine.analyze_audio(), содержащий
        duration, sample_rate, channels, quality, voice_detected и т.д.

        model_type:
            "none"        - модели ещё нет, только исходник (профиль ещё не готов к конвертации)
            "rvc"         - привязана RVC модель (.pth/.index) через attach_model()
            "builtin"     - используется встроенный (Fast Mode) пресет, файл не нужен
        """
        folder_name = self._unique_folder_name(self._safe_name(display_name))
        profile_dir = self._profile_dir(folder_name)
        profile_dir.mkdir(parents=True, exist_ok=False)

        dest_audio = profile_dir / "source.wav"
        shutil.copy2(source_audio_path, dest_audio)

        meta = {
            "display_name": display_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_filename": Path(source_audio_path).name,
            "analysis": analysis_info,
            "model_type": model_type,
            "model_files": {},
            "settings": {
                "pitch": 0,
                "energy": 0,
                "conversion_strength": 0.75,
            },
        }
        self._write_meta(profile_dir, meta)
        meta["folder"] = folder_name
        return meta

    def attach_model(self, folder_name: str, model_pth_path: str, index_path: str = None) -> dict:
        """
        Копирует обученную/импортированную RVC-модель в папку профиля
        и помечает профиль как готовый к реальной конвертации.
        """
        profile_dir = self._profile_dir(folder_name)
        if not profile_dir.exists():
            raise ProfileError(f"Профиль '{folder_name}' не найден.")

        dest_model = profile_dir / "model.pth"
        shutil.copy2(model_pth_path, dest_model)

        meta = self.get_profile(folder_name)
        meta["model_type"] = "rvc"
        meta["model_files"]["pth"] = "model.pth"

        if index_path:
            dest_index = profile_dir / "model.index"
            shutil.copy2(index_path, dest_index)
            meta["model_files"]["index"] = "model.index"

        self._write_meta(profile_dir, meta)
        meta["folder"] = folder_name
        return meta

    def update_settings(self, folder_name: str, settings: dict) -> dict:
        meta = self.get_profile(folder_name)
        meta.setdefault("settings", {}).update(settings)
        profile_dir = self._profile_dir(folder_name)
        self._write_meta(profile_dir, meta)
        meta["folder"] = folder_name
        return meta

    def rename_profile(self, folder_name: str, new_display_name: str) -> dict:
        meta = self.get_profile(folder_name)
        meta["display_name"] = new_display_name
        profile_dir = self._profile_dir(folder_name)
        self._write_meta(profile_dir, meta)
        meta["folder"] = folder_name
        return meta

    def delete_profile(self, folder_name: str) -> None:
        profile_dir = self._profile_dir(folder_name)
        if not profile_dir.exists():
            raise ProfileError(f"Профиль '{folder_name}' не найден.")
        shutil.rmtree(profile_dir)

    def duplicate_profile(self, folder_name: str, new_display_name: str) -> dict:
        src_dir = self._profile_dir(folder_name)
        if not src_dir.exists():
            raise ProfileError(f"Профиль '{folder_name}' не найден.")
        new_folder = self._unique_folder_name(self._safe_name(new_display_name))
        dst_dir = self._profile_dir(new_folder)
        shutil.copytree(src_dir, dst_dir)

        meta = self.get_profile(new_folder)
        meta["display_name"] = new_display_name
        meta["created_at"] = datetime.now(timezone.utc).isoformat()
        self._write_meta(dst_dir, meta)
        meta["folder"] = new_folder
        return meta

    def export_profile(self, folder_name: str, dest_zip_path: str) -> str:
        """Упаковывает профиль (метаданные + модель, БЕЗ трека сторонних лицензий) в .zip."""
        src_dir = self._profile_dir(folder_name)
        if not src_dir.exists():
            raise ProfileError(f"Профиль '{folder_name}' не найден.")
        base = str(Path(dest_zip_path).with_suffix(""))
        archive_path = shutil.make_archive(base, "zip", root_dir=src_dir)
        return archive_path

    def import_profile(self, zip_path: str, display_name: str = None) -> dict:
        """Распаковывает .zip профиля, созданный export_profile(), в новую папку."""
        tmp_extract_name = self._unique_folder_name("_import_tmp")
        tmp_dir = self._profile_dir(tmp_extract_name)
        shutil.unpack_archive(zip_path, tmp_dir)

        meta_path = tmp_dir / "profile.json"
        if not meta_path.exists():
            shutil.rmtree(tmp_dir)
            raise ProfileError("Файл профиля повреждён или имеет неверный формат.")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        final_display_name = display_name or meta.get("display_name", "Imported Voice")
        final_folder = self._unique_folder_name(self._safe_name(final_display_name))
        final_dir = self._profile_dir(final_folder)
        tmp_dir.rename(final_dir)

        meta["display_name"] = final_display_name
        self._write_meta(final_dir, meta)
        meta["folder"] = final_folder
        return meta

    def profile_audio_path(self, folder_name: str) -> Path:
        return self._profile_dir(folder_name) / "source.wav"

    def profile_model_path(self, folder_name: str) -> Path:
        return self._profile_dir(folder_name) / "model.pth"

    def profile_index_path(self, folder_name: str) -> Path:
        return self._profile_dir(folder_name) / "model.index"

    # ------------------------------------------------------------------ #

    def _write_meta(self, profile_dir: Path, meta: dict) -> None:
        meta_to_save = {k: v for k, v in meta.items() if k != "folder"}
        tmp_path = profile_dir / "profile.json.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(meta_to_save, f, indent=2, ensure_ascii=False)
        tmp_path.replace(profile_dir / "profile.json")
