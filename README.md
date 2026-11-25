# Voice Conversion Studio

Desktop-приложение на Python (Windows) для преобразования голоса в реальном
времени: пользователь создаёт **voice profile** из своей аудиозаписи (или
записи, на использование которой у него есть право), а затем говорит в
микрофон и слышит на выходе речь с тембром выбранного профиля.




## 1. Анализ технологий voice conversion и выбор

Перед реализацией были рассмотрены реалистичные для локального запуска на
Windows + Python варианты:

| Технология | Real-time | Готовые предобученные веса | Python API | GPU (CUDA) | Установка |
|---|---|---|---|---|---|
| **RVC** (Retrieval-based Voice Conversion) | Да (community real-time форки существуют) | Да (HuBERT + RMVPE + community-модели) | Да (`rvc-python`) | Да | Средняя |
| so-vits-svc | Частично | Да | Ограниченная | Да | Сложная (много ручных шагов) |
| Coqui TTS / YourTTS (voice cloning) | Нет (это TTS, не STT->voice conversion микрофона в реальном времени) | Да | Да | Да | Средняя |
| OpenVoice | Ограниченно real-time | Да | Да | Да | Сложная, тяжёлые зависимости |
| Простой pitch/formant DSP (без ML) | Да | Не требуется | Тривиально | Не нужен | Простая, но **не меняет тембр/идентичность голоса**, только высоту — это НЕ voice conversion в смысле ТЗ |

**Выбор: RVC**, через пакет `rvc-python`. Причины:
- единственный вариант из списка с одновременно (а) реальными
  Python-биндингами, (б) поддержкой CUDA, (в) сообществом real-time
  реализаций под Windows (в духе w-okada's Real-Time Voice Changer,
  построенного поверх той же технологии);
- поддерживает оба нужных режима: **Fast Mode** (готовая модель) и
  **Custom Voice Mode** (модель, обученная под конкретный голос).

Простые эффекты (Pitch/Formant/Reverb/Echo/...) в этой программе — это
**дополнительный, отдельный этап после voice conversion** (раздел 9 ТЗ),
а не замена ему.

---

## 2. Требования

- **Python:** 3.10 или 3.11, 64-bit (обязательно 64-bit из-за torch/CUDA).
- **ОС:** Windows 10/11 (сборка ориентирована на Windows; Linux/macOS
  технически могут запустить GUI и DSP-часть, но без гарантии по
  virtual-microphone разделу, который специфичен для Windows-инструментов
  типа VB-CABLE).
- **CPU-режим:** от 8 ГБ RAM, latency real-time-конвертации будет заметно
  выше (сотни мс), чем на GPU.
- **GPU-режим (рекомендуется для real-time):**
  - NVIDIA GPU с CUDA compute capability ≥ 6.1;
  - от 4 ГБ VRAM (комфортно — от 6 ГБ);
  - установленные NVIDIA-драйверы, совместимые с CUDA 12.x;
  - `torch` собранный с CUDA (см. установку ниже) — обычная PyPI-версия
    `torch` из requirements.txt по умолчанию CPU-only.
- **Диск:** ~3-5 ГБ на torch/зависимости + ~1 ГБ на базовые веса RVC + место
  под сами голосовые модели (~50-200 МБ на модель).

### Требования отдельно для real-time режима
Real-time означает, что за то время, пока накапливается один chunk аудио
(`chunk_size / sample_rate` секунд), должна успевать выполниться voice
conversion + эффекты. На CPU это часто НЕ укладывается в комфортные <100 мс
для чанков разумного размера — интерфейс показывает фактическую latency
честно (раздел 7 ТЗ), включая случаи, когда она высокая. На GPU среднего
уровня (RTX 3060+) latency 60-150 мс при `buffer_size`/`chunk_size` из
Settings обычно достижима, но зависит от выбранного Quality.

---

## 4. Установка (Windows PowerShell)

python -m venv .venv

.venv\Scripts\Activate.ps1

pip install torch --index-url 

pip install -r requirements.txt


pip install -r requirements-python314.txt
pip install --no-deps so-vits-svc-fork==4.2.30
pip install click cm-time fastapi lightning matplotlib pebble praat-parselmouth `
   psutil PySimpleGUI-4-foss torch torchaudio torchcrepe transformers `
   tqdm tqdm-joblib sounddevice tensorboard tensorboardx


Если на шаге 3 вы его пропустите, `pip install -r requirements.txt`
поставит CPU-версию `torch` — программа всё равно запустится и будет
работать, но в режиме CPU (см. предупреждение "GPU acceleration
unavailable. Using CPU mode.").

---

## 5. Сборка в EXE (PyInstaller)

```powershell
pip install pyinstaller

pyinstaller --noconfirm --windowed --name VoiceConversionStudio ^
    --icon NONE ^
    main.py
```

`--windowed` гарантирует, что при обычном запуске .exe консольное окно не
открывается (ошибки всё равно попадают в `data/last_error.log`, см.
main.py).

Базовые веса RVC и голосовые модели **не** зашиваются внутрь .exe (слишком
большие, и это ломало бы обновление моделей без пересборки). После сборки
разложите файлы так:

```text
VoiceConversionStudio/
├── VoiceConversionStudio.exe
├── models/                 <- скопируйте сюда содержимое data/models/ (shared/, pretrained/)
├── profiles/               <- при желании перенесите заранее подготовленные voices/
└── config/
```

Обратите внимание: сама программа при первом запуске .exe создаст рядом с
ним папку `data/` (models/, voices/, config.json) автоматически — папки
выше можно either создать заранее и предзаполнить моделями, либо просто
положить нужные файлы в `data/models/shared/` и `data/models/pretrained/`
после первого запуска.


## 6. Как проверить программу

1. Запустить программу (`python main.py`, либо собранный .exe).
2. Нажать **"+ Import"**, выбрать разрешённую к использованию аудиозапись
   голоса (WAV/MP3/FLAC/OGG) и подтвердить создание профиля — убедиться, что
   в отчёте отображаются Duration/Sample Rate/Channels/Quality/Voice
   detected.
3. Убедиться, что для профиля создана папка в `data/voices/<Имя>/`.
4. Через **"Attach RVC Model..."** прикрепить готовую или заранее
   обученную `.pth` (+опционально `.index`) модель — либо использовать
   Fast Mode модель из `data/models/pretrained/`.
5. Нажать **"● RECORD"** в блоке **Test Voice**, произнести фразу, дождаться
   Original/Converted и прослушать оба варианта через **▶ Play**.
6. В нижней панели выбрать реальный микрофон в **Input** и обычные
   динамики/наушники в **Output** (для первой проверки — НЕ виртуальный
   кабель, чтобы сразу услышать результат).
7. Нажать **START**, говорить в микрофон, проверить индикатор **Input
   Level** и значение **Latency** — оно должно быть честным (не
   фиксированной заглушкой).
8. Нажать **STOP**.
9. Установить VB-CABLE, в **Settings → Audio** выбрать "CABLE Input" как
   Output программы.
10. В Discord (Settings → Voice & Video → Input Device) выбрать
    "CABLE Output" и проверить, что собеседники слышат преобразованный
    голос.



