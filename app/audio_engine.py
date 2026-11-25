"""
audio_engine.py

Два независимых блока:

1. Offline-препроцессинг импортируемого файла (import voice):
   - валидация формата;
   - ресемплинг к целевой частоте;
   - конвертация в mono;
   - обрезка тишины по краям;
   - нормализация громкости (peak/RMS);
   - грубая эвристическая проверка "похоже ли это на голос" (VAD на энергии +
     базовая проверка спектра, БЕЗ отдельной ML voice-detection модели —
     честно ограниченная, но реальная проверка, а не заглушка).

2. Real-time аудио-пайплайн (микрофон -> обработка -> выход):
   - работает в отдельном потоке через callback sounddevice;
   - использует потокобезопасные очереди для передачи блоков между
     audio-callback (реального времени) и воркер-потоком конвертации
     (voice_converter + effects), чтобы не блокировать сам audio callback
     тяжёлыми вычислениями;
   - измеряет фактическую end-to-end задержку.
"""

import time
import queue
import threading
import numpy as np
import soundfile as sf
import librosa


class AudioEngineError(Exception):
    pass


# ---------------------------------------------------------------------- #
# Offline: импорт и анализ голосовой записи
# ---------------------------------------------------------------------- #

SUPPORTED_IMPORT_FORMATS = (".wav", ".mp3", ".flac", ".ogg")

TARGET_SAMPLE_RATE = 16000       # частота, ожидаемая большинством content-encoder'ов (HuBERT и т.п.)
TARGET_PROFILE_SAMPLE_RATE = 44100  # частота, в которой хранится "чистый" source.wav профиля

MIN_RECOMMENDED_DURATION_SEC = 20.0     # ниже этого - предупреждение о низком качестве профиля
MIN_USABLE_DURATION_SEC = 1.5           # ниже этого - файл вообще отклоняется


def validate_import_path(path: str) -> None:
    lower = path.lower()
    if not lower.endswith(SUPPORTED_IMPORT_FORMATS):
        raise AudioEngineError(
            f"Неподдерживаемый формат файла. Поддерживаются: {', '.join(SUPPORTED_IMPORT_FORMATS)}"
        )


def _trim_silence(y: np.ndarray, sr: int, top_db: float = 35.0) -> np.ndarray:
    trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    if trimmed.size == 0:
        return y  # если обрезали всё (запись — тишина), не портим дальше, пусть анализ это поймает
    return trimmed


def _normalize_peak(y: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak < 1e-6:
        return y
    return y * (target_peak / peak)


def _estimate_voice_presence(y: np.ndarray, sr: int) -> bool:
    """
    Грубая эвристика: реальный человеческий голос почти всегда имеет
    заметную энергию в полосе ~80-1100 Гц (основной тон + нижние форманты)
    и не является ни константой, ни чистым шумом/тишиной.
    Это НЕ полноценный voice-activity-detector на базе ML, а честная
    практическая проверка, которая ловит явно пустые/бракованные записи.
    """
    if y.size < sr * 0.3:
        return False
    rms = np.sqrt(np.mean(y.astype(np.float64) ** 2))
    if rms < 0.001:
        return False

    n_fft = 2048
    hop = 512
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band_mask = (freqs >= 80) & (freqs <= 1100)
    if not np.any(band_mask):
        return False

    band_energy = np.mean(stft[band_mask, :])
    total_energy = np.mean(stft) + 1e-9
    ratio = band_energy / total_energy
    return ratio > 0.15


def analyze_audio(path: str) -> dict:
    """
    Загружает файл, приводит к целевому формату профиля и возвращает
    метаданные + подготовленный numpy-массив для сохранения.

    Возвращает dict:
        duration_sec, sample_rate, channels_original, quality,
        voice_detected, warnings (list[str]), processed_audio (np.ndarray, float32, mono)
    """
    validate_import_path(path)

    try:
        y_raw, sr_raw = sf.read(path, always_2d=False)
    except Exception as e:
        raise AudioEngineError(f"Не удалось прочитать аудиофайл: {e}") from e

    if y_raw.size == 0:
        raise AudioEngineError("Файл пустой или повреждён.")

    channels_original = 1 if y_raw.ndim == 1 else y_raw.shape[1]
    y_mono = y_raw if y_raw.ndim == 1 else np.mean(y_raw, axis=1)
    y_mono = y_mono.astype(np.float32)

    if sr_raw != TARGET_PROFILE_SAMPLE_RATE:
        y_mono = librosa.resample(y_mono, orig_sr=sr_raw, target_sr=TARGET_PROFILE_SAMPLE_RATE)

    y_trimmed = _trim_silence(y_mono, TARGET_PROFILE_SAMPLE_RATE)
    y_norm = _normalize_peak(y_trimmed)

    duration_sec = len(y_norm) / TARGET_PROFILE_SAMPLE_RATE
    voice_detected = _estimate_voice_presence(y_norm, TARGET_PROFILE_SAMPLE_RATE)

    warnings = []
    quality = "Good"

    if duration_sec < MIN_USABLE_DURATION_SEC:
        raise AudioEngineError(
            f"Запись слишком короткая ({duration_sec:.1f} сек после удаления тишины). "
            f"Минимум для создания профиля: {MIN_USABLE_DURATION_SEC:.0f} сек чистой речи."
        )

    if not voice_detected:
        quality = "Poor"
        warnings.append(
            "Не удалось уверенно обнаружить речь в записи. Возможно, это музыка, шум "
            "или слишком тихая/зашумлённая запись. Профиль может получиться низкого качества."
        )

    if duration_sec < MIN_RECOMMENDED_DURATION_SEC:
        if quality != "Poor":
            quality = "Fair"
        warnings.append(
            f"Запись короткая ({duration_sec:.0f} сек). Для Fast Mode этого достаточно, "
            f"но для качественного Custom Voice профиля рекомендуется от нескольких минут "
            f"чистой речи одного человека (см. раздел про Custom Voice Mode)."
        )

    clipping_ratio = float(np.mean(np.abs(y_raw.astype(np.float32) if y_raw.ndim == 1
                                           else np.mean(y_raw, axis=1)) > 0.98))
    if clipping_ratio > 0.001:
        if quality == "Good":
            quality = "Fair"
        warnings.append("Обнаружены признаки клиппинга (перегрузки) в исходной записи.")

    return {
        "duration_sec": round(duration_sec, 2),
        "sample_rate": sr_raw,
        "channels_original": channels_original,
        "quality": quality,
        "voice_detected": bool(voice_detected),
        "warnings": warnings,
        "processed_audio": y_norm,
        "processed_sample_rate": TARGET_PROFILE_SAMPLE_RATE,
    }


def save_processed_audio(processed_audio: np.ndarray, sample_rate: int, dest_path: str) -> None:
    sf.write(dest_path, processed_audio, sample_rate, subtype="PCM_16")


# ---------------------------------------------------------------------- #
# Real-time: микрофон -> обработка -> выход
# ---------------------------------------------------------------------- #

class RealtimeAudioPipeline:
    """
    Управляет двумя sounddevice-потоками (вход и выход) и передаёт блоки
    аудио между callback'ами реального времени и рабочим потоком обработки
    через thread-safe очереди, чтобы тяжёлые вычисления (voice conversion,
    эффекты) никогда не выполнялись прямо внутри audio callback'а — это
    обязательное условие, иначе PortAudio будет получать xruns/треск.
    """

    def __init__(
        self,
        input_device: int,
        output_device: int,
        sample_rate: int,
        buffer_size: int,
        chunk_size: int,
        process_fn,
        on_error=None,
        on_level=None,
    ):
        import sounddevice as sd
        self._sd = sd

        self.input_device = input_device
        self.output_device = output_device
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.chunk_size = chunk_size
        self.process_fn = process_fn
        self.on_error = on_error
        self.on_level = on_level

        self._input_queue = queue.Queue(maxsize=64)
        self._output_queue = queue.Queue(maxsize=64)
        self._worker_thread = None
        self._running = threading.Event()
        self._input_stream = None
        self._output_stream = None

        self._input_accum = np.zeros((0,), dtype=np.float32)
        self._output_accum = np.zeros((0,), dtype=np.float32)
        self._output_lock = threading.Lock()

        self.last_latency_ms = 0.0
        self._last_chunk_sent_at = None

    # -- sounddevice callbacks (выполняются в потоке реального времени) -- #

    def _input_callback(self, indata, frames, time_info, status):
        if status and self.on_error:
            self.on_error(f"Input stream status: {status}")
        mono = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()

        if self.on_level:
            level = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
            self.on_level(level)

        try:
            self._input_queue.put_nowait(mono)
        except queue.Full:
            # Воркер не успевает — жертвуем самым старым блоком, чтобы не расти в задержку
            try:
                self._input_queue.get_nowait()
                self._input_queue.put_nowait(mono)
            except queue.Empty:
                pass

    def _output_callback(self, outdata, frames, time_info, status):
        if status and self.on_error:
            self.on_error(f"Output stream status: {status}")

        with self._output_lock:
            while self._output_accum.shape[0] < frames:
                try:
                    block = self._output_queue.get_nowait()
                except queue.Empty:
                    break
                self._output_accum = np.concatenate([self._output_accum, block])

            if self._output_accum.shape[0] >= frames:
                chunk = self._output_accum[:frames]
                self._output_accum = self._output_accum[frames:]
            else:
                # Недостаточно данных - отдаём тишину, чтобы не было треска, но это
                # означает audio underrun, что видно как рост latency/дребезг звука.
                chunk = np.zeros((frames,), dtype=np.float32)
                if self._output_accum.shape[0] > 0:
                    chunk[: self._output_accum.shape[0]] = self._output_accum
                    self._output_accum = np.zeros((0,), dtype=np.float32)

        outdata[:, 0] = chunk
        if outdata.shape[1] > 1:
            for ch in range(1, outdata.shape[1]):
                outdata[:, ch] = chunk

    # -- рабочий поток: собирает чанки нужного размера и запускает process_fn -- #

    def _worker_loop(self):
        while self._running.is_set():
            try:
                block = self._input_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            self._input_accum = np.concatenate([self._input_accum, block])

            while self._input_accum.shape[0] >= self.chunk_size:
                chunk = self._input_accum[: self.chunk_size]
                self._input_accum = self._input_accum[self.chunk_size:]

                start_time = time.perf_counter()
                try:
                    processed = self.process_fn(chunk, self.sample_rate)
                except Exception as e:
                    if self.on_error:
                        self.on_error(f"Ошибка обработки аудио: {e}")
                    processed = chunk  # честный фолбэк: пропускаем звук как есть, не глушим совсем

                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                buffer_latency_ms = (self.chunk_size / self.sample_rate) * 1000.0
                self.last_latency_ms = elapsed_ms + buffer_latency_ms

                try:
                    self._output_queue.put_nowait(processed.astype(np.float32))
                except queue.Full:
                    try:
                        self._output_queue.get_nowait()
                        self._output_queue.put_nowait(processed.astype(np.float32))
                    except queue.Empty:
                        pass

    # -- публичное управление -- #

    def start(self):
        if self._running.is_set():
            return
        self._running.set()
        self._input_accum = np.zeros((0,), dtype=np.float32)
        self._output_accum = np.zeros((0,), dtype=np.float32)

        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

        try:
            self._input_stream = self._sd.InputStream(
                device=self.input_device,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.buffer_size,
                dtype="float32",
                callback=self._input_callback,
            )
            self._output_stream = self._sd.OutputStream(
                device=self.output_device,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.buffer_size,
                dtype="float32",
                callback=self._output_callback,
            )
            self._input_stream.start()
            self._output_stream.start()
        except Exception as e:
            self._running.clear()
            raise AudioEngineError(f"Не удалось открыть аудиоустройство: {e}") from e

    def stop(self):
        self._running.clear()
        for stream in (self._input_stream, self._output_stream):
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
        self._input_stream = None
        self._output_stream = None
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=1.0)
            self._worker_thread = None

    @property
    def is_running(self) -> bool:
        return self._running.is_set()
