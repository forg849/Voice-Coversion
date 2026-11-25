"""
effects.py

Дополнительные аудио-эффекты, применяемые ПОСЛЕ voice conversion
(см. пайплайн в voice_converter.py / audio_engine.py):

    Microphone -> Noise Reduction -> Voice Conversion -> Pitch/Formant
                -> EQ -> Compressor -> Output

Все эффекты реализованы на numpy/scipy/librosa и работают на небольших
чанках в реальном времени. Это осознанный компромисс: библиотеки классов
"studio grade" (rubberband, iZotope и т.п.) либо не являются open-source,
либо не имеют официальных Python-биндингов под Windows — поэтому здесь
используются реализации на основе librosa (phase vocoder для pitch/formant)
и ручных DSP-функций (EQ/compressor/reverb/echo/distortion), что даёт
предсказуемое качество без внешних бинарных зависимостей.

Каждый эффект — простая функция (chunk, sr, **params) -> chunk,
и отдельно есть EffectsChain, которая хранит включенность и параметры
каждого эффекта и применяет их по порядку.
"""

import numpy as np
from scipy import signal


# ------------------------------------------------------------------ #
# Отдельные эффекты
# ------------------------------------------------------------------ #

def pitch_shift(chunk: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    if abs(semitones) < 1e-3:
        return chunk
    import librosa
    return librosa.effects.pitch_shift(chunk, sr=sr, n_steps=semitones).astype(np.float32)


def formant_shift(chunk: np.ndarray, sr: int, shift_ratio: float) -> np.ndarray:
    """
    Простой формантный сдвиг через ресемплинг спектральной огибающей
    (приближённая реализация: меняем длину сигнала, затем возвращаем
    исходную длительность через ресемплинг обратно — это сдвигает форманты
    сильнее, чем основной тон, но не идеально их разделяет).
    shift_ratio: 1.0 = без изменений, >1.0 = "выше"/тоньше голос, <1.0 = "ниже"/грубее.
    """
    if abs(shift_ratio - 1.0) < 1e-3:
        return chunk
    import librosa
    n = len(chunk)
    stretched = librosa.effects.time_stretch(chunk, rate=shift_ratio)
    resampled = librosa.resample(stretched, orig_sr=sr, target_sr=int(sr * shift_ratio))
    if len(resampled) >= n:
        return resampled[:n].astype(np.float32)
    out = np.zeros(n, dtype=np.float32)
    out[: len(resampled)] = resampled
    return out


def noise_reduction(chunk: np.ndarray, sr: int, strength: float = 0.5) -> np.ndarray:
    """
    Спектральное вычитание шума на лету: оценивает "шумовой пол" как
    нижний перцентиль спектральной энергии текущего чанка и ослабляет
    частоты около него. Это упрощённая (но реальная) реализация в духе
    классического spectral gating, без необходимости отдельного calibration-шага.
    """
    if strength <= 0:
        return chunk
    import librosa
    n_fft = 512
    hop = 128
    S = librosa.stft(chunk, n_fft=n_fft, hop_length=hop)
    mag, phase = np.abs(S), np.angle(S)
    noise_floor = np.percentile(mag, 20, axis=1, keepdims=True)
    mag_denoised = np.maximum(mag - noise_floor * strength, 0.0)
    S_denoised = mag_denoised * np.exp(1j * phase)
    out = librosa.istft(S_denoised, hop_length=hop, length=len(chunk))
    return out.astype(np.float32)


def reverb(chunk: np.ndarray, sr: int, mix: float = 0.25, decay: float = 0.35) -> np.ndarray:
    """Простой алгоритмический реверб на основе гребёнки задержек (Schroeder-подобный, упрощённый)."""
    if mix <= 0:
        return chunk
    delays_ms = [29.7, 37.1, 41.3, 43.7]
    out = chunk.copy()
    for d_ms in delays_ms:
        d_samples = max(1, int(sr * d_ms / 1000.0))
        delayed = np.zeros_like(chunk)
        if d_samples < len(chunk):
            delayed[d_samples:] = chunk[:-d_samples] * decay
        out = out + delayed / len(delays_ms)
    return ((1 - mix) * chunk + mix * out).astype(np.float32)


def echo(chunk: np.ndarray, sr: int, delay_ms: float = 180.0, feedback: float = 0.3) -> np.ndarray:
    if feedback <= 0 or delay_ms <= 0:
        return chunk
    # Realtime chunks are shorter than a typical musical delay; keep the
    # effect audible within the current chunk instead of silently skipping it.
    d_samples = min(
        max(1, int(sr * delay_ms / 1000.0)),
        max(1, len(chunk) // 4),
    )
    out = chunk.copy()
    if d_samples < len(chunk):
        out[d_samples:] += chunk[:-d_samples] * feedback
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def distortion(chunk: np.ndarray, sr: int, drive: float = 0.0) -> np.ndarray:
    if drive <= 0:
        return chunk
    gain = 1.0 + drive * 9.0
    driven = np.tanh(chunk * gain)
    return (driven / np.tanh(gain)).astype(np.float32)


def equalizer(chunk: np.ndarray, sr: int, low_gain_db: float = 0.0,
              mid_gain_db: float = 0.0, high_gain_db: float = 0.0) -> np.ndarray:
    """3-полосный EQ на биквад-фильтрах (low shelf / mid peak / high shelf)."""
    if abs(low_gain_db) < 1e-3 and abs(mid_gain_db) < 1e-3 and abs(high_gain_db) < 1e-3:
        return chunk
    out = chunk.astype(np.float64)

    if abs(low_gain_db) > 1e-3:
        sos = signal.butter(2, 250, btype="lowpass", fs=sr, output="sos")
        low_band = signal.sosfilt(sos, out)
        out = out + low_band * (10 ** (low_gain_db / 20.0) - 1.0)

    if abs(mid_gain_db) > 1e-3:
        sos = signal.butter(2, [500, 3000], btype="bandpass", fs=sr, output="sos")
        mid_band = signal.sosfilt(sos, out)
        out = out + mid_band * (10 ** (mid_gain_db / 20.0) - 1.0)

    if abs(high_gain_db) > 1e-3:
        sos = signal.butter(2, 4000, btype="highpass", fs=sr, output="sos")
        high_band = signal.sosfilt(sos, out)
        out = out + high_band * (10 ** (high_gain_db / 20.0) - 1.0)

    return np.clip(out, -1.0, 1.0).astype(np.float32)


def compressor(chunk: np.ndarray, sr: int, threshold_db: float = -18.0,
               ratio: float = 3.0, makeup_db: float = 4.0) -> np.ndarray:
    """Простой feed-forward компрессор с усреднением огибающей по RMS в окне ~5мс."""
    if ratio <= 1.0:
        return chunk
    eps = 1e-9
    window = max(1, int(sr * 0.005))
    abs_chunk = np.abs(chunk) + eps
    kernel = np.ones(window) / window
    envelope = np.sqrt(np.convolve(chunk ** 2, kernel, mode="same") + eps)

    envelope_db = 20 * np.log10(envelope + eps)
    over_db = np.maximum(envelope_db - threshold_db, 0.0)
    gain_reduction_db = over_db * (1.0 - 1.0 / ratio)
    gain = 10 ** (-gain_reduction_db / 20.0)
    makeup = 10 ** (makeup_db / 20.0)

    out = chunk * gain * makeup
    return np.clip(out, -1.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------ #
# Цепочка эффектов
# ------------------------------------------------------------------ #

class EffectsChain:
    """
    Хранит включенность/выключенность и параметры каждого эффекта и
    применяет их в фиксированном порядке, соответствующем разделу 9
    технического задания:

        Noise Reduction -> Pitch/Formant -> EQ -> Compressor
        (Reverb/Echo/Distortion применяются между Formant и EQ,
         как дополнительная "окраска" звука, если включены).

    Voice Conversion в эту цепочку НЕ входит - он выполняется отдельно,
    в voice_converter.py, до вызова EffectsChain.process().
    """

    def __init__(self):
        self.enabled = {
            "noise_reduction": False,
            "pitch": False,
            "formant": False,
            "reverb": False,
            "echo": False,
            "distortion": False,
            "equalizer": False,
            "compressor": False,
        }
        self.params = {
            "noise_reduction": {"strength": 0.5},
            "pitch": {"semitones": 0.0},
            "formant": {"shift_ratio": 1.0},
            "reverb": {"mix": 0.25, "decay": 0.35},
            "echo": {"delay_ms": 180.0, "feedback": 0.3},
            "distortion": {"drive": 0.35},
            "equalizer": {"low_gain_db": 2.0, "mid_gain_db": -1.0, "high_gain_db": 2.0},
            "compressor": {"threshold_db": -18.0, "ratio": 3.0, "makeup_db": 4.0},
        }

    def set_enabled(self, effect_name: str, value: bool):
        if effect_name not in self.enabled:
            raise KeyError(f"Неизвестный эффект: {effect_name}")
        self.enabled[effect_name] = value

    def set_params(self, effect_name: str, **kwargs):
        if effect_name not in self.params:
            raise KeyError(f"Неизвестный эффект: {effect_name}")
        self.params[effect_name].update(kwargs)

    def process(self, chunk: np.ndarray, sr: int) -> np.ndarray:
        out = chunk
        if self.enabled["noise_reduction"]:
            out = noise_reduction(out, sr, **self.params["noise_reduction"])
        if self.enabled["pitch"]:
            out = pitch_shift(out, sr, **self.params["pitch"])
        if self.enabled["formant"]:
            out = formant_shift(out, sr, **self.params["formant"])
        if self.enabled["reverb"]:
            out = reverb(out, sr, **self.params["reverb"])
        if self.enabled["echo"]:
            out = echo(out, sr, **self.params["echo"])
        if self.enabled["distortion"]:
            out = distortion(out, sr, **self.params["distortion"])
        if self.enabled["equalizer"]:
            out = equalizer(out, sr, **self.params["equalizer"])
        if self.enabled["compressor"]:
            out = compressor(out, sr, **self.params["compressor"])
        return out
