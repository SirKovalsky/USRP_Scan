"""Параметры приёма и отображения."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


# Диапазон B210 (общий): 70 МГц .. 6 ГГц, но "по частотам" чаще всего
# интересует FM/авиация/ISM. Значения по умолчанию - FM-диапазон.
MIN_B210_FREQ = 70e6
MAX_B210_FREQ = 6e9


@dataclass
class AcquisitionConfig:
    """Полный набор параметров одного сеанса измерения."""

    # --- Устройство ---
    mode: str = "sweep"                 # "sweep" | "realtime" | "gnss"
    device_args: str = "type=b200"
    channel: int = 0
    antenna: str = "RX2"
    gain: float = 30.0                  # дБ
    sample_rate: float = 4e6            # Гц

    # --- Свип ---
    start_freq: float = 88e6            # Гц
    stop_freq: float = 108e6            # Гц
    sweep_step: float = 0.0             # Гц; 0 => авто
    settle_time: float = 0.02           # с, пауза после перестройки

    # --- Реальное время ---
    center_freq: float = 100e6          # Гц

    # --- Мониторинг ГНСС (обнаружение помех) ---
    gnss_band: str = "ГНСС: L1/E1/B1 (1559-1610 МГц)"
    gnss_jam_delta_db: float = 6.0      # рост шумовой дорожки = глушение
    gnss_nb_db: float = 10.0            # превышение над шумом для узкополосной помехи
    gnss_saturation_db: float = -6.0    # уровень пика, при котором АЦП перегружен

    # --- Обработка ---
    fft_size: int = 4096
    overlap: float = 0.25               # доля перекрытия сегментов Welch 0..0.9
    averages: int = 4                   # сколько сегментов усреднять
    window: str = "hann"                # hann | hamming | blackman-harris | rect

    # --- Отображение/детекция ---
    threshold_db: float = -70.0
    min_peak_distance: float = 200e3    # Гц
    min_prominence_db: float = 6.0
    max_peaks: int = 40
    waterfall_rows: int = 300
    wf_level_low: float = -110.0
    wf_level_high: float = -20.0

    # ------------------------------------------------------------------
    @property
    def bin_width(self) -> float:
        return self.sample_rate / float(self.fft_size)

    def effective_step(self) -> float:
        """Шаг перестройки центра, обеспечивающий сплошное покрытие."""
        if self.sweep_step and self.sweep_step > 0:
            return float(self.sweep_step)
        overlap = min(max(self.overlap, 0.0), 0.9)
        return max(self.bin_width, self.sample_rate * (1.0 - overlap))

    def sweep_frequencies(self) -> np.ndarray:
        """Список центральных частот для свипа.

        Первый центр = start + fs/2, чтобы нижняя граница попала точно в start.
        """
        span = self.stop_freq - self.start_freq
        if span <= 0:
            return np.array([self.start_freq], dtype=float)
        step = self.effective_step()
        half = self.sample_rate / 2.0
        count = int(np.floor(span / step)) + 1
        centers = self.start_freq + half + np.arange(count, dtype=float) * step
        # если последний захват не дотягивает до stop - добавляем ещё один центр
        if centers[-1] + half < self.stop_freq:
            centers = np.append(centers, centers[-1] + step)
        return centers

    def samples_per_capture(self) -> int:
        """Сколько сэмплов снимать за одну точку свипа."""
        seg = int(self.fft_size)
        avg = max(1, int(self.averages))
        return seg * avg

    def samples_per_block(self) -> int:
        """Сколько сэмплов брать в одном блоке реального времени."""
        return self.samples_per_capture()

    def validate(self) -> None:
        if self.mode == "sweep":
            if self.stop_freq <= self.start_freq:
                raise ValueError("Стоп-частота должна быть больше начальной.")
            if not (MIN_B210_FREQ <= self.start_freq <= MAX_B210_FREQ):
                raise ValueError(
                    f"Начальная частота вне диапазона B210 "
                    f"({MIN_B210_FREQ/1e6:.0f}..{MAX_B210_FREQ/1e9:.0f} МГц/ГГц)."
                )
        else:
            if not (MIN_B210_FREQ <= self.center_freq <= MAX_B210_FREQ):
                raise ValueError(
                    f"Центральная частота вне диапазона B210 "
                    f"({MIN_B210_FREQ/1e6:.0f}..{MAX_B210_FREQ/1e9:.0f} МГц/ГГц)."
                )
        if self.fft_size < 64:
            raise ValueError("fft_size слишком мал.")
        if self.sample_rate <= 0:
            raise ValueError("Некорректная частота дискретизации.")

    def copy(self, **changes) -> "AcquisitionConfig":
        return replace(self, **changes)
