"""Мониторинг полосы ГНСС и обнаружение помех.

Идея: полезные сигналы ГНСС - широкополосные (1-10 МГц) и лежат НИЖЕ
шумовой дорожки приёмника, поэтому в спектре они не видны как пики.
Любой заметный узкополосный пик над шумом, рост шумовой дорожки,
перегрузка АЦП или резкий рост мощности в полосе - признаки помехи.

Алгоритм не декодирует навигационное сообщение: он оценивает только
обстановку по спектру (обнаружение глушения и постановки помех).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import dsp, presets

NORMAL = "normal"
WARNING = "warning"
JAMMING = "jamming"

STATUS_TEXT = {
    NORMAL: "НОРМА",
    WARNING: "ВНИМАНИЕ",
    JAMMING: "ПОМЕХА / ГЛУШЕНИЕ",
}
STATUS_COLOR = {
    NORMAL: "#4ade80",
    WARNING: "#fbbf24",
    JAMMING: "#f87171",
}


def band_options():
    """Список предустановленных диапазонов ГНСС (см. presets)."""
    return presets.presets_for("ГНСС (GNSS)")


@dataclass
class GnssAnalysis:
    """Результат одного анализа спектра."""

    noise_floor_db: float = -180.0
    band_power_db: float = -180.0
    peak_db: float = -180.0
    peak_freq: float = 0.0
    occupied_bw: float = 0.0
    n_narrowband: int = 0
    narrowband_freq: float = 0.0
    narrowband_db: float = -180.0
    flatness: float = 1.0
    delta_noise_db: float = 0.0
    delta_power_db: float = 0.0
    saturation: bool = False
    status: str = NORMAL
    reasons: list[str] = field(default_factory=list)


class GnssMonitor:
    """Следит за полосой ГНСС и держит «обученный» уровень фона.

    Пороговые параметры:
      * ``jam_delta_db`` - рост шумовой дорожки, считающийся глушением;
      * ``nb_prominence_db`` - превышение над шумом для узкополосной помехи;
      * ``saturation_db`` - уровень пика, при котором АЦП перегружен.
    """

    def __init__(
        self,
        jam_delta_db: float = 6.0,
        nb_prominence_db: float = 10.0,
        saturation_db: float = -6.0,
        baseline_tau: float = 25.0,
    ) -> None:
        self.jam_delta_db = float(jam_delta_db)
        self.nb_prominence_db = float(nb_prominence_db)
        self.saturation_db = float(saturation_db)
        self.baseline_tau = max(2.0, float(baseline_tau))
        self._nf: float | None = None
        self._power: float | None = None
        self.frames = 0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._nf = None
        self._power = None
        self.frames = 0

    @property
    def baseline_noise_db(self) -> float | None:
        return self._nf

    @property
    def baseline_power_db(self) -> float | None:
        return self._power

    # ------------------------------------------------------------------
    def analyze(self, freqs: np.ndarray, psd_db: np.ndarray) -> GnssAnalysis:
        psd = np.asarray(psd_db, dtype=np.float64)
        fre = np.asarray(freqs, dtype=np.float64)
        if psd.size < 8 or fre.size != psd.size:
            return GnssAnalysis(reasons=["Нет данных"])

        nf = dsp.noise_floor_db(psd, 30.0)
        lin = np.power(10.0, psd / 10.0)
        amean = float(np.mean(lin)) + 1e-30
        band_power = 10.0 * np.log10(amean)
        gmean = float(np.exp(np.mean(np.log(lin + 1e-30))))
        flatness = gmean / amean

        i = int(np.argmax(psd))
        peak_db = float(psd[i])
        peak_freq = float(fre[i])
        saturation = peak_db >= self.saturation_db

        span = float(fre[-1] - fre[0]) if fre.size > 1 else 1.0
        nb = dsp.find_peaks(
            fre, psd,
            threshold_db=nf + self.nb_prominence_db,
            min_peak_distance=max(span * 0.01, 1.0),
            min_prominence_db=self.nb_prominence_db,
            max_peaks=16,
            exclude_dc_bins=0,
        )
        nb = [p for p in nb if 0.0 < p.get("bandwidth", 0.0) < 0.25 * span]

        result = GnssAnalysis(
            noise_floor_db=nf,
            band_power_db=band_power,
            peak_db=peak_db,
            peak_freq=peak_freq,
            occupied_bw=float(nb[0]["bandwidth"]) if nb else 0.0,
            n_narrowband=len(nb),
            narrowband_freq=float(nb[0]["freq"]) if nb else 0.0,
            narrowband_db=float(nb[0]["power"]) if nb else -180.0,
            flatness=flatness,
            saturation=saturation,
        )

        if self._nf is None or self._power is None:
            self._nf = nf
            self._power = band_power
            self.frames = 1
            result.status = NORMAL
            result.reasons = ["Фон обучен, помех не обнаружено"]
            return result

        dn = nf - self._nf
        dp = band_power - self._power
        result.delta_noise_db = dn
        result.delta_power_db = dp

        # Пока обстановка спокойна - медленно подстраиваем фон под дрейф.
        if abs(dn) < self.jam_delta_db and not saturation:
            a = 1.0 / self.baseline_tau
            self._nf += a * dn
            self._power += a * dp
        self.frames += 1

        reasons: list[str] = []
        status = NORMAL
        if saturation:
            reasons.append(
                f"Перегрузка АЦП: пик {peak_db:.1f} дБ (порог {self.saturation_db:.0f} дБ)"
            )
            status = JAMMING
        if dn >= self.jam_delta_db:
            reasons.append(f"Шумовая дорожка выросла на {dn:.1f} дБ")
            status = JAMMING
        if len(nb) >= 3:
            reasons.append(f"Много узкополосных помех: {len(nb)}")
            status = JAMMING
        elif len(nb) >= 1:
            reasons.append(
                f"Узкополосная помеха на {dsp.format_freq(result.narrowband_freq)} "
                f"({result.narrowband_db:.1f} дБ)"
            )
            if status != JAMMING:
                status = WARNING
        if status == NORMAL and dp >= self.jam_delta_db:
            reasons.append(f"Мощность в полосе выросла на {dp:.1f} дБ")
            status = WARNING
        if status == NORMAL:
            reasons.append("Помех не обнаружено")

        result.status = status
        result.reasons = reasons
        return result
