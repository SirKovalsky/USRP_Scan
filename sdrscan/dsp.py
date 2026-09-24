"""Цифровая обработка: PSD (метод Уэлча) и поиск пиков.

Мощность нормируется так, чтобы комплексный тон амплитуды 1.0 на центре
бин давал 0 дБ. Это привычная для анализатора спектра шкала dBFS:
    - максимальный уровень = 0 дБ,
    - шумовая дорожка в зависимости от усиления уходит вниз.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-24


def make_window(kind: str, size: int) -> np.ndarray:
    kind = (kind or "hann").lower()
    if kind in ("rect", "rectangular", "none", "boxcar"):
        return np.ones(size, dtype=np.float64)
    if kind == "hamming":
        return np.hamming(size)
    if kind in ("blackman-harris", "blackmanharris", "bh"):
        return np.blackman(size)
    return np.hanning(size)


def compute_psd(
    samples: np.ndarray,
    sample_rate: float,
    fft_size: int | None = None,
    window: str = "hann",
    overlap: float = 0.25,
    averages: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Оценить спектральную плотность мощности (Welch).

    Возвращает ``(freqs, psd_db)`` где ``freqs`` - смещённые частоты
    (от -fs/2 до +fs/2), ``psd_db`` - уровень в дБ относительно полной шкалы.
    """
    x = np.asarray(samples)
    if x.dtype != np.complex64 and x.dtype != np.complex128:
        x = x.astype(np.complex64)
    if x.size == 0:
        return np.zeros(0), np.zeros(0)

    n = int(fft_size or x.size)
    n = min(n, x.size) if x.size >= n else n

    if x.size < n:
        x = np.pad(x, (0, n - x.size))

    win = make_window(window, n)
    # Нормировка на когерентное усиление окна: тон амплитуды 1.0 -> 0 дБ.
    cgain = float(np.sum(win))
    norm = cgain * cgain if cgain > 0 else 1.0

    step = max(1, int(n * (1.0 - float(np.clip(overlap, 0.0, 0.95)))))
    max_seg = max(1, int(averages))

    acc = np.zeros(n, dtype=np.float64)
    count = 0
    start = 0
    while start + n <= x.size and count < max_seg:
        seg = x[start:start + n].astype(np.complex128) * win
        spec = np.fft.fftshift(np.fft.fft(seg))
        acc += (spec.real ** 2 + spec.imag ** 2)
        count += 1
        start += step

    if count == 0:  # страховка
        seg = x[:n].astype(np.complex128) * win
        spec = np.fft.fftshift(np.fft.fft(seg))
        acc = spec.real ** 2 + spec.imag ** 2
        count = 1

    psd = acc / (count * norm)
    psd_db = 10.0 * np.log10(psd + _EPS)
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / float(sample_rate)))
    return freqs, psd_db


# ----------------------------------------------------------------------
# Поиск пиков (без scipy)
# ----------------------------------------------------------------------

def _local_maxima(x: np.ndarray, order: int) -> np.ndarray:
    """Индексы локальных максимумов: элемент не меньше соседей в окне."""
    n = x.size
    if n < 3:
        return np.arange(n)
    order = max(1, min(order, (n - 1) // 2))
    width = 2 * order + 1
    padded = np.pad(x, order, mode="constant", constant_values=-np.inf)
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    is_max = x >= windows.max(axis=1)
    return np.flatnonzero(is_max)


def _prominence_db(x: np.ndarray, idx: int) -> float:
    """Выступаемость пика над наибольшей из ближайших седловин.

    Плато обрабатывается корректно: пока значение не больше вершины,
    мы продолжаем движение наружу и ищем минимум (нужно для плоских
    цифровых сигналов, у которых вершина занимает несколько бинов).
    """
    n = x.size
    peak = float(x[idx])
    left_min = peak
    i = idx - 1
    while i >= 0 and x[i] <= peak:
        if x[i] < left_min:
            left_min = float(x[i])
        i -= 1
    right_min = peak
    j = idx + 1
    while j < n and x[j] <= peak:
        if x[j] < right_min:
            right_min = float(x[j])
        j += 1
    return float(peak - max(left_min, right_min))


def _refine_index(x: np.ndarray, idx: int, drop_db: float = 0.75) -> int:
    """Уточнить позицию пика: центроид плато в пределах ``drop_db`` от вершины."""
    n = x.size
    thr = float(x[idx]) - drop_db
    lo = idx
    while lo - 1 >= 0 and x[lo - 1] >= thr:
        lo -= 1
    hi = idx
    while hi + 1 < n and x[hi + 1] >= thr:
        hi += 1
    if hi <= lo:
        return int(idx)
    weights = np.power(10.0, x[lo:hi + 1] / 10.0)
    positions = np.arange(lo, hi + 1, dtype=np.float64)
    return int(round(float((weights * positions).sum() / weights.sum())))


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    """Сглаживание «скользящим средним» без искажения краёв массива."""
    if window < 3 or x.size < window:
        return x
    kernel = np.ones(window, dtype=np.float64)
    num = np.convolve(x, kernel, mode="same")
    den = np.convolve(np.ones_like(x), kernel, mode="same")
    return num / np.maximum(den, 1.0)


def _band_edges(
    freqs: np.ndarray, psd_db: np.ndarray, index: int, drop_db: float = 6.0
):
    """Границы «горба» пика: (Гц, Гц) по спаду ``drop_db`` дБ в обе стороны.

    Спад отсчитывается по слегка сглаженному спектру: это делает оценку
    устойчивой к «ряби» на плоских вершинах широкополосных сигналов
    (Wi-Fi, LTE), где одиночный провал иначе обрывал бы измерение.
    """
    n = psd_db.size
    if n < 3 or index < 0 or index >= n or freqs.size != n:
        return None
    work = _smooth(psd_db.astype(np.float64), max(3, (n // 256) | 1))
    thr = float(psd_db[index]) - abs(float(drop_db))

    lo = int(index)
    while lo - 1 >= 0 and work[lo - 1] >= thr:
        lo -= 1
    hi = int(index)
    while hi + 1 < n and work[hi + 1] >= thr:
        hi += 1
    return float(freqs[lo]), float(freqs[hi])


def peak_bandwidth(
    freqs: np.ndarray, psd_db: np.ndarray, index: int, drop_db: float = 6.0
) -> float:
    """Оценка ширины пика по спаду на ``drop_db`` дБ в обе стороны (Гц)."""
    n = psd_db.size
    if n < 3:
        return 0.0
    edges = _band_edges(freqs, psd_db, index, drop_db)
    bin_w = abs(float(freqs[1] - freqs[0])) if n > 1 else 0.0
    if edges is None:
        return bin_w
    lo, hi = edges
    if hi <= lo:
        return bin_w
    return abs(hi - lo) + bin_w


def find_peaks(
    freqs: np.ndarray,
    psd_db: np.ndarray,
    threshold_db: float = -70.0,
    min_peak_distance: float = 0.0,
    min_prominence_db: float = 0.0,
    max_peaks: int = 40,
    exclude_dc_bins: int = 1,
) -> list[dict]:
    """Найти пики в спектре.

    Возвращает список словарей, отсортированный по уровню:
    ``{"freq": Гц, "power": дБ, "prominence": дБ, "bandwidth": Гц, "index": int}``.

    Пики ближе ``min_peak_distance`` подавляются (остаётся сильнейший),
    что устраняет дубли на плоских вершинах. Дополнительно максимумы,
    лежащие на одном «горбе» (провал между ними меньше
    ``min_prominence_db`` дБ), считаются одним сигналом.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    psd = np.asarray(psd_db, dtype=np.float64)
    if psd.size < 3:
        return []

    if freqs.size > 1:
        bin_w = float(np.median(np.abs(np.diff(freqs)))) or 1.0
    else:
        bin_w = 1.0

    # Окно локального максимума - половина минимальной дистанции.
    order = 1
    if min_peak_distance > 0:
        order = max(1, int(round(0.5 * min_peak_distance / bin_w)))
    order = min(order, (psd.size - 1) // 2)

    candidates = _local_maxima(psd, order)

    # Исключаем окрестность DC (нулевая частота - протечка гетеродина).
    if exclude_dc_bins > 0:
        dc = int(np.argmin(np.abs(freqs)))
        candidates = candidates[np.abs(candidates - dc) > exclude_dc_bins]

    raw: list[dict] = []
    for i in candidates:
        p = float(psd[i])
        if p < threshold_db:
            continue
        prom = _prominence_db(psd, int(i)) if min_prominence_db > 0 else 0.0
        if min_prominence_db > 0 and prom < min_prominence_db:
            continue
        raw.append({"index": int(i), "power": p, "prominence": prom})

    # Подавление близких пиков: оставляем сильнейший. Дополнительно
    # максимумы внутри «горба» более сильного пика (в пределах спада
    # ``min_prominence_db`` дБ) считаются одним широкополосным сигналом,
    # поэтому рябь на плоской вершине не даёт десятков ложных пиков.
    merge_db = float(min_prominence_db)
    raw.sort(key=lambda r: r["power"], reverse=True)
    kept: list[dict] = []
    bands: list[tuple[float, float]] = []
    for cand in raw:
        i = _refine_index(psd, cand["index"])
        f = float(freqs[i])
        if min_peak_distance > 0 and any(
            abs(f - float(freqs[k["index"]])) < min_peak_distance for k in kept
        ):
            continue
        edges = _band_edges(freqs, psd, i, drop_db=merge_db) if merge_db > 0 else None
        if edges is not None and any(lo <= f <= hi for lo, hi in bands):
            continue
        cand["index"] = i
        cand["edges"] = edges
        kept.append(cand)
        if edges is not None:
            bands.append(edges)

    # Итоговые параметры пика: частота - центр «горба» (устойчив и совпадает
    # с центром канала для широкополосных сигналов), уровень - максимум,
    # полоса - ширина горба.
    results: list[dict] = []
    bin_w_ = abs(float(freqs[1] - freqs[0])) if freqs.size > 1 else 0.0
    for cand in kept:
        idx = _refine_index(psd, cand["index"])
        edges = cand.get("edges")
        if edges is None:
            edges = _band_edges(freqs, psd, idx, drop_db=6.0)
        a = b = idx
        if edges is not None:
            a = int(np.argmin(np.abs(freqs - edges[0])))
            b = int(np.argmin(np.abs(freqs - edges[1])))
            if a > b:
                a, b = b, a
        if b > a:
            weights = np.power(10.0, psd[a:b + 1] / 10.0)
            positions = np.arange(a, b + 1, dtype=np.float64)
            top = a + int(np.argmax(psd[a:b + 1]))
            center = int(round(float((weights * positions).sum() / weights.sum())))
            width = abs(float(freqs[b]) - float(freqs[a])) + bin_w_
        else:
            top = center = idx
            width = peak_bandwidth(freqs, psd, idx)
        power = float(psd[top])
        prom = _prominence_db(psd, top) if min_prominence_db > 0 else 0.0
        results.append(
            {
                "freq": float(freqs[center]),
                "power": power,
                "prominence": prom,
                "bandwidth": width,
                "index": int(top),
            }
        )

    results.sort(key=lambda r: r["power"], reverse=True)
    return results[: max(1, int(max_peaks))]


def noise_floor_db(psd_db: np.ndarray, percentile: float = 25.0) -> float:
    """Оценка шумовой дорожки по нижнему перцентилю."""
    if psd_db.size == 0:
        return -180.0
    return float(np.percentile(psd_db, percentile))


def format_freq(freq_hz: float, precision: int = 3) -> str:
    """Человекочитаемая частота: Гц/кГц/МГц/ГГц."""
    f = float(freq_hz)
    sign = "-" if f < 0 else ""
    f = abs(f)
    if f >= 1e9:
        return f"{sign}{f/1e9:.{precision}f} ГГц"
    if f >= 1e6:
        return f"{sign}{f/1e6:.{precision}f} МГц"
    if f >= 1e3:
        return f"{sign}{f/1e3:.{precision}f} кГц"
    return f"{sign}{f:.{precision}f} Гц"
