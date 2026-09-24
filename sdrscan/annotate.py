"""Распознавание канала/диапазона по частоте.

Модуль отвечает на вопрос «на что я смотрю?»: Wi-Fi канал, полоса LTE,
сигнал ГНСС, BLE-канал, ISM и т.п. Используется для подписей пиков и таблицы.

Подбирается самое узкое подходящее определение, поэтому конкретный канал
приоритетнее широкой полосы.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import presets

MHZ = 1e6


@dataclass(frozen=True)
class BandInfo:
    """Короткая подпись диапазона и пояснение."""

    label: str
    detail: str = ""


_ENTRIES: list[tuple[float, float, BandInfo]] | None = None


def _add(out, lo_hz, hi_hz, label, detail="") -> None:
    out.append((float(lo_hz), float(hi_hz), BandInfo(label, detail)))


def _build() -> list[tuple[float, float, BandInfo]]:
    out: list[tuple[float, float, BandInfo]] = []

    # --- Wi-Fi 2.4 ГГц ---
    for ch in range(1, 14):
        fc = 2412 * MHZ + 5 * MHZ * (ch - 1)
        _add(out, fc - 10 * MHZ, fc + 10 * MHZ, f"Wi-Fi ch {ch}",
             f"2.4 ГГц, центр {fc / MHZ:.0f} МГц")
    _add(out, 2474 * MHZ, 2494 * MHZ, "Wi-Fi ch 14", "2.4 ГГц, центр 2484 МГц")

    # --- Wi-Fi 5 ГГц ---
    for ch in presets._WIFI_5_CHANNELS:
        fc = 5000 * MHZ + 5 * MHZ * ch
        _add(out, fc - 10 * MHZ, fc + 10 * MHZ, f"Wi-Fi ch {ch}",
             f"5 ГГц, центр {fc / MHZ:.0f} МГц")

    # --- Wi-Fi 6E (доступная часть) ---
    for ch in (1, 5, 9, 13, 17):
        fc = 5950 * MHZ + 5 * MHZ * ch
        if fc + 10 * MHZ > 6000 * MHZ:
            continue
        _add(out, fc - 10 * MHZ, fc + 10 * MHZ, f"Wi-Fi 6E ch {ch}",
             f"6 ГГц, центр {fc / MHZ:.0f} МГц")

    # --- BLE: рекламные каналы важнее, поэтому идут раньше ---
    for ch, fc in ((37, 2402 * MHZ), (38, 2426 * MHZ), (39, 2480 * MHZ)):
        _add(out, fc - 1 * MHZ, fc + 1 * MHZ, f"BLE adv ch {ch}",
             f"Advertising, центр {fc / MHZ:.0f} МГц")
    for ch in range(0, 37):
        fc = 2402 * MHZ + 2 * MHZ * ch
        _add(out, fc - 1 * MHZ, fc + 1 * MHZ, f"BLE ch {ch}",
             f"центр {fc / MHZ:.0f} МГц")

    # --- LTE ---
    for n, label, dl, ul in presets._LTE_FDD:
        if dl:
            lo, hi = dl
            _add(out, lo * MHZ, hi * MHZ, f"LTE B{n} DL", f"{label} МГц, базовая станция")
        if ul:
            lo, hi = ul
            _add(out, lo * MHZ, hi * MHZ, f"LTE B{n} UL", f"{label} МГц, абонент")
    for n, label, tdd in presets._LTE_TDD:
        lo, hi = tdd
        _add(out, lo * MHZ, hi * MHZ, f"LTE B{n} TDD", f"{label} МГц")

    # --- ГНСС ---
    for system, signal, fc_mhz in presets._GNSS_SIGNALS:
        fc = fc_mhz * MHZ
        _add(out, fc - 2.5 * MHZ, fc + 2.5 * MHZ, f"ГНСС {system} {signal}",
             f"центр {fc_mhz:.3f} МГц")

    # --- Прочие известные полосы ---
    others = [
        (87.5, 108, "FM-вещание", "УКВ радиовещание"),
        (174, 240, "DAB/DAB+", "цифровое радиовещание"),
        (470, 790, "DVB-T", "цифровое ТВ"),
        (925, 960, "GSM/LTE 900 DL", "базовая станция"),
        (890, 915, "GSM 900 UL", "абонент"),
        (1805, 1880, "GSM/LTE 1800 DL", "базовая станция"),
        (2110, 2170, "UMTS/LTE 2100 DL", "базовая станция"),
        (380, 400, "TETRA", "профессиональная связь"),
        (410, 430, "TETRA", "профессиональная связь"),
        (433.05, 434.79, "ISM 433", "пульты, датчики, LoRa"),
        (863, 870, "ISM/LoRa 868", "LoRa, Sigfox"),
        (902, 928, "ISM 915", "ISM, LoRa"),
        (446.0, 446.2, "PMR446", "безлицензионные рации"),
        (1880, 1900, "DECT", "беспроводные телефоны"),
        (1080, 1100, "ADS-B", "ответчики ВС, 1090 МГц"),
        (161.5, 162.5, "AIS", "морские суда"),
    ]
    for lo, hi, label, detail in others:
        _add(out, lo * MHZ, hi * MHZ, label, detail)
    return out


def entries() -> list[tuple[float, float, BandInfo]]:
    global _ENTRIES
    if _ENTRIES is None:
        _ENTRIES = _build()
    return _ENTRIES


def describe(hz: float, bandwidth_hz: float | None = None) -> BandInfo | None:
    """Наиболее подходящее определение частоты.

    Если известна измеренная ширина сигнала (``bandwidth_hz``), выбирается
    диапазон, ближайший к ней по ширине и к частоте по центру. Это важно там,
    где сетки каналов накладываются (BLE 2 МГц внутри Wi-Fi 20 МГц).
    Без ширины выбирается самое узкое определение.
    """
    f = float(hz)
    cands = [
        (hi - lo, 0.5 * (lo + hi), info)
        for lo, hi, info in entries()
        if lo <= f <= hi
    ]
    if not cands:
        return None
    bw = abs(float(bandwidth_hz or 0.0))
    if bw > 0:
        covering = [c for c in cands if c[0] >= 0.5 * bw]
        if covering:
            return min(covering, key=lambda c: (abs(c[0] - bw), abs(c[1] - f)))[2]
    return min(cands, key=lambda c: (c[0], abs(c[1] - f)))[2]


def label(hz: float, bandwidth_hz: float | None = None) -> str:
    info = describe(hz, bandwidth_hz)
    return info.label if info is not None else ""
