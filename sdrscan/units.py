"""Единицы измерения частоты: единый выбор МГц / кГц / ГГц / Гц.

Значения внутри приложения всегда хранятся в герцах. Эти функции только
переводят их в выбранную единицу для полей ввода и подписей.
"""

from __future__ import annotations

UNITS = ("ГГц", "МГц", "кГц", "Гц")
DEFAULT = "МГц"

_FACTORS = {"ГГц": 1e9, "МГц": 1e6, "кГц": 1e3, "Гц": 1.0}
# Сколько знаков после запятой разумно показывать в каждой единице.
_DECIMALS = {"ГГц": 6, "МГц": 3, "кГц": 1, "Гц": 0}


def factor(unit: str) -> float:
    """Множитель единицы к герцам."""
    return _FACTORS.get(unit, 1e6)


def decimals(unit: str) -> int:
    return _DECIMALS.get(unit, 3)


def to_hz(value: float, unit: str) -> float:
    return float(value) * factor(unit)


def from_hz(hz: float, unit: str) -> float:
    return float(hz) / factor(unit)


def auto_unit(hz: float) -> str:
    """Наиболее читаемая единица для величины."""
    a = abs(float(hz))
    if a >= 1e9:
        return "ГГц"
    if a >= 1e6:
        return "МГц"
    if a >= 1e3:
        return "кГц"
    return "Гц"


def format_hz(hz: float, unit: str | None = None, decimals_: int | None = None) -> str:
    """Строка вида «99.900 МГц»."""
    u = unit or auto_unit(hz)
    d = decimals(u) if decimals_ is None else int(decimals_)
    return f"{from_hz(hz, u):.{d}f} {u}"


def format_span(hz: float) -> str:
    """Ширина полосы: «200.0 кГц», «1.500 МГц», «20.000 МГц»."""
    a = abs(float(hz))
    if a >= 1e6:
        return f"{a / 1e6:.3f} МГц"
    if a >= 1e3:
        return f"{a / 1e3:.1f} кГц"
    return f"{a:.0f} Гц"
