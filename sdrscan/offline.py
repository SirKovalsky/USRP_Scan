"""Чтение записанных файлов I/Q (см. recorder.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .recorder import bytes_per_sample


def read_metadata(iq_path: str) -> dict | None:
    """Прочитать соседний ``.json`` с метаданными, если он есть."""
    p = Path(iq_path)
    candidate = p.with_suffix(".json")
    if not candidate.exists():
        return None
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return None


def iq_sample_count(iq_path: str, metadata: dict | None = None) -> int:
    """Сколько комплексных отсчётов в файле."""
    fmt = (metadata or {}).get("format", "complex64")
    size = Path(iq_path).stat().st_size
    return int(size // bytes_per_sample(fmt))


def read_iq(
    iq_path: str,
    metadata: dict | None = None,
    start: int = 0,
    count: int | None = None,
) -> np.ndarray:
    """Прочитать ``count`` комплексных отсчётов, начиная с ``start``."""
    fmt = (metadata or {}).get("format", "complex64")
    if fmt == "int16":
        raw = np.fromfile(
            iq_path, dtype=np.int16,
            count=-1 if count is None else int(count) * 2,
            offset=int(start) * 4,
        )
        if raw.size < 2:
            return np.zeros(0, dtype=np.complex64)
        return (
            raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
        ).astype(np.complex64) / 32768.0
    raw = np.fromfile(
        iq_path, dtype=np.float32,
        count=-1 if count is None else int(count) * 2,
        offset=int(start) * 8,
    )
    if raw.size < 2:
        return np.zeros(0, dtype=np.complex64)
    return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
