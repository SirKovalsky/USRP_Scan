"""Запись потока I/Q в файл вместе с метаданными.

Формат: сырые чередующиеся I,Q без заголовка.
  * ``complex64`` - float32 I, float32 Q (8 байт на отсчёт, точнее);
  * ``int16`` - int16 I, int16 Q (4 байта на отсчёт, компактнее).

Рядом создаётся ``<имя>.json`` с параметрами приёма, чтобы файл был
самодостаточным для офлайн-разбора.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

FORMATS = ("complex64", "int16")
_BYTES_PER_SAMPLE = {"complex64": 8, "int16": 4}


class IQRecorder:
    """Пишет отсчёты в ``<base>.iq`` и метаданные в ``<base>.json``."""

    def __init__(self, base_path: str, metadata: dict | None = None,
                 fmt: str = "complex64") -> None:
        p = Path(base_path)
        if p.suffix.lower() == ".iq":
            p = p.with_suffix("")
        self.base = p
        self.iq_path = p.with_suffix(".iq")
        self.json_path = p.with_suffix(".json")
        self.fmt = fmt if fmt in FORMATS else "complex64"
        self.metadata = dict(metadata or {})
        self.samples = 0
        self.started: datetime | None = None
        self._fh = None

    # ------------------------------------------------------------------
    @property
    def recording(self) -> bool:
        return self._fh is not None

    def open(self) -> "IQRecorder":
        self.iq_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.iq_path, "wb")
        self.started = datetime.now()
        return self

    def write(self, samples: np.ndarray) -> None:
        if self._fh is None:
            return
        x = np.asarray(samples)
        if x.size == 0:
            return
        if self.fmt == "int16":
            i = np.clip(x.real, -1.0, 1.0) * 32767.0
            q = np.clip(x.imag, -1.0, 1.0) * 32767.0
            out = np.empty(x.size * 2, dtype=np.int16)
            out[0::2] = i.astype(np.int16)
            out[1::2] = q.astype(np.int16)
        else:
            out = np.empty(x.size * 2, dtype=np.float32)
            out[0::2] = x.real.astype(np.float32)
            out[1::2] = x.imag.astype(np.float32)
        self._fh.write(np.ascontiguousarray(out).tobytes())
        self.samples += int(x.size)

    def close(self) -> Path | None:
        if self._fh is None:
            return None
        self._fh.flush()
        self._fh.close()
        self._fh = None

        rate = float(self.metadata.get("sample_rate", 0.0) or 0.0)
        size = self.iq_path.stat().st_size if self.iq_path.exists() else 0
        meta = dict(self.metadata)
        meta.update(
            {
                "format": self.fmt,
                "components": "interleaved IQ (I,Q,I,Q,...)",
                "component_dtype": "float32" if self.fmt == "complex64" else "int16",
                "samples": self.samples,
                "size_bytes": size,
                "started": self.started.isoformat(timespec="seconds") if self.started else "",
                "finished": datetime.now().isoformat(timespec="seconds"),
                "duration_s": (self.samples / rate) if rate > 0 else 0.0,
            }
        )
        try:
            self.json_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass
        return self.json_path


def bytes_per_sample(fmt: str) -> int:
    return _BYTES_PER_SAMPLE.get(fmt, 8)
