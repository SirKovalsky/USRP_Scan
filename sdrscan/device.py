"""Обёртка над UHD (MultiUSRP) для USRP B210.

Модуль импортирует ``uhd`` лениво, чтобы приложение можно было запустить
(и увидеть понятную ошибку) даже там, где UHD ещё не установлен.
"""

from __future__ import annotations

import sys
import threading
from typing import Any

import numpy as np


class DeviceError(RuntimeError):
    """Ошибка работы с SDR-устройством."""


if sys.platform.startswith("win"):
    _UHD_HINT = (
        "Не удалось импортировать модуль 'uhd'. Установите UHD:\n"
        "  1) UHD-инсталлятор для Windows (он задаёт UHD_PKG_PATH и ставит uhd.dll)\n"
        "  2) pip install uhd==<версия из uhd_config_info.exe --version>\n"
        "Подробности - в README.md, раздел «Установка UHD»."
    )
else:
    _UHD_HINT = (
        "Не удалось импортировать модуль 'uhd'. Установите UHD:\n"
        "  sudo apt install libuhd-dev uhd-host python3-uhd\n"
        "  sudo uhd_images_downloader\n"
        "Виртуальное окружение должно видеть системный пакет python3-uhd\n"
        "(python -m venv --system-site-packages .venv).\n"
        "Либо соберите UHD из исходников: https://github.com/EttusResearch/uhd\n"
        "Подробности - в README.md, раздел «Установка UHD (Linux)»."
    )


def uhd_available() -> bool:
    """True, если модуль uhd импортируется."""
    try:
        import uhd  # noqa: F401
    except Exception:
        return False
    return True


def uhd_version() -> str:
    try:
        import uhd
    except Exception:
        return "не установлен"
    try:
        return str(uhd.__version__)
    except Exception:
        return "неизвестно"


def list_devices(args: str = "") -> list[dict]:
    """Найти подключённые USRP (может вернуть пустой список)."""
    try:
        import uhd
    except Exception as exc:  # pragma: no cover
        raise DeviceError(_UHD_HINT) from exc
    try:
        found = uhd.find(args)
        return [dict(d) for d in found]
    except Exception as exc:  # pragma: no cover
        raise DeviceError(f"Поиск устройств не удался: {exc}") from exc


class USRPDevice:
    """Тонкая обёртка вокруг ``uhd.usrp.MultiUSRP`` для одного RX-канала."""

    def __init__(self, args: str = "type=b200", channel: int = 0) -> None:
        try:
            import uhd
        except Exception as exc:
            raise DeviceError(_UHD_HINT) from exc

        self._uhd = uhd
        self.args = args
        self.channel = int(channel)
        self._lock = threading.RLock()
        self._streamer: Any = None
        self._rx_buff: np.ndarray | None = None
        self._max_samps = 0
        self._rate: float | None = None
        self._streaming = False

        with self._lock:
            self.usrp = uhd.usrp.MultiUSRP(args)

    # ------------------------------------------------------------------
    # Настройки
    # ------------------------------------------------------------------
    def set_sample_rate(self, rate: float) -> float:
        with self._lock:
            self.usrp.set_rx_rate(float(rate), self.channel)
            self._rate = float(self.usrp.get_rx_rate(self.channel))
            self._make_streamer()
        return self._rate

    def set_center_freq(self, freq: float) -> float:
        with self._lock:
            req = self._uhd.types.TuneRequest(float(freq))
            self.usrp.set_rx_freq(req, self.channel)
            return float(self.usrp.get_rx_freq(self.channel))

    def set_gain(self, gain_db: float) -> float:
        with self._lock:
            self.usrp.set_rx_gain(float(gain_db), self.channel)
            return float(self.usrp.get_rx_gain(self.channel))

    def set_antenna(self, antenna: str) -> None:
        with self._lock:
            self.usrp.set_rx_antenna(str(antenna), self.channel)

    def set_bandwidth(self, bw: float) -> float:
        with self._lock:
            self.usrp.set_rx_bandwidth(float(bw), self.channel)
            return float(self.usrp.get_rx_bandwidth(self.channel))

    def get_antennas(self) -> list[str]:
        with self._lock:
            try:
                return list(self.usrp.get_rx_antennas(self.channel))
            except Exception:
                return ["RX2", "RX1", "TX/RX"]

    @property
    def sample_rate(self) -> float:
        return float(self._rate or 0.0)

    # ------------------------------------------------------------------
    # Поток
    # ------------------------------------------------------------------
    def _make_streamer(self) -> None:
        uhd = self._uhd
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = [self.channel]
        self._streamer = self.usrp.get_rx_stream(st_args)
        self._max_samps = int(self._streamer.get_max_num_samps())
        self._rx_buff = np.zeros((1, self._max_samps), dtype=np.complex64)

    def _ensure_streamer(self) -> None:
        if self._streamer is None:
            self._make_streamer()

    def recv_samples(self, num: int, timeout: float = 1.0) -> np.ndarray:
        """Забрать ровно ``num`` сэмплов (одноразовый поток, num_done)."""
        uhd = self._uhd
        num = int(num)
        if num <= 0:
            return np.zeros(0, dtype=np.complex64)

        with self._lock:
            self._ensure_streamer()
            cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
            cmd.num_samps = num
            cmd.stream_now = True
            self._streamer.issue_stream_cmd(cmd)

            out = np.empty(num, dtype=np.complex64)
            got = 0
            ec = uhd.types.RXMetadataErrorCode
            while got < num:
                md = uhd.types.RXMetadata()
                nread = int(self._streamer.recv(self._rx_buff, md, float(timeout)))
                if md.error_code == ec.timeout and nread == 0:
                    break
                if nread > 0:
                    out[got:got + nread] = self._rx_buff[0, :nread]
                    got += nread
            return out[:got].copy()

    def start_continuous(self) -> None:
        uhd = self._uhd
        with self._lock:
            self._ensure_streamer()
            cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
            cmd.stream_now = True
            self._streamer.issue_stream_cmd(cmd)
            self._streaming = True

    def stop_continuous(self) -> None:
        if not self._streaming:
            return
        uhd = self._uhd
        with self._lock:
            try:
                cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
                self._streamer.issue_stream_cmd(cmd)
            except Exception:
                pass
            self._streaming = False

    def recv_block(self, num: int, timeout: float = 0.25) -> np.ndarray:
        """Забрать до ``num`` сэмплов из непрерывного потока."""
        uhd = self._uhd
        chunks: list[np.ndarray] = []
        got = 0
        with self._lock:
            self._ensure_streamer()
            while got < num:
                md = uhd.types.RXMetadata()
                nread = int(self._streamer.recv(self._rx_buff, md, float(timeout)))
                if nread <= 0:
                    break
                chunks.append(self._rx_buff[0, :nread].copy())
                got += nread
        if not chunks:
            return np.zeros(0, dtype=np.complex64)
        return np.concatenate(chunks)

    # ------------------------------------------------------------------
    def get_info(self) -> dict:
        info: dict[str, Any] = {
            "args": self.args,
            "uhd_version": uhd_version(),
            "channel": self.channel,
        }
        with self._lock:
            for key, fn in (
                ("mboard", lambda: self.usrp.get_mboard_name()),
                ("subdev", lambda: self.usrp.get_rx_subdev_name(self.channel)),
                ("antenna", lambda: self.usrp.get_rx_antenna(self.channel)),
                ("rate", lambda: self.usrp.get_rx_rate(self.channel)),
                ("freq", lambda: self.usrp.get_rx_freq(self.channel)),
                ("gain", lambda: self.usrp.get_rx_gain(self.channel)),
            ):
                try:
                    info[key] = fn()
                except Exception:
                    pass
            try:
                tree = self.usrp.get_usrp_rx_info(self.channel)
                info["serial"] = tree.get("mboard_serial", "")
            except Exception:
                pass
        return info

    def close(self) -> None:
        try:
            self.stop_continuous()
        except Exception:
            pass
        with self._lock:
            self._streamer = None
            self._rx_buff = None
        self.usrp = None
