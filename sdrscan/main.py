"""Точка входа приложения SDR Scan."""

from __future__ import annotations

import argparse
import sys

from .config import AcquisitionConfig
from .device import uhd_available, uhd_version

# ВАЖНО: модуль uhd уже предзагружен в sdrscan/__init__.py ДО библиотек Qt.
# Импорты PyQt5/pyqtgraph выполняются лениво, только когда нужен GUI, чтобы
# гарантировать корректный порядок загрузки DLL на Windows (иначе - 0xC0000005).


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sdrscan",
        description="Анализатор спектра для USRP B210 (UHD).",
    )
    p.add_argument("--mode", choices=("sweep", "realtime", "gnss"), default=None)
    p.add_argument("--start", type=float, default=None, help="начало свипа, МГц")
    p.add_argument("--stop", type=float, default=None, help="конец свипа, МГц")
    p.add_argument("--center", type=float, default=None, help="центр, МГц (real-time)")
    p.add_argument("--rate", type=float, default=None, help="частота дискр., МГц")
    p.add_argument("--gain", type=float, default=None, help="усиление, дБ")
    p.add_argument("--antenna", default=None, help="антенна RX2/RX1/TX-RX")
    p.add_argument("--fft", type=int, default=None, help="размер FFT")
    p.add_argument("--averages", type=int, default=None, help="накоплений на точку")
    p.add_argument("--overlap", type=float, default=None, help="перекрытие 0..0.9")
    p.add_argument("--threshold", type=float, default=None, help="порог пиков, дБ")
    p.add_argument("--args", dest="device_args", default=None, help="аргументы UHD")
    p.add_argument("--list-devices", action="store_true",
                   help="показать подключённые USRP и выйти")
    return p


def _config_from_args(args: argparse.Namespace) -> AcquisitionConfig:
    cfg = AcquisitionConfig()
    if args.mode is not None:
        cfg = cfg.copy(mode=args.mode)
    if args.start is not None:
        cfg = cfg.copy(start_freq=args.start * 1e6)
    if args.stop is not None:
        cfg = cfg.copy(stop_freq=args.stop * 1e6)
    if args.center is not None:
        cfg = cfg.copy(center_freq=args.center * 1e6)
    if args.rate is not None:
        cfg = cfg.copy(sample_rate=args.rate * 1e6)
    if args.gain is not None:
        cfg = cfg.copy(gain=args.gain)
    if args.antenna is not None:
        cfg = cfg.copy(antenna=args.antenna)
    if args.fft is not None:
        cfg = cfg.copy(fft_size=int(args.fft))
    if args.averages is not None:
        cfg = cfg.copy(averages=int(args.averages))
    if args.overlap is not None:
        cfg = cfg.copy(overlap=float(args.overlap))
    if args.threshold is not None:
        cfg = cfg.copy(threshold_db=float(args.threshold))
    if args.device_args is not None:
        cfg = cfg.copy(device_args=args.device_args)
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print(f"UHD: {uhd_version()}")

    if args.list_devices:
        if not uhd_available():
            print("Модуль 'uhd' не установлен. См. README.md.")
            return 2
        from .device import list_devices
        try:
            found = list_devices(args.device_args or "")
        except Exception as exc:
            print(f"Ошибка поиска: {exc}")
            return 1
        if not found:
            print("USRP не найдены.")
            return 1
        for i, dev in enumerate(found):
            print(f"--- устройство {i} ---")
            for k, v in dev.items():
                print(f"  {k}: {v}")
        return 0

    cfg = _config_from_args(args)

    # Qt подключаем здесь: uhd уже загружен (см. sdrscan/__init__.py).
    import pyqtgraph as pg
    from PyQt5 import QtWidgets

    pg.setConfigOptions(antialias=False, background="#101014", foreground="#d8d8d8")
    app = QtWidgets.QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    app.setApplicationName("SDR Scan")

    from .gui import MainWindow

    window = MainWindow(cfg)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
