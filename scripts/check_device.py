"""Диагностика подключения USRP.

Запуск:
    python scripts/check_device.py [аргументы UHD]

Примеры:
    python scripts/check_device.py
    python scripts/check_device.py type=b200
    python scripts/check_device.py addr=192.168.10.2
"""

from __future__ import annotations

import sys
from pathlib import Path

# Разрешаем запуск без установки пакета: добавляем корень проекта в sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    args = sys.argv[1] if len(sys.argv) > 1 else ""

    try:
        import uhd
    except Exception as exc:  # noqa: BLE001
        print("[FAIL] import uhd:", exc)
        print("        Убедитесь, что установлен UHD-инсталлятор (UHD_PKG_PATH)")
        print("        и модуль uhd: pip install uhd==<версия>")
        return 2

    print(f"[ OK ] UHD Python API: {getattr(uhd, '__version__', '?')}")

    try:
        found = uhd.find(args)
        found = [dict(d) for d in found]
    except Exception as exc:  # noqa: BLE001
        print("[FAIL] find:", exc)
        return 1

    if not found:
        print("[ ---- ] Устройства не найдены.")
        print("         Проверьте: USB-кабель, драйвер libusb, питание, образ FPGA.")
        return 1

    print(f"[ OK ] Найдено устройств: {len(found)}")
    for i, dev in enumerate(found):
        print(f"--- устройство {i} ---")
        for k, v in dev.items():
            print(f"    {k}: {v}")

    try:
        usrp = uhd.usrp.MultiUSRP(args or "type=b200")
        print("[ OK ] MultiUSRP открыт")
        print("       mboard :", usrp.get_mboard_name())
        print("       rx rate:", usrp.get_rx_rate(0))
        print("       rx freq:", usrp.get_rx_freq(0))
        print("       антенны:", list(usrp.get_rx_antennas(0)))
    except Exception as exc:  # noqa: BLE001
        print("[FAIL] не удалось открыть MultiUSRP:", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
