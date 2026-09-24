#!/usr/bin/env bash
# Установка SDR Scan на Linux (Ubuntu/Debian). Требуется sudo для пакетов UHD.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/4. Системные пакеты UHD (нужен sudo) =="
sudo apt update
sudo apt install -y libuhd-dev uhd-host python3-uhd python3-venv python3-pip

echo "== 2/4. Образы FPGA и прошивки USRP =="
sudo uhd_images_downloader || true

echo "== 3/4. Правила udev (доступ к USRP по USB без root) =="
RULES=/lib/uhd/utils/uhd-usrp.rules
if [ -f "$RULES" ]; then
    sudo cp "$RULES" /etc/udev/rules.d/
    sudo udevadm control --reload-rules
    sudo udevadm trigger
else
    echo "   $RULES не найден (пакет uhd-host?) — пропускаю"
fi

echo "== 4/4. Виртуальное окружение Python =="
# --system-site-packages нужен, чтобы в venv был виден системный модуль uhd:
# python3-uhd ставится через apt, а не через pip.
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

cat <<'EOF'

Готово.
  Проверка железа : .venv/bin/python scripts/check_device.py
  Запуск          : ./run.sh
  Список устройств: ./run.sh --list-devices

Если окно не появляется под Wayland:
  QT_QPA_PLATFORM=xcb ./run.sh
EOF
