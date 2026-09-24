"""Главное окно анализатора спектра (PyQt5 + pyqtgraph)."""

from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from . import annotate, dsp, offline, presets, units
from .config import MAX_B210_FREQ, MIN_B210_FREQ, AcquisitionConfig
from .device import list_devices, uhd_available, uhd_version
from .gnss import STATUS_COLOR, STATUS_TEXT, GnssMonitor
from .waterfall import WaterfallWidget
from .worker import AcquisitionWorker, OfflineWorker


class MainWindow(QtWidgets.QMainWindow):
    MODES = (
        ("sweep", "Свип по частотам"),
        ("realtime", "Реальное время"),
        ("gnss", "Мониторинг ГНСС (помехи)"),
    )
    RECORD_FORMATS = (("complex64", "complex64 (точнее, 8 байт)"),
                      ("int16", "int16 (компактнее, 4 байта)"))

    def __init__(self, config: AcquisitionConfig | None = None,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("SDR Scan — анализатор спектра (USRP B210)")
        self.resize(1440, 940)

        self.worker: AcquisitionWorker | None = None
        self.offline: OfflineWorker | None = None
        self._running = False
        self._last_freqs = np.zeros(0)
        self._last_psd = np.zeros(0)
        self._last_peak_time = 0.0

        self._tracks: list[dict] = []
        self._track_next_id = 1
        self._peak_labels: list[pg.TextItem] = []
        self._table_items: list[list[QtWidgets.QTableWidgetItem]] = []

        self._freq_unit = units.DEFAULT
        self._freq_spins: list[dict] = []
        self._gnss_monitor = GnssMonitor()
        self._gnss_hist_noise: list[float] = []
        self._gnss_hist_power: list[float] = []
        self._gnss_last_log = 0.0
        self._gnss_last_status = ""

        self._recording = False
        self._paused = False
        self._selected_peak: dict | None = None
        self._table_rows: list[dict] = []

        # Кэш применённых диапазонов осей: чтобы график не «дёргался» каждый кадр.
        self._x_applied_hz: tuple[float, float] | None = None
        self._y_range_applied: tuple[float, float] | None = None

        self._build_ui()
        self._build_toolbar()
        self._build_status_bar()
        self._connect_controls()
        self._apply_freq_unit(self._freq_unit)

        cfg = config or AcquisitionConfig()
        self._load_config(cfg)
        self._set_running(False)
        self.statusBar().showMessage(
            f"UHD: {uhd_version()} | устройство не запущено"
        )

    # ==================================================================
    # Построение интерфейса
    # ==================================================================
    def _build_ui(self) -> None:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_plots())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([440, 1000])

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(splitter)
        self.setCentralWidget(central)

    def _build_left_panel(self) -> QtWidgets.QWidget:
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_settings_tab(), "Настройки")
        self.peaks_page = self._build_peaks_tab()
        tabs.addTab(self.peaks_page, "Пики")
        self.gnss_page = self._build_gnss_tab()
        tabs.addTab(self.gnss_page, "ГНСС")
        self.tabs = tabs
        return tabs

    def _build_settings_tab(self) -> QtWidgets.QWidget:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        host = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(host)

        vbox.addWidget(self._build_presets_group())
        vbox.addWidget(self._build_units_group())
        vbox.addWidget(self._build_device_group())
        vbox.addWidget(self._build_sweep_group())
        vbox.addWidget(self._build_processing_group())
        vbox.addWidget(self._build_detection_group())
        vbox.addWidget(self._build_labels_group())
        vbox.addWidget(self._build_display_group())
        vbox.addWidget(self._build_record_group())
        vbox.addStretch(1)

        buttons = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("▶ Старт")
        self.stop_btn = QtWidgets.QPushButton("■ Стоп")
        self.find_btn = QtWidgets.QPushButton("Найти")
        buttons.addWidget(self.start_btn)
        buttons.addWidget(self.stop_btn)
        buttons.addWidget(self.find_btn)
        vbox.addLayout(buttons)

        scroll.setWidget(host)
        return scroll

    # ------------------------------------------------------------------
    def _register_freq_spin(self, spin: QtWidgets.QDoubleSpinBox,
                            min_hz: float, max_hz: float) -> None:
        self._freq_spins.append(
            {"spin": spin, "min_hz": float(min_hz), "max_hz": float(max_hz)}
        )

    def _hz_of(self, spin: QtWidgets.QDoubleSpinBox) -> float:
        return units.to_hz(spin.value(), self._freq_unit)

    def _set_spin_hz(self, spin: QtWidgets.QDoubleSpinBox, hz: float) -> None:
        spin.setValue(units.from_hz(hz, self._freq_unit))

    def _fmt_freq(self, hz: float) -> str:
        return units.format_hz(hz, self._freq_unit)

    def _build_units_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Единицы измерения")
        form = QtWidgets.QFormLayout(box)
        self.unit_combo = QtWidgets.QComboBox()
        self.unit_combo.addItems(list(units.UNITS))
        self.unit_combo.setCurrentText(units.DEFAULT)
        hint = QtWidgets.QLabel(
            "Единица для полей ввода, осей графиков, меток пиков и таблицы."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #8f8f8f;")
        form.addRow("Частота", self.unit_combo)
        form.addRow(hint)
        self.units_group = box
        return box

    def _build_presets_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Диапазоны (предустановки)")
        form = QtWidgets.QFormLayout(box)

        self.preset_cat_combo = QtWidgets.QComboBox()
        self.preset_cat_combo.addItems(presets.category_names())
        self.preset_cat_combo.setToolTip(
            "Готовые диапазоны: BLE, Wi-Fi по каналам, LTE по бэндам, ГНСС."
        )

        self.preset_item_combo = QtWidgets.QComboBox()
        self.preset_item_combo.setToolTip(
            "Выберите диапазон (двойной клик или кнопка ниже - применить)."
        )

        self.preset_apply_btn = QtWidgets.QPushButton("Применить диапазон")

        self.preset_note = QtWidgets.QLabel("")
        self.preset_note.setWordWrap(True)
        self.preset_note.setStyleSheet("color: #8f8f8f;")

        form.addRow("Категория", self.preset_cat_combo)
        form.addRow("Диапазон", self.preset_item_combo)
        form.addRow(self.preset_apply_btn)
        form.addRow(self.preset_note)

        self.preset_cat_combo.currentIndexChanged.connect(self._on_preset_category)
        self.preset_item_combo.currentIndexChanged.connect(self._update_preset_note)
        self.preset_item_combo.activated.connect(self._apply_preset)
        self.preset_apply_btn.clicked.connect(self._apply_preset)

        self._reload_preset_items()
        self.presets_group = box
        return box

    def _reload_preset_items(self) -> None:
        items = presets.presets_for(self.preset_cat_combo.currentText())
        self.preset_item_combo.blockSignals(True)
        self.preset_item_combo.clear()
        for preset in items:
            self.preset_item_combo.addItem(preset.name)
        self.preset_item_combo.blockSignals(False)
        self.preset_item_combo.setCurrentIndex(0)
        self._update_preset_note()

    def _on_preset_category(self, _index: int) -> None:
        self._reload_preset_items()

    def _current_preset(self) -> "presets.Preset | None":
        items = presets.presets_for(self.preset_cat_combo.currentText())
        idx = self.preset_item_combo.currentIndex()
        if 0 <= idx < len(items):
            return items[idx]
        return None

    def _update_preset_note(self, *_args) -> None:
        preset = self._current_preset()
        self.preset_note.setText(preset.note if preset is not None else "")

    def _apply_preset(self, *_args) -> None:
        preset = self._current_preset()
        if preset is None:
            return
        lo = max(preset.start_hz, MIN_B210_FREQ)
        hi = min(preset.stop_hz, MAX_B210_FREQ)
        if hi <= lo:
            QtWidgets.QMessageBox.warning(
                self,
                "Диапазон",
                "Диапазон вне полосы приёма B210 (70 МГц .. 6 ГГц).",
            )
            return

        self._set_spin_hz(self.start_spin, lo)
        self._set_spin_hz(self.stop_spin, hi)
        self._set_spin_hz(self.center_spin, 0.5 * (lo + hi))
        if preset.rate_hz:
            rate_mhz = preset.rate_hz / 1e6
            if self.rate_spin.minimum() <= rate_mhz <= self.rate_spin.maximum():
                self.rate_spin.setValue(rate_mhz)

        note = preset.note or preset.name
        if hi < preset.stop_hz or lo > preset.start_hz:
            note += " (обрезано до полосы B210)"
        self._apply_display_span(force=True)
        self.statusBar().showMessage(
            f"Диапазон: {dsp.format_freq(lo)} .. {dsp.format_freq(hi)} | {note}"
        )

    def _build_device_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Устройство")
        form = QtWidgets.QFormLayout(box)

        self.mode_combo = QtWidgets.QComboBox()
        for key, label in self.MODES:
            self.mode_combo.addItem(label, key)
        self.mode_combo.setToolTip(
            "Свип - обзор широкого диапазона; реальное время - один канал;\n"
            "ГНСС - слежение за полосой спутников и поиск помех."
        )

        self.device_args_edit = QtWidgets.QLineEdit("type=b200")
        self.channel_spin = QtWidgets.QSpinBox()
        self.channel_spin.setRange(0, 1)
        self.antenna_combo = QtWidgets.QComboBox()
        self.antenna_combo.setEditable(True)
        self.antenna_combo.addItems(["RX2", "RX1", "TX/RX"])

        self.gain_spin = QtWidgets.QDoubleSpinBox()
        self.gain_spin.setRange(0.0, 76.0)
        self.gain_spin.setSuffix(" дБ")
        self.gain_spin.setDecimals(1)

        self.rate_spin = QtWidgets.QDoubleSpinBox()
        self.rate_spin.setRange(0.1, 61.44)
        self.rate_spin.setSuffix(" МГц")
        self.rate_spin.setDecimals(3)

        form.addRow("Режим", self.mode_combo)
        form.addRow("Аргументы UHD", self.device_args_edit)
        form.addRow("Канал RX", self.channel_spin)
        form.addRow("Антенна", self.antenna_combo)
        form.addRow("Усиление", self.gain_spin)
        form.addRow("Частота дискр.", self.rate_spin)
        self.device_group = box
        return box

    def _build_sweep_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Свип")
        form = QtWidgets.QFormLayout(box)

        self.start_spin = QtWidgets.QDoubleSpinBox()
        self.stop_spin = QtWidgets.QDoubleSpinBox()
        for spin in (self.start_spin, self.stop_spin):
            spin.setRange(70.0, 6000.0)
            spin.setSuffix(" МГц")
            spin.setDecimals(3)
            self._register_freq_spin(spin, MIN_B210_FREQ, MAX_B210_FREQ)

        self.step_spin = QtWidgets.QDoubleSpinBox()
        self.step_spin.setRange(0.0, 100000.0)
        self.step_spin.setSuffix(" кГц")
        self.step_spin.setDecimals(1)
        self.step_spin.setToolTip("0 = авто (по полосе приёма и перекрытию)")
        self._register_freq_spin(self.step_spin, 0.0, 100e6)

        self.settle_spin = QtWidgets.QSpinBox()
        self.settle_spin.setRange(0, 1000)
        self.settle_spin.setSuffix(" мс")

        self.center_spin = QtWidgets.QDoubleSpinBox()
        self.center_spin.setRange(70.0, 6000.0)
        self.center_spin.setSuffix(" МГц")
        self.center_spin.setDecimals(3)
        self._register_freq_spin(self.center_spin, MIN_B210_FREQ, MAX_B210_FREQ)

        form.addRow("Начало", self.start_spin)
        form.addRow("Конец", self.stop_spin)
        form.addRow("Шаг", self.step_spin)
        form.addRow("Установление", self.settle_spin)
        form.addRow("Центр (реал. время/ГНСС)", self.center_spin)
        self.sweep_group = box
        return box

    def _build_processing_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Обработка")
        form = QtWidgets.QFormLayout(box)

        self.fft_combo = QtWidgets.QComboBox()
        self.fft_combo.addItems(["512", "1024", "2048", "4096", "8192", "16384"])

        self.averages_spin = QtWidgets.QSpinBox()
        self.averages_spin.setRange(1, 64)

        self.overlap_spin = QtWidgets.QSpinBox()
        self.overlap_spin.setRange(0, 90)
        self.overlap_spin.setSuffix(" %")

        self.window_combo = QtWidgets.QComboBox()
        self.window_combo.addItems(["hann", "hamming", "blackman-harris", "rect"])

        form.addRow("Размер FFT", self.fft_combo)
        form.addRow("Накоплений", self.averages_spin)
        form.addRow("Перекрытие", self.overlap_spin)
        form.addRow("Окно", self.window_combo)
        self.processing_group = box
        return box

    def _build_detection_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Детектор пиков")
        form = QtWidgets.QFormLayout(box)

        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setRange(-160.0, 0.0)
        self.threshold_spin.setSuffix(" дБ")
        self.threshold_spin.setDecimals(1)

        self.distance_spin = QtWidgets.QDoubleSpinBox()
        self.distance_spin.setRange(0.0, 100000.0)
        self.distance_spin.setSuffix(" кГц")
        self.distance_spin.setDecimals(1)
        self._register_freq_spin(self.distance_spin, 0.0, 100e6)

        self.prominence_spin = QtWidgets.QDoubleSpinBox()
        self.prominence_spin.setRange(0.0, 40.0)
        self.prominence_spin.setSuffix(" дБ")
        self.prominence_spin.setDecimals(1)

        self.maxpeaks_spin = QtWidgets.QSpinBox()
        self.maxpeaks_spin.setRange(1, 200)

        form.addRow("Порог", self.threshold_spin)
        form.addRow("Мин. расстояние", self.distance_spin)
        form.addRow("Мин. выступаемость", self.prominence_spin)
        form.addRow("Макс. пиков", self.maxpeaks_spin)
        self.detection_group = box
        return box

    def _build_labels_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Метки пиков")
        form = QtWidgets.QFormLayout(box)

        self.labels_check = QtWidgets.QCheckBox("Показывать метки на графике")
        self.labels_check.setChecked(True)

        self.label_count_spin = QtWidgets.QSpinBox()
        self.label_count_spin.setRange(0, 20)
        self.label_count_spin.setValue(6)

        self.lbl_level_check = QtWidgets.QCheckBox("Уровень")
        self.lbl_level_check.setChecked(True)
        self.lbl_prom_check = QtWidgets.QCheckBox("Выступаемость")
        self.lbl_bw_check = QtWidgets.QCheckBox("Полоса")
        self.lbl_channel_check = QtWidgets.QCheckBox("Канал / диапазон")
        self.lbl_channel_check.setChecked(True)

        form.addRow(self.labels_check)
        form.addRow("Сколько меток", self.label_count_spin)
        form.addRow(self.lbl_level_check)
        form.addRow(self.lbl_prom_check)
        form.addRow(self.lbl_bw_check)
        form.addRow(self.lbl_channel_check)
        self.labels_group = box
        return box

    def _build_display_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Отображение")
        form = QtWidgets.QFormLayout(box)

        self.auto_y_check = QtWidgets.QCheckBox("Авто-диапазон по Y")
        self.auto_x_check = QtWidgets.QCheckBox("Авто-масштаб по X")
        self.auto_x_check.setChecked(True)

        self.ymin_spin = QtWidgets.QDoubleSpinBox()
        self.ymax_spin = QtWidgets.QDoubleSpinBox()
        for spin in (self.ymin_spin, self.ymax_spin):
            spin.setRange(-200.0, 40.0)
            spin.setSuffix(" дБ")
            spin.setDecimals(1)
        self.ymin_spin.setValue(-120.0)
        self.ymax_spin.setValue(0.0)

        self.wf_rows_spin = QtWidgets.QSpinBox()
        self.wf_rows_spin.setRange(50, 1000)

        self.wf_lo_spin = QtWidgets.QDoubleSpinBox()
        self.wf_hi_spin = QtWidgets.QDoubleSpinBox()
        for spin in (self.wf_lo_spin, self.wf_hi_spin):
            spin.setRange(-200.0, 40.0)
            spin.setSuffix(" дБ")
            spin.setDecimals(1)
        self.wf_lo_spin.setValue(-110.0)
        self.wf_hi_spin.setValue(-20.0)

        self.reset_view_btn = QtWidgets.QPushButton("Сброс вида графиков")

        form.addRow(self.auto_y_check)
        form.addRow(self.auto_x_check)
        form.addRow("Y мин", self.ymin_spin)
        form.addRow("Y макс", self.ymax_spin)
        form.addRow("Строк водопада", self.wf_rows_spin)
        form.addRow("Водопад ур. мин", self.wf_lo_spin)
        form.addRow("Водопад ур. макс", self.wf_hi_spin)
        form.addRow(self.reset_view_btn)
        self.display_group = box
        return box

    def _build_record_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Запись I/Q")
        form = QtWidgets.QFormLayout(box)

        self.record_format_combo = QtWidgets.QComboBox()
        for key, label in self.RECORD_FORMATS:
            self.record_format_combo.addItem(label, key)

        hint = QtWidgets.QLabel(
            "Пишет сырые I/Q в .iq и параметры в .json (для офлайн-разбора).\n"
            "Доступно в режимах «Реальное время» и «Мониторинг ГНСС»."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #8f8f8f;")

        form.addRow("Формат", self.record_format_combo)
        form.addRow(hint)
        self.record_group = box
        return box

    def _build_peaks_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        self.peaks_table = QtWidgets.QTableWidget(0, 5)
        self.peaks_table.setHorizontalHeaderLabels(
            ["Частота", "Уровень, дБ", "Выступ., дБ", "Канал", "Полоса"]
        )
        self.peaks_table.horizontalHeader().setStretchLastSection(True)
        self.peaks_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.peaks_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.peaks_table.verticalHeader().setVisible(False)
        layout.addWidget(self.peaks_table)
        return page

    def _build_gnss_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        self.gnss_status = QtWidgets.QLabel("—")
        self.gnss_status.setAlignment(QtCore.Qt.AlignCenter)
        self.gnss_status.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #d8d8d8; padding: 6px;"
        )
        layout.addWidget(self.gnss_status)

        hint = QtWidgets.QLabel(
            "Полезные сигналы ГНСС лежат ниже шумовой дорожки, поэтому любой "
            "заметный пик над шумом, рост шума или перегрузка АЦП — признак "
            "помехи/глушения. Выберите диапазон в категории «ГНСС (GNSS)», "
            "режим «Мониторинг ГНСС» и нажмите «Старт»."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #8f8f8f;")
        layout.addWidget(hint)

        info = QtWidgets.QFormLayout()
        self.gnss_labels: dict[str, QtWidgets.QLabel] = {}
        for key in (
            "Шумовой порог", "Мощность в полосе", "Макс. пик", "Частота пика",
            "Узкополосных помех", "Полоса помехи", "Δ шума", "Δ мощности",
        ):
            lbl = QtWidgets.QLabel("—")
            lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            self.gnss_labels[key] = lbl
            info.addRow(key + ":", lbl)
        info_host = QtWidgets.QWidget()
        info_host.setLayout(info)
        layout.addWidget(info_host)

        params = QtWidgets.QFormLayout()
        self.gnss_delta_spin = QtWidgets.QDoubleSpinBox()
        self.gnss_delta_spin.setRange(0.0, 40.0)
        self.gnss_delta_spin.setDecimals(1)
        self.gnss_delta_spin.setSuffix(" дБ")
        self.gnss_delta_spin.setValue(6.0)
        self.gnss_nb_spin = QtWidgets.QDoubleSpinBox()
        self.gnss_nb_spin.setRange(0.0, 40.0)
        self.gnss_nb_spin.setDecimals(1)
        self.gnss_nb_spin.setSuffix(" дБ")
        self.gnss_nb_spin.setValue(10.0)
        params.addRow("Рост шума = глушение", self.gnss_delta_spin)
        params.addRow("Помеха над шумом", self.gnss_nb_spin)
        params_host = QtWidgets.QWidget()
        params_host.setLayout(params)
        layout.addWidget(params_host)

        self.gnss_plot = pg.PlotWidget()
        self.gnss_plot.setLabel("left", "Уровень", units="дБ")
        self.gnss_plot.setLabel("bottom", "Отсчёт")
        self.gnss_plot.showGrid(x=True, y=True, alpha=0.3)
        for axis in ("bottom", "left"):
            try:
                self.gnss_plot.getAxis(axis).enableAutoSIPrefix(False)
            except Exception:
                pass
        self.gnss_plot.addLegend()
        self.gnss_noise_curve = self.gnss_plot.plot(
            pen=pg.mkPen("#7dd3fc", width=1), name="Шумовой порог"
        )
        self.gnss_power_curve = self.gnss_plot.plot(
            pen=pg.mkPen("#fbbf24", width=1), name="Мощность в полосе"
        )
        layout.addWidget(self.gnss_plot, 1)

        self.gnss_log = QtWidgets.QPlainTextEdit()
        self.gnss_log.setReadOnly(True)
        self.gnss_log.setMaximumBlockCount(500)
        layout.addWidget(self.gnss_log, 1)

        btns = QtWidgets.QHBoxLayout()
        self.gnss_reset_btn = QtWidgets.QPushButton("Обучить фон")
        self.gnss_clear_btn = QtWidgets.QPushButton("Очистить журнал")
        btns.addWidget(self.gnss_reset_btn)
        btns.addWidget(self.gnss_clear_btn)
        layout.addLayout(btns)
        return page

    def _build_plots(self) -> QtWidgets.QWidget:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setLabel("bottom", "Частота", units=self._freq_unit)
        self.spectrum_plot.setLabel("left", "Уровень", units="дБ")
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.3)
        self.spectrum_plot.setYRange(-120, 0, padding=0.0)
        self.spectrum_plot.setDownsampling(auto=True, mode="peak")
        self.spectrum_plot.setClipToView(True)
        self.spectrum_plot.enableAutoRange(axis="y", enable=False)
        for axis in ("bottom", "left"):
            try:
                # Отключаем авто-приставки СИ: иначе 1000 МГц подписывается
                # как «1 kМГц» (1000 МГц = 1 кМГц), и ось путает.
                self.spectrum_plot.getAxis(axis).enableAutoSIPrefix(False)
            except Exception:
                pass
        try:
            # Ручной зум/сдвиг мышью выключает авто-масштаб (см. обработчик).
            self.spectrum_plot.getViewBox().sigRangeChangedManually.connect(
                self._on_manual_view_change
            )
        except Exception:
            pass
        try:
            # Левая кнопка мыши — выделение области («рамка») для увеличения.
            self.spectrum_plot.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        except Exception:
            pass

        self.curve = self.spectrum_plot.plot(pen=pg.mkPen("#33ff88", width=1))
        self.peak_scatter = pg.ScatterPlotItem(
            size=9, pen=pg.mkPen("#ffaa00"), brush=pg.mkBrush(255, 170, 0, 160)
        )
        self.spectrum_plot.addItem(self.peak_scatter)
        self.threshold_line = pg.InfiniteLine(
            angle=0, movable=False, pen=pg.mkPen("#ff5555", style=QtCore.Qt.DashLine)
        )
        self.spectrum_plot.addItem(self.threshold_line)

        # Вертикальная линия на выбранном пике (для «К пику» и «К записи»).
        self.selected_line = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen("#ffd166", style=QtCore.Qt.DashLine, width=1),
        )
        self.selected_line.setVisible(False)
        self.spectrum_plot.addItem(self.selected_line, ignoreBounds=True)
        self.spectrum_plot.scene().sigMouseClicked.connect(self._on_plot_clicked)

        # Курсор-«прицел»: показывает частоту и уровень под мышью.
        self.readout_line = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen("#8899aa", style=QtCore.Qt.DotLine, width=1),
        )
        self.readout_line.setVisible(False)
        self.spectrum_plot.addItem(self.readout_line, ignoreBounds=True)
        self.readout_text = pg.TextItem(
            color="#e6edf5", anchor=(0.0, 1.0),
            fill=pg.mkBrush(10, 10, 18, 210), border=pg.mkPen("#8899aa"),
        )
        self.readout_text.setVisible(False)
        self.spectrum_plot.addItem(self.readout_text, ignoreBounds=True)
        self._mouse_proxy = pg.SignalProxy(
            self.spectrum_plot.scene().sigMouseMoved,
            rateLimit=30,
            slot=self._on_mouse_moved,
        )

        self.waterfall = WaterfallWidget(rows=300)
        splitter.addWidget(self.spectrum_plot)
        splitter.addWidget(self.waterfall)
        splitter.setSizes([540, 340])

        self.plots_splitter = splitter
        return splitter

    def _apply_view_limits(self) -> None:
        """Пределы масштабирования, чтобы график нельзя было «потерять»."""
        f = units.factor(self._freq_unit) or 1.0
        limits = dict(
            xMin=MIN_B210_FREQ / f,
            xMax=MAX_B210_FREQ / f,
            yMin=-200.0,
            yMax=50.0,
            minXRange=max(1e-9, 1.0 / f),
            minYRange=0.5,
        )
        for plot in (self.spectrum_plot,):
            try:
                plot.getViewBox().setLimits(**limits)
            except Exception:
                pass

    def _build_toolbar(self) -> None:
        toolbar = self.addToolBar("Основное")
        toolbar.setMovable(False)

        self.act_start = toolbar.addAction("▶ Старт", self._on_start)
        self.act_stop = toolbar.addAction("■ Стоп", self._on_stop)
        self.act_pause = toolbar.addAction("⏸ Пауза")
        self.act_pause.setCheckable(True)
        self.act_pause.setShortcut("Ctrl+P")
        self.act_pause.setToolTip(
            "Заморозить отображение, не прерывая приём и запись I/Q (Ctrl+P)"
        )
        self.act_pause.triggered.connect(self._on_toggle_pause)
        toolbar.addSeparator()
        self.act_zoom_in = toolbar.addAction("＋ Увеличить", lambda: self._zoom_view(0.5))
        self.act_zoom_in.setToolTip("Приблизить (можно также выделить рамку мышью)")
        self.act_zoom_out = toolbar.addAction("－ Уменьшить", lambda: self._zoom_view(2.0))
        self.act_zoom_out.setToolTip("Отдалить")
        self.act_zoom_peak = toolbar.addAction("К пику", self._zoom_to_peak)
        self.act_zoom_peak.setToolTip(
            "Показать участок вокруг выбранного пика (спектр и водопад)"
        )
        toolbar.addAction("Сброс вида", self._reset_view)
        toolbar.addSeparator()
        self.act_record = toolbar.addAction("● Запись I/Q")
        self.act_record.setCheckable(True)
        self.act_record.triggered.connect(self._on_toggle_record)
        toolbar.addAction("Открыть I/Q…", self._on_open_iq)
        toolbar.addSeparator()
        toolbar.addAction("Экспорт CSV", self._on_export_csv)
        toolbar.addAction("Скриншот", self._on_screenshot)
        toolbar.addAction("Очистить водопад", self.waterfall_clear)

    def _build_status_bar(self) -> None:
        self.pause_label = QtWidgets.QLabel("  ПАУЗА (приём продолжается)  ")
        self.pause_label.setStyleSheet(
            "color: #101014; background: #fbbf24; font-weight: bold; padding: 2px 6px;"
        )
        self.pause_label.setVisible(False)
        self.statusBar().addPermanentWidget(self.pause_label)

    # ==================================================================
    # Конфигурация
    # ==================================================================
    def _build_config(self) -> AcquisitionConfig:
        cell = lambda s: s.value()  # noqa: E731
        return AcquisitionConfig(
            mode=self.mode_combo.currentData(),
            device_args=self.device_args_edit.text().strip() or "type=b200",
            channel=int(cell(self.channel_spin)),
            antenna=self.antenna_combo.currentText().strip(),
            gain=cell(self.gain_spin),
            sample_rate=cell(self.rate_spin) * 1e6,
            start_freq=self._hz_of(self.start_spin),
            stop_freq=self._hz_of(self.stop_spin),
            sweep_step=self._hz_of(self.step_spin),
            settle_time=cell(self.settle_spin) / 1000.0,
            center_freq=self._hz_of(self.center_spin),
            fft_size=int(self.fft_combo.currentText()),
            overlap=cell(self.overlap_spin) / 100.0,
            averages=int(cell(self.averages_spin)),
            window=self.window_combo.currentText(),
            threshold_db=cell(self.threshold_spin),
            min_peak_distance=self._hz_of(self.distance_spin),
            min_prominence_db=cell(self.prominence_spin),
            max_peaks=int(cell(self.maxpeaks_spin)),
            waterfall_rows=int(cell(self.wf_rows_spin)),
            wf_level_low=cell(self.wf_lo_spin),
            wf_level_high=cell(self.wf_hi_spin),
            gnss_jam_delta_db=cell(self.gnss_delta_spin),
            gnss_nb_db=cell(self.gnss_nb_spin),
        )

    def _load_config(self, cfg: AcquisitionConfig) -> None:
        idx = self.mode_combo.findData(cfg.mode)
        self.mode_combo.setCurrentIndex(max(0, idx))
        self.device_args_edit.setText(cfg.device_args)
        self.channel_spin.setValue(cfg.channel)
        self.antenna_combo.setCurrentText(cfg.antenna)
        self.gain_spin.setValue(cfg.gain)
        self.rate_spin.setValue(cfg.sample_rate / 1e6)
        self._set_spin_hz(self.start_spin, cfg.start_freq)
        self._set_spin_hz(self.stop_spin, cfg.stop_freq)
        self._set_spin_hz(self.step_spin, cfg.sweep_step)
        self.settle_spin.setValue(int(cfg.settle_time * 1000))
        self._set_spin_hz(self.center_spin, cfg.center_freq)
        self.fft_combo.setCurrentText(str(cfg.fft_size))
        self.overlap_spin.setValue(int(cfg.overlap * 100))
        self.averages_spin.setValue(cfg.averages)
        self.window_combo.setCurrentText(cfg.window)
        self.threshold_spin.setValue(cfg.threshold_db)
        self._set_spin_hz(self.distance_spin, cfg.min_peak_distance)
        self.prominence_spin.setValue(cfg.min_prominence_db)
        self.maxpeaks_spin.setValue(cfg.max_peaks)
        self.wf_rows_spin.setValue(cfg.waterfall_rows)
        self.wf_lo_spin.setValue(cfg.wf_level_low)
        self.wf_hi_spin.setValue(cfg.wf_level_high)
        self.gnss_delta_spin.setValue(cfg.gnss_jam_delta_db)
        self.gnss_nb_spin.setValue(cfg.gnss_nb_db)
        self.threshold_line.setValue(cfg.threshold_db)
        self.waterfall.set_levels(cfg.wf_level_low, cfg.wf_level_high)

    def _connect_controls(self) -> None:
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.find_btn.clicked.connect(self._on_find)
        self.mode_combo.currentIndexChanged.connect(self._update_mode_enabled)
        self.unit_combo.currentTextChanged.connect(self._apply_freq_unit)

        self.threshold_spin.valueChanged.connect(self.threshold_line.setValue)
        self.threshold_spin.valueChanged.connect(self._refresh_peaks)
        self.ymin_spin.valueChanged.connect(self._refresh_y_range)
        self.ymax_spin.valueChanged.connect(self._refresh_y_range)
        self.auto_y_check.toggled.connect(self._on_auto_y_toggled)
        self.auto_x_check.toggled.connect(self._on_auto_x_toggled)
        for spin in (self.start_spin, self.stop_spin, self.center_spin):
            spin.valueChanged.connect(self._on_span_changed)
        self.rate_spin.valueChanged.connect(self._on_span_changed)
        self.mode_combo.currentIndexChanged.connect(self._on_span_changed)
        self.wf_lo_spin.valueChanged.connect(self._refresh_wf_levels)
        self.wf_hi_spin.valueChanged.connect(self._refresh_wf_levels)
        self.wf_rows_spin.valueChanged.connect(
            lambda v: self.waterfall.set_history(int(v))
        )
        self.reset_view_btn.clicked.connect(self._reset_view)
        self.peaks_table.itemSelectionChanged.connect(self._on_peak_selected)

        self.labels_check.toggled.connect(self._refresh_peaks)
        self.label_count_spin.valueChanged.connect(self._refresh_peaks)
        for chk in (self.lbl_level_check, self.lbl_prom_check,
                    self.lbl_bw_check, self.lbl_channel_check):
            chk.toggled.connect(self._refresh_peaks)

        self.gnss_delta_spin.valueChanged.connect(self._on_gnss_params)
        self.gnss_nb_spin.valueChanged.connect(self._on_gnss_params)
        self.gnss_reset_btn.clicked.connect(self._on_gnss_reset)
        self.gnss_clear_btn.clicked.connect(self.gnss_log.clear)

    # ==================================================================
    # Единицы измерения
    # ==================================================================
    def _apply_freq_unit(self, unit: str | None = None) -> None:
        new = unit or self.unit_combo.currentText()
        if new not in units.UNITS:
            new = units.DEFAULT
        old_f = units.factor(self._freq_unit)
        new_f = units.factor(new)
        values_hz = [spec["spin"].value() * old_f for spec in self._freq_spins]
        view_x = self.spectrum_plot.getViewBox().viewRange()[0]

        self._freq_unit = new
        dec = units.decimals(new)
        for spec, hz in zip(self._freq_spins, values_hz):
            spin = spec["spin"]
            spin.blockSignals(True)
            spin.setDecimals(dec)
            spin.setSuffix(" " + new)
            spin.setRange(spec["min_hz"] / new_f, spec["max_hz"] / new_f)
            spin.setValue(hz / new_f)
            spin.blockSignals(False)

        self.spectrum_plot.setLabel("bottom", "Частота", units=new)
        self.waterfall.set_unit(new, new_f)
        self._apply_view_limits()

        # Переносим текущий масштаб в новую единицу, чтобы вид не «прыгал».
        self._x_applied_hz = None
        self._y_range_applied = None
        if self.auto_x_check.isChecked():
            self._apply_display_span(force=True)
        else:
            self.spectrum_plot.setXRange(
                float(view_x[0]) * old_f / new_f,
                float(view_x[1]) * old_f / new_f,
                padding=0.0,
            )

        if self._last_psd.size:
            self._on_spectrum(self._last_freqs, self._last_psd)
        self._refresh_peaks()

    # ==================================================================
    # Управление запуском
    # ==================================================================
    def _set_running(self, running: bool) -> None:
        self._running = bool(running)
        self.start_btn.setEnabled(not running)
        self.act_start.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.act_stop.setEnabled(running)
        self.act_pause.setEnabled(running)
        if not running:
            self._set_paused(False)
        self.device_group.setEnabled(not running)
        self.processing_group.setEnabled(not running)
        self.sweep_group.setEnabled(not running)
        self.presets_group.setEnabled(not running)
        self.units_group.setEnabled(not running)
        self.record_group.setEnabled(not running)
        self._update_mode_enabled()
        self._update_record_action()

    def _update_mode_enabled(self) -> None:
        running = self._running
        mode = self.mode_combo.currentData()
        sweep = mode == "sweep"
        self.start_spin.setEnabled(sweep and not running)
        self.stop_spin.setEnabled(sweep and not running)
        self.step_spin.setEnabled(sweep and not running)
        self.settle_spin.setEnabled(sweep and not running)
        self.center_spin.setEnabled((not sweep) and not running)
        self._update_record_action()

    def _update_record_action(self) -> None:
        mode = self.mode_combo.currentData()
        usable = self._running and mode != "sweep" and self.offline is None
        self.act_record.setEnabled(self._recording or usable)
        if not usable and not self._recording and self.act_record.isChecked():
            self.act_record.blockSignals(True)
            self.act_record.setChecked(False)
            self.act_record.blockSignals(False)

    def _on_find(self) -> None:
        if not uhd_available():
            QtWidgets.QMessageBox.critical(
                self, "UHD не найден",
                "Модуль 'uhd' не установлен. См. README.md, раздел «Установка UHD».",
            )
            return
        try:
            found = list_devices(self.device_args_edit.text().strip())
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Ошибка поиска", str(exc))
            return
        if not found:
            QtWidgets.QMessageBox.information(
                self, "Устройства", "USRP не найдены. Проверьте USB и драйвер."
            )
            return
        text = "\n\n".join(
            "\n".join(f"{k}: {v}" for k, v in d.items()) for d in found
        )
        QtWidgets.QMessageBox.information(self, f"Найдено: {len(found)}", text)

    def _on_start(self) -> None:
        try:
            cfg = self._build_config()
            cfg.validate()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Проверьте параметры", str(exc))
            return
        if not uhd_available():
            QtWidgets.QMessageBox.critical(
                self, "UHD не найден",
                "Модуль 'uhd' не установлен, устройство недоступно.\n"
                "См. README.md, раздел «Установка UHD».",
            )
            return

        self._on_stop()
        self.waterfall.clear()
        self._x_applied_hz = None
        self._y_range_applied = None
        if self.auto_x_check.isChecked():
            span = self._display_span_hz()
            if span is not None:
                f = units.factor(self._freq_unit) or 1.0
                self.spectrum_plot.setXRange(
                    span[0] / f, span[1] / f, padding=0.01
                )
                self._x_applied_hz = span
                self.waterfall.set_auto_span(span[0], span[1])
        self._reset_tracks()
        self._last_peak_time = 0.0
        self._gnss_monitor.reset()
        self._gnss_hist_noise.clear()
        self._gnss_hist_power.clear()
        if cfg.mode == "gnss":
            self.tabs.setCurrentWidget(self.gnss_page)

        self.worker = AcquisitionWorker(cfg, self)
        self.worker.spectrum_ready.connect(self._on_spectrum)
        self.worker.row_ready.connect(self._on_row)
        self.worker.gnss_ready.connect(self._on_gnss)
        self.worker.device_info.connect(self._on_device_info)
        self.worker.progress.connect(self._on_progress)
        self.worker.recording.connect(self._on_recording_state)
        self.worker.status.connect(self.statusBar().showMessage)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(lambda: self._set_running(False))
        self.worker.start()
        self._set_running(True)

    def _on_stop(self) -> None:
        self._stop_recording()
        worker = self.worker
        self.worker = None
        if worker is not None:
            worker.stop()
            if worker.wait(5000):
                worker.deleteLater()
        offline_worker = self.offline
        self.offline = None
        if offline_worker is not None:
            offline_worker.stop()
            if offline_worker.wait(5000):
                offline_worker.deleteLater()
        self._set_running(False)

    # ==================================================================
    # Запись I/Q
    # ==================================================================
    def _on_toggle_record(self, checked: bool) -> None:
        if checked:
            if self.worker is None or not self._running:
                self._uncheck_record()
                QtWidgets.QMessageBox.information(
                    self, "Запись I/Q", "Сначала запустите измерение."
                )
                return
            if self.mode_combo.currentData() == "sweep":
                self._uncheck_record()
                QtWidgets.QMessageBox.information(
                    self, "Запись I/Q",
                    "Запись доступна в режимах «Реальное время» и «Мониторинг ГНСС».",
                )
                return
            default = f"iq_{datetime.now():%Y%m%d_%H%M%S}.iq"
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Запись I/Q", default, "I/Q (*.iq);;Все файлы (*)"
            )
            if not path:
                self._uncheck_record()
                return
            fmt = self.record_format_combo.currentData() or "complex64"
            self.worker.start_recording(path, fmt, self._record_metadata())
        else:
            self._stop_recording()

    def _uncheck_record(self) -> None:
        self.act_record.blockSignals(True)
        self.act_record.setChecked(False)
        self.act_record.blockSignals(False)

    def _stop_recording(self) -> None:
        if self.worker is not None:
            self.worker.stop_recording()
        self._recording = False
        self._update_record_action()

    def _on_recording_state(self, enabled: bool, path: str) -> None:
        self._recording = bool(enabled)
        if enabled:
            self.act_record.blockSignals(True)
            self.act_record.setChecked(True)
            self.act_record.blockSignals(False)
            self.statusBar().showMessage(f"Запись I/Q: {path}")
        else:
            self._uncheck_record()
            if path:
                self.statusBar().showMessage(f"Запись завершена: {Path(path).name}")
        self._update_record_action()

    def _record_metadata(self) -> dict:
        cfg = self._build_config()
        center = (
            0.5 * (cfg.start_freq + cfg.stop_freq)
            if cfg.mode == "sweep" else cfg.center_freq
        )
        return {
            "mode": cfg.mode,
            "sample_rate": cfg.sample_rate,
            "center_freq": center,
            "gain": cfg.gain,
            "antenna": cfg.antenna,
            "channel": cfg.channel,
            "device_args": cfg.device_args,
            "uhd": uhd_version(),
            "created": datetime.now().isoformat(timespec="seconds"),
        }

    # ==================================================================
    # Офлайн-разбор файла I/Q
    # ==================================================================
    def _on_open_iq(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Открыть I/Q", "", "I/Q (*.iq);;Все файлы (*)"
        )
        if not path:
            return
        meta = offline.read_metadata(path)
        if meta and meta.get("sample_rate") and meta.get("center_freq"):
            rate = float(meta["sample_rate"])
            center = float(meta["center_freq"])
        else:
            rate_mhz, ok = QtWidgets.QInputDialog.getDouble(
                self, "Частота дискретизации", "Частота дискретизации, МГц:",
                4.0, 0.05, 61.44, 3,
            )
            if not ok:
                return
            center_mhz, ok = QtWidgets.QInputDialog.getDouble(
                self, "Центральная частота", "Центральная частота, МГц:",
                1575.42, 70.0, 6000.0, 3,
            )
            if not ok:
                return
            rate, center = rate_mhz * 1e6, center_mhz * 1e6

        self._on_stop()
        self.waterfall.clear()
        self._reset_tracks()
        self._gnss_monitor.reset()
        self._gnss_hist_noise.clear()
        self._gnss_hist_power.clear()

        # Показываем в настройках реальные параметры файла, чтобы ось X
        # соответствовала записи.
        self._set_spin_hz(self.center_spin, center)
        rate_mhz = rate / 1e6
        if self.rate_spin.minimum() <= rate_mhz <= self.rate_spin.maximum():
            self.rate_spin.setValue(rate_mhz)
        self._x_applied_hz = None
        self._on_span_changed()

        cfg = self._build_config()
        gnss = cfg.mode == "gnss"
        if gnss:
            self.tabs.setCurrentWidget(self.gnss_page)
        self.offline = OfflineWorker(path, cfg, rate, center, meta, gnss, self)
        self.offline.spectrum_ready.connect(self._on_spectrum)
        self.offline.row_ready.connect(self._on_row)
        self.offline.gnss_ready.connect(self._on_gnss)
        self.offline.progress.connect(self._on_progress)
        self.offline.status.connect(self.statusBar().showMessage)
        self.offline.error.connect(self._on_error)
        self.offline.finished.connect(lambda: self._on_offline_finished())
        self.offline.start()
        self._set_running(True)
        self.statusBar().showMessage(f"Разбор файла: {Path(path).name}")

    def _on_offline_finished(self) -> None:
        self.offline = None
        self._set_running(False)

    # ==================================================================
    # Слоты данных
    # ==================================================================
    def _on_device_info(self, info: dict) -> None:
        parts = [f"{k}: {v}" for k, v in info.items()]
        self.statusBar().showMessage(" | ".join(parts))

    def _on_progress(self, done: int, total: int) -> None:
        self.statusBar().showMessage(f"Обработано: {done}/{total} точек")

    def _on_error(self, text: str) -> None:
        self.statusBar().showMessage(text)
        QtWidgets.QMessageBox.critical(self, "Ошибка", text)

    # ------------------------------------------------------------------
    # Диапазон оси X и курсор
    # ------------------------------------------------------------------
    def _display_span_hz(self) -> tuple[float, float] | None:
        """Настроенный диапазон отображения (без учёта принятых данных)."""
        if self.mode_combo.currentData() == "sweep":
            lo, hi = self._hz_of(self.start_spin), self._hz_of(self.stop_spin)
        else:
            center = self._hz_of(self.center_spin)
            half = self.rate_spin.value() * 1e6 / 2.0
            lo, hi = center - half, center + half
        lo = max(lo, MIN_B210_FREQ)
        hi = min(hi, MAX_B210_FREQ)
        if hi <= lo:
            return None
        return lo, hi

    def _display_target_hz(self) -> tuple[float, float] | None:
        """Что показывать по X: настроенный диапазон, иначе — по данным."""
        data_span = None
        if self._last_freqs.size > 1:
            data_span = (float(self._last_freqs[0]), float(self._last_freqs[-1]))
        span = self._display_span_hz()
        if span is not None and data_span is not None:
            if span[0] <= data_span[0] and data_span[1] <= span[1]:
                return span
            return data_span
        return span if span is not None else data_span

    def _apply_display_span(self, force: bool = False) -> None:
        """Стабилизировать масштаб по X.

        Ось привязана к настроенному диапазону (Начало..Конец или центр ±
        полоса), а не к «растущим» данным свипа, поэтому она не проматывается.
        Повторно ``setXRange`` вызывается только при заметном изменении.
        """
        if self._paused:
            return
        target = self._display_target_hz()
        if target is None:
            return
        if self.auto_x_check.isChecked():
            prev = self._x_applied_hz
            width = max(target[1] - target[0], 1.0)
            stale = (
                prev is None
                or abs(prev[0] - target[0]) > 0.005 * width
                or abs(prev[1] - target[1]) > 0.005 * width
            )
            if force or stale:
                f = units.factor(self._freq_unit) or 1.0
                self.spectrum_plot.setXRange(
                    target[0] / f, target[1] / f, padding=0.01
                )
                self._x_applied_hz = target
        self.waterfall.set_auto_span(target[0], target[1])

    def _on_span_changed(self, *_args) -> None:
        """Изменились настройки частот — обновляем диапазон оси."""
        self._apply_display_span(force=True)

    def _on_manual_view_change(self, *_args) -> None:
        """Пользователь сам подвинул/приблизил график мышью."""
        changed = False
        for check in (self.auto_x_check, self.auto_y_check):
            if check.isChecked():
                check.blockSignals(True)
                check.setChecked(False)
                check.blockSignals(False)
                changed = True
        self._x_applied_hz = None
        self._y_range_applied = None
        if changed:
            self.statusBar().showMessage(
                "Ручной масштаб: автомасштаб выключен, "
                "кнопка «Сброс вида» вернёт его."
            )

    def _on_auto_y_toggled(self, checked: bool) -> None:
        self._y_range_applied = None
        self._refresh_y_range()

    def _on_mouse_moved(self, evt) -> None:
        """Показать частоту и уровень под курсором."""
        if self._last_freqs.size == 0 or self._last_psd.size != self._last_freqs.size:
            return
        pos = evt[0]
        if not self.spectrum_plot.sceneBoundingRect().contains(pos):
            self.readout_line.setVisible(False)
            self.readout_text.setVisible(False)
            return
        point = self.spectrum_plot.getViewBox().mapSceneToView(pos)
        f = units.factor(self._freq_unit) or 1.0
        hz = float(point.x()) * f
        idx = int(np.searchsorted(self._last_freqs, hz))
        idx = min(max(idx, 0), self._last_freqs.size - 1)
        if idx > 0 and abs(self._last_freqs[idx - 1] - hz) < abs(
            self._last_freqs[idx] - hz
        ):
            idx -= 1
        freq = float(self._last_freqs[idx])
        level = float(self._last_psd[idx])
        self.readout_line.setPos(freq / f)
        self.readout_line.setVisible(True)
        self.readout_text.setText(f"{self._fmt_freq(freq)}\n{level:.1f} дБ")
        self.readout_text.setPos(freq / f, level)
        self.readout_text.setVisible(True)

    def _on_spectrum(self, freqs, psd) -> None:
        freqs = np.asarray(freqs, dtype=np.float64)
        psd = np.asarray(psd, dtype=np.float64)
        if psd.size == 0:
            return
        self._last_freqs = freqs
        self._last_psd = psd
        if self._paused:
            # Пауза: приём продолжается, но экран не обновляем.
            return
        f = units.factor(self._freq_unit) or 1.0
        self.curve.setData(freqs / f, psd)
        self._apply_display_span()
        self._refresh_y_range()

        now = time.monotonic()
        if now - self._last_peak_time >= 0.12:
            self._last_peak_time = now
            self._refresh_peaks()

    def _on_row(self, freqs, psd) -> None:
        if self._paused:
            return
        self.waterfall.add_row(np.asarray(freqs, dtype=np.float64),
                              np.asarray(psd, dtype=np.float64))

    def _refresh_y_range(self) -> None:
        if self._last_psd.size == 0:
            return
        if not self.auto_y_check.isChecked():
            lo = float(self.ymin_spin.value())
            hi = float(self.ymax_spin.value())
            self.spectrum_plot.setYRange(lo, hi, padding=0.0)
            self._y_range_applied = (lo, hi)
            return

        data_lo = float(np.percentile(self._last_psd, 5))
        data_hi = float(np.max(self._last_psd))
        lo = data_lo - 5.0
        hi = data_hi + 5.0
        prev = self._y_range_applied
        if prev is not None:
            plo, phi = prev
            # Держим прежнюю «рамку», пока данные в неё вписываются и она не
            # слишком просторная: так график не дёргается каждый кадр.
            if (
                plo <= data_lo
                and data_hi <= phi
                and (phi - plo) <= 3.0 * max(10.0, data_hi - data_lo)
            ):
                lo, hi = plo, phi
        lo = max(lo, -200.0)
        hi = min(hi, 50.0)
        if hi - lo < 20.0:
            hi = lo + 20.0
        if prev is not None and abs(prev[0] - lo) < 1e-6 and abs(prev[1] - hi) < 1e-6:
            return
        self.spectrum_plot.setYRange(lo, hi, padding=0.0)
        self._y_range_applied = (lo, hi)

    def _on_auto_x_toggled(self, checked: bool) -> None:
        if checked:
            self._reset_view()

    def _reset_view(self) -> None:
        self._x_applied_hz = None
        self._y_range_applied = None
        for check in (self.auto_x_check, self.auto_y_check):
            check.blockSignals(True)
            check.setChecked(True)
            check.blockSignals(False)
        self.waterfall.reset_view()
        self._apply_display_span(force=True)
        self._refresh_y_range()

    def _refresh_wf_levels(self) -> None:
        self.waterfall.set_levels(self.wf_lo_spin.value(), self.wf_hi_spin.value())

    # ------------------------------------------------------------------
    # Стабильные метки пиков (слежение вместо «дёргания»)
    # ------------------------------------------------------------------
    def _reset_tracks(self) -> None:
        self._tracks = []
        self._peak_scatter_clear()
        for item in self._peak_labels:
            item.setVisible(False)
        self.peaks_table.setRowCount(0)
        self._table_items = []

    def _peak_scatter_clear(self) -> None:
        self.peak_scatter.setData([], [])

    def _current_peaks(self) -> list[dict]:
        if self._last_psd.size == 0:
            return []
        return dsp.find_peaks(
            self._last_freqs,
            self._last_psd,
            threshold_db=self.threshold_spin.value(),
            min_peak_distance=self._hz_of(self.distance_spin),
            min_prominence_db=self.prominence_spin.value(),
            max_peaks=int(self.maxpeaks_spin.value()),
        )

    def _match_tolerance(self) -> float:
        bin_w = 1.0
        if self._last_freqs.size > 1:
            bin_w = float(np.median(np.abs(np.diff(self._last_freqs)))) or 1.0
        dist = self._hz_of(self.distance_spin)
        return max(3.0 * bin_w, 0.3 * dist) if dist > 0 else 3.0 * bin_w

    def _update_tracks(self, peaks: list[dict]) -> list[dict]:
        old = self._tracks
        matched = [False] * len(old)
        tol = self._match_tolerance()
        fresh: list[dict] = []
        alpha_f, alpha_p = 0.4, 0.35

        for p in sorted(peaks, key=lambda item: item["power"], reverse=True):
            best, best_d = -1, None
            for i, t in enumerate(old):
                if matched[i]:
                    continue
                d = abs(p["freq"] - t["freq"])
                if d <= tol and (best_d is None or d < best_d):
                    best, best_d = i, d
            if best < 0:
                fresh.append({
                    "id": self._track_next_id,
                    "freq": float(p["freq"]),
                    "power": float(p["power"]),
                    "prominence": float(p["prominence"]),
                    "bandwidth": float(p.get("bandwidth", 0.0)),
                    "age": 1,
                    "misses": 0,
                })
                self._track_next_id += 1
            else:
                t = old[best]
                matched[best] = True
                t["freq"] = (1 - alpha_f) * t["freq"] + alpha_f * p["freq"]
                t["power"] = (1 - alpha_p) * t["power"] + alpha_p * p["power"]
                t["prominence"] = (1 - alpha_p) * t["prominence"] + alpha_p * p["prominence"]
                t["bandwidth"] = (1 - alpha_p) * t["bandwidth"] + alpha_p * p.get("bandwidth", 0.0)
                t["age"] += 1
                t["misses"] = 0

        for i, t in enumerate(old):
            if not matched[i]:
                t["misses"] += 1

        self._tracks = [t for t in old if t["misses"] <= 8] + fresh
        return sorted(self._tracks, key=lambda t: t["power"], reverse=True)

    def _refresh_peaks(self) -> None:
        tracks = self._update_tracks(self._current_peaks())
        f = units.factor(self._freq_unit) or 1.0
        if tracks:
            self.peak_scatter.setData(
                [t["freq"] / f for t in tracks], [t["power"] for t in tracks]
            )
        else:
            self._peak_scatter_clear()
        self._update_peak_labels(tracks)
        self._update_peaks_table(tracks)

    def _label_text(self, t: dict) -> str:
        lines = [self._fmt_freq(t["freq"])]
        parts: list[str] = []
        if self.lbl_level_check.isChecked():
            parts.append(f"{t['power']:.1f} дБ")
        if self.lbl_prom_check.isChecked():
            parts.append(f"выст. {t['prominence']:.1f} дБ")
        if self.lbl_bw_check.isChecked() and t["bandwidth"] > 0:
            parts.append(f"≈{units.format_span(t['bandwidth'])}")
        if parts:
            lines.append("  ".join(parts))
        if self.lbl_channel_check.isChecked():
            info = annotate.describe(t["freq"], t["bandwidth"])
            if info is not None:
                lines.append(info.label)
        return "\n".join(lines)

    def _update_peak_labels(self, tracks: list[dict]) -> None:
        count = int(self.label_count_spin.value())
        if not self.labels_check.isChecked() or count <= 0:
            for item in self._peak_labels:
                item.setVisible(False)
            return

        stable = [t for t in tracks if t["age"] >= 1]
        chosen = stable[:count]
        if len(chosen) < count and len(tracks) > len(chosen):
            chosen = tracks[:count]

        f = units.factor(self._freq_unit) or 1.0
        span = 0.0
        if self._last_freqs.size > 1:
            span = float(self._last_freqs[-1] - self._last_freqs[0])
        gap_thr = (span / f) * 0.07 if span > 0 else 0.0
        levels = [2.0, 9.0, 16.0, 23.0]

        ordered = sorted(chosen, key=lambda t: t["freq"])
        placed: list[tuple[dict, float]] = []
        prev_x: float | None = None
        level = 0
        for t in ordered:
            x = t["freq"] / f
            if prev_x is not None and gap_thr > 0 and abs(x - prev_x) < gap_thr:
                level = min(level + 1, len(levels) - 1)
            else:
                level = 0
            prev_x = x
            placed.append((t, levels[level]))

        while len(self._peak_labels) < len(placed):
            item = pg.TextItem(
                anchor=(0.5, 1.0),
                color="#ffd166",
                fill=pg.mkBrush(10, 10, 18, 220),
                border=pg.mkPen("#ffd166"),
            )
            self.spectrum_plot.addItem(item)
            self._peak_labels.append(item)

        _, (_, y_hi) = self.spectrum_plot.viewRange()
        for i, item in enumerate(self._peak_labels):
            if i < len(placed):
                t, off = placed[i]
                item.setText(self._label_text(t))
                y = min(t["power"] + off, y_hi - 1.0)
                item.setPos(t["freq"] / f, y)
                item.setVisible(True)
            else:
                item.setVisible(False)

    def _update_peaks_table(self, tracks: list[dict]) -> None:
        rows = sorted(tracks, key=lambda t: t["freq"])[: int(self.maxpeaks_spin.value())]
        table = self.peaks_table
        self._table_rows = rows
        if table.rowCount() != len(rows):
            table.setRowCount(len(rows))
        if len(self._table_items) > len(rows):
            self._table_items = self._table_items[: len(rows)]
        while len(self._table_items) < len(rows):
            row_items = [QtWidgets.QTableWidgetItem("") for _ in range(table.columnCount())]
            for col, item in enumerate(row_items):
                item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                table.setItem(len(self._table_items), col, item)
            self._table_items.append(row_items)

        for r, t in enumerate(rows):
            info = annotate.describe(t["freq"], t["bandwidth"])
            values = [
                self._fmt_freq(t["freq"]),
                f"{t['power']:.1f}",
                f"{t['prominence']:.1f}",
                info.label if info is not None else "",
                units.format_span(t["bandwidth"]) if t["bandwidth"] > 0 else "",
            ]
            for col, item in enumerate(self._table_items[r]):
                item.setText(values[col])

        self._restore_table_selection()

    # ------------------------------------------------------------------
    # Выбор пика и масштаб
    # ------------------------------------------------------------------
    def _selected_or_strongest_peak(self) -> dict | None:
        if self._selected_peak is not None:
            return self._selected_peak
        if self._tracks:
            return max(self._tracks, key=lambda t: t["power"])
        return None

    def _nearest_track(self, freq_hz: float) -> dict | None:
        if not self._tracks:
            return None
        tol = max(self._match_tolerance() * 3.0, 0.05 * self._data_span_hz())
        best, best_d = None, None
        for t in self._tracks:
            d = abs(t["freq"] - freq_hz)
            if best_d is None or d < best_d:
                best, best_d = t, d
        if best is None or best_d is None or best_d > tol:
            return None
        return best

    def _data_span_hz(self) -> float:
        if self._last_freqs.size > 1:
            return float(self._last_freqs[-1] - self._last_freqs[0])
        return MAX_B210_FREQ - MIN_B210_FREQ

    def _update_selected_line(self) -> None:
        if self._selected_peak is None:
            self.selected_line.setVisible(False)
            return
        f = units.factor(self._freq_unit) or 1.0
        self.selected_line.setValue(float(self._selected_peak["freq"]) / f)
        self.selected_line.setVisible(True)

    def _select_table_row(self, peak: dict) -> None:
        """Подсветить строку таблицы, соответствующую выбранному пику."""
        try:
            r = self._table_rows.index(peak)
        except ValueError:
            return
        table = self.peaks_table
        if r >= table.rowCount():
            return
        table.blockSignals(True)
        table.selectRow(r)
        table.blockSignals(False)

    def _restore_table_selection(self) -> None:
        if self._selected_peak is None:
            return
        try:
            r = self._table_rows.index(self._selected_peak)
        except ValueError:
            return
        table = self.peaks_table
        if r >= table.rowCount():
            return
        table.blockSignals(True)
        table.selectRow(r)
        table.blockSignals(False)

    def _on_peak_selected(self) -> None:
        rows = self.peaks_table.selectionModel().selectedRows()
        if not rows:
            return
        r = rows[0].row()
        if 0 <= r < len(self._table_rows):
            self._selected_peak = self._table_rows[r]
            self._update_selected_line()

    def _on_plot_clicked(self, event) -> None:
        if event.button() != QtCore.Qt.LeftButton:
            return
        if not self.spectrum_plot.sceneBoundingRect().contains(event.scenePos()):
            return
        point = self.spectrum_plot.getViewBox().mapSceneToView(event.scenePos())
        f = units.factor(self._freq_unit) or 1.0
        peak = self._nearest_track(float(point.x()) * f)
        if peak is None:
            return
        self._selected_peak = peak
        self._update_selected_line()
        self._select_table_row(peak)

    def _data_span_view(self) -> tuple[float, float]:
        f = units.factor(self._freq_unit) or 1.0
        target = self._display_target_hz()
        if target is not None:
            return target[0] / f, target[1] / f
        return MIN_B210_FREQ / f, MAX_B210_FREQ / f

    def _set_manual_x(self, lo: float, hi: float) -> None:
        """Задать диапазон X вручную и снять галочку «Авто-масштаб по X»."""
        self.auto_x_check.blockSignals(True)
        self.auto_x_check.setChecked(False)
        self.auto_x_check.blockSignals(False)
        self.spectrum_plot.setXRange(lo, hi, padding=0.0)

    def _zoom_view(self, factor: float) -> None:
        """Увеличить (factor<1) или уменьшить (factor>1) вид графика."""
        f = units.factor(self._freq_unit) or 1.0
        (x0, x1), (y0, y1) = self.spectrum_plot.getViewBox().viewRange()
        d0, d1 = self._data_span_view()
        span = (x1 - x0) * factor
        span = min(span, d1 - d0)
        span = max(span, 1.0 / f)
        center = 0.5 * (x0 + x1)
        lo, hi = center - span / 2.0, center + span / 2.0
        if lo < d0:
            lo, hi = d0, d0 + span
        if hi > d1:
            hi, lo = d1, d1 - span
        self._set_manual_x(lo, hi)
        self._x_applied_hz = None
        self.waterfall.set_x_span(lo * f, hi * f)
        if not self.auto_y_check.isChecked():
            yc = 0.5 * (y0 + y1)
            self.spectrum_plot.setYRange(
                yc - (yc - y0) * factor, yc + (y1 - yc) * factor, padding=0.0
            )

    def _zoom_to_peak(self) -> None:
        """Показать участок вокруг выбранного (или самого мощного) пика."""
        peak = self._selected_or_strongest_peak()
        if peak is None:
            QtWidgets.QMessageBox.information(
                self, "К пику",
                "Пока нет пиков. Запустите измерение или уменьшите порог.",
            )
            return
        f = units.factor(self._freq_unit) or 1.0
        bin_w = 1.0
        if self._last_freqs.size > 1:
            bin_w = float(np.median(np.abs(np.diff(self._last_freqs)))) or 1.0
        half = max(1.0 * float(peak.get("bandwidth", 0.0)), 10.0 * bin_w)
        center = float(peak["freq"])
        lo_hz, hi_hz = center - half, center + half
        d0, d1 = self._data_span_view()
        lo, hi = lo_hz / f, hi_hz / f
        if lo < d0:
            lo, hi = d0, d0 + (hi - lo)
        if hi > d1:
            hi, lo = d1, d1 - (hi - lo)
        self._set_manual_x(lo, hi)

        self.auto_y_check.blockSignals(True)
        self.auto_y_check.setChecked(False)
        self.auto_y_check.blockSignals(False)
        y_hi = float(peak["power"]) + 8.0
        self.spectrum_plot.setYRange(y_hi - 45.0, y_hi, padding=0.0)

        self.waterfall.set_x_span(
            max(lo_hz, MIN_B210_FREQ), min(hi_hz, MAX_B210_FREQ)
        )
        self.statusBar().showMessage(
            f"Показан пик {self._fmt_freq(center)} "
            f"(полоса {units.format_span(hi_hz - lo_hz)})"
        )

    # ------------------------------------------------------------------
    # Пауза
    # ------------------------------------------------------------------
    def _set_paused(self, value: bool) -> None:
        value = bool(value)
        self._paused = value
        self.act_pause.blockSignals(True)
        self.act_pause.setChecked(value)
        self.act_pause.blockSignals(False)
        self.act_pause.setText("▶ Продолжить" if value else "⏸ Пауза")
        self.pause_label.setVisible(value)

    def _on_toggle_pause(self, checked: bool) -> None:
        self._set_paused(checked)
        if self._paused:
            self.statusBar().showMessage(
                "Пауза: отображение заморожено, приём и запись I/Q продолжаются"
            )
            return
        self.statusBar().showMessage("Возобновлено")
        if self._last_psd.size:
            self._on_spectrum(self._last_freqs, self._last_psd)
        self._refresh_peaks()

    # ------------------------------------------------------------------
    # ГНСС
    # ------------------------------------------------------------------
    def _on_gnss_params(self, *_args) -> None:
        self._gnss_monitor.jam_delta_db = self.gnss_delta_spin.value()
        self._gnss_monitor.nb_prominence_db = self.gnss_nb_spin.value()

    def _on_gnss_reset(self) -> None:
        self._gnss_monitor.reset()
        self._gnss_hist_noise.clear()
        self._gnss_hist_power.clear()
        self.gnss_log.appendPlainText(
            f"[{datetime.now():%H:%M:%S}] Фон сброшен, идёт повторное обучение"
        )

    def _on_gnss(self, analysis) -> None:
        a = analysis
        text = self._log_gnss_event(a)
        if self._paused:
            return
        color = STATUS_COLOR.get(a.status, "#d8d8d8")
        self.gnss_status.setText(text)
        self.gnss_status.setStyleSheet(
            f"font-size: 16px; font-weight: bold; color: {color}; padding: 6px;"
        )
        labels = self.gnss_labels
        labels["Шумовой порог"].setText(f"{a.noise_floor_db:.1f} дБ")
        labels["Мощность в полосе"].setText(f"{a.band_power_db:.1f} дБ")
        labels["Макс. пик"].setText(f"{a.peak_db:.1f} дБ")
        labels["Частота пика"].setText(self._fmt_freq(a.peak_freq))
        labels["Узкополосных помех"].setText(str(a.n_narrowband))
        labels["Полоса помехи"].setText(
            units.format_span(a.occupied_bw) if a.occupied_bw > 0 else "—"
        )
        labels["Δ шума"].setText(f"{a.delta_noise_db:+.1f} дБ")
        labels["Δ мощности"].setText(f"{a.delta_power_db:+.1f} дБ")

        self._gnss_hist_noise.append(a.noise_floor_db)
        self._gnss_hist_power.append(a.band_power_db)
        limit = 400
        if len(self._gnss_hist_noise) > limit:
            del self._gnss_hist_noise[:-limit]
            del self._gnss_hist_power[:-limit]
        x = np.arange(len(self._gnss_hist_noise))
        self.gnss_noise_curve.setData(x, self._gnss_hist_noise)
        self.gnss_power_curve.setData(x, self._gnss_hist_power)

    def _log_gnss_event(self, a) -> str:
        """Журнал событий ГНСС (ведётся и на паузе) + текст статуса."""
        text = STATUS_TEXT.get(a.status, "—")
        now = time.monotonic()
        changed = a.status != self._gnss_last_status
        if changed or (a.status != "normal" and now - self._gnss_last_log >= 2.0):
            self._gnss_last_log = now
            self._gnss_last_status = a.status
            stamp = f"[{datetime.now():%H:%M:%S}] {text}: "
            self.gnss_log.appendPlainText(stamp + "; ".join(a.reasons))
        return text

    # ==================================================================
    # Экспорт
    # ==================================================================
    def waterfall_clear(self) -> None:
        self.waterfall.clear()

    def _on_export_csv(self) -> None:
        if self._last_psd.size == 0:
            QtWidgets.QMessageBox.information(self, "Экспорт", "Нет данных для экспорта.")
            return
        default = f"spectrum_{datetime.now():%Y%m%d_%H%M%S}.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Сохранить спектр", default, "CSV (*.csv)"
        )
        if not path:
            return
        tracks = sorted(self._tracks, key=lambda t: t["freq"])
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                fh.write("# SDR Scan export\n")
                fh.write(f"# generated: {datetime.now().isoformat(timespec='seconds')}\n")
                fh.write(f"# mode: {self.mode_combo.currentData()}\n")
                fh.write(f"# uhd: {uhd_version()}\n")
                fh.write("freq_hz,freq_mhz,level_db\n")
                writer = csv.writer(fh)
                for f, p in zip(self._last_freqs, self._last_psd):
                    writer.writerow([f"{f:.3f}", f"{f/1e6:.6f}", f"{p:.3f}"])
                if tracks:
                    fh.write("# peaks\n")
                    fh.write(
                        "freq_hz,freq_mhz,level_db,prominence_db,bandwidth_hz,channel\n"
                    )
                    for t in tracks:
                        info = annotate.describe(t["freq"], t["bandwidth"])
                        writer.writerow([
                            f"{t['freq']:.3f}",
                            f"{t['freq']/1e6:.6f}",
                            f"{t['power']:.3f}",
                            f"{t['prominence']:.3f}",
                            f"{t['bandwidth']:.1f}",
                            info.label if info is not None else "",
                        ])
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Ошибка записи", str(exc))
            return
        self.statusBar().showMessage(f"Сохранено: {Path(path).name}")

    def _on_screenshot(self) -> None:
        default = f"spectrum_{datetime.now():%Y%m%d_%H%M%S}.png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Сохранить скриншот", default, "PNG (*.png)"
        )
        if not path:
            return
        pixmap = self.plots_splitter.grab()
        if not pixmap.save(path, "PNG"):
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Не удалось сохранить PNG.")
            return
        self.statusBar().showMessage(f"Скриншот: {Path(path).name}")

    # ==================================================================
    def closeEvent(self, event) -> None:  # noqa: N802
        self._on_stop()
        event.accept()
