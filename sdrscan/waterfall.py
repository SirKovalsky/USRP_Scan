"""Виджет водопада (spectrogram) на pyqtgraph."""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg


class WaterfallWidget(QtWidgets.QWidget):
    """Прокручиваемый спектр во времени.

    Новые строки добавляются сверху. ``x`` - частота в выбранной единице,
    ``y`` - время. На вход подаются частоты в герцах.
    """

    def __init__(
        self,
        rows: int = 300,
        levels: tuple[float, float] = (-110.0, -20.0),
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.rows = int(rows)
        self._levels = (float(levels[0]), float(levels[1]))
        self._data: np.ndarray | None = None
        self._hz0 = 0.0
        self._hz1 = 1.0
        self.unit_name = "МГц"
        self.unit_factor = 1e6
        self._view_locked = False
        # Диапазон, заданный приложением (настройками), в герцах.
        self._auto_span_hz: tuple[float, float] | None = None
        # Ручной зум (кнопка «К пику» или мышь), тоже в герцах.
        self._locked_span_hz: tuple[float, float] | None = None
        # Что реально применено к setXRange: чтобы не дёргать масштаб зря.
        self._applied_hz: tuple[float, float] | None = None

        self.glw = pg.GraphicsLayoutWidget()
        self.plot = self.glw.addPlot(row=0, col=0)
        self.plot.setLabel("bottom", "Частота", units=self.unit_name)
        self.plot.setLabel("left", "Время")
        self.plot.invertY(True)          # строка 0 - сверху (свежие данные)
        self.plot.showGrid(x=True, y=False, alpha=0.2)
        self.plot.getAxis("left").setStyle(showValues=False)
        for axis in ("bottom", "left"):
            try:
                # Отключаем авто-приставки СИ: иначе pyqtgraph подписывает
                # 1000 МГц как «1 kМГц», что путает.
                self.plot.getAxis(axis).enableAutoSIPrefix(False)
            except Exception:
                pass
        try:
            # Ручной сдвиг/зум мышью фиксирует вид, пока не нажат «Сброс вида».
            self.plot.getViewBox().sigRangeChangedManually.connect(
                self._on_manual_range
            )
        except Exception:
            pass

        self.image = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self.image)

        self._apply_colormap()
        self.image.setLevels(self._levels)

        try:
            cmap = pg.colormap.get("viridis")
            self.colorbar = pg.ColorBarItem(values=self._levels, colorMap=cmap, label="дБ")
            self.colorbar.setImageItem(self.image)
            self.glw.addItem(self.colorbar, row=0, col=1)
        except Exception:
            self.colorbar = None

        self._apply_limits()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.glw)

    # ------------------------------------------------------------------
    def _apply_colormap(self) -> None:
        try:
            self.image.setColorMap(pg.colormap.get("viridis"))
        except Exception:
            try:
                self.image.setLookupTable(pg.colormap.get("viridis").getLookupTable())
            except Exception:
                pass

    def _apply_limits(self) -> None:
        """Не даём «уехать» в пустоту: ограничиваем область приёма B210."""
        try:
            f = self.unit_factor
            vb = self.plot.getViewBox()
            vb.setLimits(
                xMin=70e6 / f,
                xMax=6e9 / f,
                yMin=0.0,
                yMax=float(self.rows),
                minXRange=max(1e-9, 1.0 / f),
                minYRange=1.0,
            )
        except Exception:
            pass

    def set_unit(self, name: str, factor: float) -> None:
        """Сменить единицу частоты по оси X."""
        self.unit_name = name
        self.unit_factor = float(factor) or 1.0
        try:
            self.plot.setLabel("bottom", "Частота", units=self.unit_name)
        except Exception:
            pass
        self._apply_limits()
        self._sync_x(force=True)

    def set_levels(self, low: float, high: float) -> None:
        self._levels = (float(low), float(high))
        self.image.setLevels(self._levels)
        if self.colorbar is not None:
            try:
                self.colorbar.setLevels(self._levels)
            except Exception:
                pass

    def set_history(self, rows: int) -> None:
        self.rows = max(1, int(rows))
        self._data = None  # пересоздастся при следующей строке
        self._apply_limits()

    def clear(self) -> None:
        self._data = None
        self.image.clear()
        self._hz0, self._hz1 = 0.0, 1.0
        self._view_locked = False
        self._auto_span_hz = None
        self._locked_span_hz = None
        self._applied_hz = None
        self.plot.setXRange(0.0, 1.0, padding=0.0)

    def reset_view(self) -> None:
        """Вернуть исходный масштаб по частоте (снять ручной зум)."""
        self._view_locked = False
        self._locked_span_hz = None
        self._sync_x(force=True)

    @property
    def view_locked(self) -> bool:
        """True, если по частоте включён ручной зум."""
        return self._view_locked

    def set_x_span(self, lo_hz: float, hi_hz: float) -> None:
        """Показать только участок ``[lo_hz, hi_hz]`` (зум «к пику»).

        Пока ручной зум включён, новые строки водопада не меняют масштаб.
        Снимается кнопкой «Сброс вида».
        """
        span = self._clamp_span(lo_hz, hi_hz)
        if span is None:
            return
        self._locked_span_hz = span
        self._view_locked = True
        self._apply_limits()
        self._sync_x(force=True)

    def set_auto_span(self, lo_hz: float, hi_hz: float) -> None:
        """Задать участок, который водопад показывает по умолчанию.

        Пока пользователь не приблизил вид вручную, водопад показывает
        именно этот диапазон, а не «растущие» по мере свипа данные.
        """
        span = self._clamp_span(lo_hz, hi_hz)
        if span is None or self._auto_span_hz == span:
            return
        self._auto_span_hz = span
        self._apply_limits()
        self._sync_x()

    @staticmethod
    def _clamp_span(lo_hz: float, hi_hz: float) -> tuple[float, float] | None:
        try:
            lo = float(lo_hz)
            hi = float(hi_hz)
        except (TypeError, ValueError):
            return None
        if not (hi > lo):
            return None
        lo = min(max(lo, 70e6), 6e9)
        hi = min(max(hi, 70e6), 6e9)
        if hi <= lo:
            return None
        return lo, hi

    def _on_manual_range(self, *_args) -> None:
        """Пользователь сам подвинул/приблизил вид мышью - фиксируем диапазон."""
        f = self.unit_factor or 1.0
        x0, x1 = self.plot.getViewBox().viewRange()[0]
        self._view_locked = True
        self._locked_span_hz = (float(x0) * f, float(x1) * f)
        self._applied_hz = self._locked_span_hz

    def _sync_x(self, force: bool = False) -> None:
        """Применить нужный диапазон по частоте (с гистерезисом)."""
        span = self._locked_span_hz or self._auto_span_hz
        if span is None:
            if self._hz1 > self._hz0:
                span = (self._hz0, self._hz1)
            else:
                return
        span = self._clamp_span(*span)
        if span is None:
            return
        prev = self._applied_hz
        if not force and prev is not None:
            width = max(span[1] - span[0], 1.0)
            if (
                abs(prev[0] - span[0]) <= 0.005 * width
                and abs(prev[1] - span[1]) <= 0.005 * width
            ):
                return
        f = self.unit_factor or 1.0
        self.plot.setXRange(span[0] / f, span[1] / f, padding=0.0)
        self._applied_hz = span

    def _update_rect(self) -> None:
        f = self.unit_factor
        x0 = self._hz0 / f
        x1 = self._hz1 / f
        if x1 <= x0:
            x1 = x0 + 1.0
        self.image.setRect(QtCore.QRectF(x0, 0.0, x1 - x0, float(self.rows)))
        self._sync_x()

    def add_row(self, freqs_hz: np.ndarray, psd_db: np.ndarray) -> None:
        f = np.asarray(freqs_hz, dtype=np.float64)
        p = np.asarray(psd_db, dtype=np.float32)
        if p.size == 0:
            return

        need_new = (
            self._data is None
            or self._data.shape[0] != self.rows
            or self._data.shape[1] != p.size
        )
        if need_new:
            buf = np.full(
                (self.rows, p.size), float(self._levels[0]), dtype=np.float32
            )
            # Если строка стала длиннее — это тот же свип: старые строки
            # прижаты влево (свип идёт от начала диапазона вверх), сохраняем их.
            # Если короче — начался новый свип, буфер обнуляем.
            grew = (
                self._data is not None
                and self._data.shape[0] == self.rows
                and self._data.shape[1] < p.size
            )
            if grew:
                buf[:, : self._data.shape[1]] = self._data
                buf[1:] = buf[:-1]
            self._data = buf
            if f.size:
                self._hz0, self._hz1 = float(f[0]), float(f[-1])
                self._update_rect()
        else:
            self._data[1:] = self._data[:-1]

        self._data[0, :] = p
        if f.size and (float(f[0]) != self._hz0 or float(f[-1]) != self._hz1):
            self._hz0, self._hz1 = float(f[0]), float(f[-1])
            self._update_rect()

        # image[width=частота, height=время]
        self.image.setImage(self._data.T, autoLevels=False, levels=self._levels)
