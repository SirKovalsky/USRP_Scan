"""Потоки обработки: свип, реальное время, мониторинг ГНСС и офлайн-разбор."""

from __future__ import annotations

import threading
import time

import numpy as np
from PyQt5 import QtCore

from . import dsp, offline
from .config import AcquisitionConfig
from .device import USRPDevice
from .gnss import GnssMonitor
from .recorder import IQRecorder


class AcquisitionWorker(QtCore.QThread):
    """Владеет устройством и выполняет измерения в отдельном потоке."""

    spectrum_ready = QtCore.pyqtSignal(object, object)   # freqs[Гц], psd[дБ]
    row_ready = QtCore.pyqtSignal(object, object)        # готовая строка водопада
    gnss_ready = QtCore.pyqtSignal(object)               # gnss.GnssAnalysis
    device_info = QtCore.pyqtSignal(object)              # dict
    progress = QtCore.pyqtSignal(int, int)               # сделано, всего
    recording = QtCore.pyqtSignal(bool, str)             # идёт запись, путь
    status = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)

    def __init__(self, config: AcquisitionConfig, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._lock = threading.Lock()
        self._cfg = config
        self._dirty = True
        self._stop = threading.Event()
        self._device: USRPDevice | None = None
        self._rec_lock = threading.Lock()
        self._recorder: IQRecorder | None = None

    # ------------------------------------------------------------------
    # Управление из GUI
    # ------------------------------------------------------------------
    def configure(self, config: AcquisitionConfig) -> None:
        with self._lock:
            self._cfg = config
            self._dirty = True

    def stop(self) -> None:
        self._stop.set()

    def _snapshot(self) -> AcquisitionConfig:
        with self._lock:
            return self._cfg

    # ------------------------------------------------------------------
    # Запись I/Q
    # ------------------------------------------------------------------
    def start_recording(self, base_path: str, fmt: str, metadata: dict) -> None:
        recorder = IQRecorder(base_path, metadata, fmt).open()
        with self._rec_lock:
            self._recorder = recorder
        self.recording.emit(True, str(recorder.iq_path))

    def stop_recording(self) -> None:
        with self._rec_lock:
            recorder = self._recorder
            self._recorder = None
        if recorder is not None:
            json_path = recorder.close()
            self.recording.emit(False, str(json_path) if json_path else "")

    def _record(self, samples: np.ndarray) -> None:
        with self._rec_lock:
            recorder = self._recorder
            if recorder is None:
                return
            try:
                recorder.write(samples)
            except Exception as exc:  # noqa: BLE001
                self._recorder = None
                self.error.emit(f"Ошибка записи I/Q: {exc}")
                try:
                    recorder.close()
                except Exception:
                    pass
                self.recording.emit(False, "")

    # ------------------------------------------------------------------
    def run(self) -> None:  # noqa: C901 - основной цикл
        cfg = self._snapshot()
        try:
            device = USRPDevice(cfg.device_args, cfg.channel)
        except Exception as exc:
            self.error.emit(str(exc))
            return

        self._device = device
        monitor = GnssMonitor(
            jam_delta_db=cfg.gnss_jam_delta_db,
            nb_prominence_db=cfg.gnss_nb_db,
            saturation_db=cfg.gnss_saturation_db,
        )
        try:
            info = device.get_info()
            self.device_info.emit(info)
            self._apply_device_settings(device, cfg)
            self.status.emit("Запущено")
            while not self._stop.is_set():
                cfg = self._snapshot()
                with self._lock:
                    dirty = self._dirty
                    self._dirty = False
                if dirty:
                    self._apply_device_settings(device, cfg)
                    monitor = GnssMonitor(
                        jam_delta_db=cfg.gnss_jam_delta_db,
                        nb_prominence_db=cfg.gnss_nb_db,
                        saturation_db=cfg.gnss_saturation_db,
                    )
                if cfg.mode == "realtime":
                    self._run_realtime(device, cfg)
                elif cfg.mode == "gnss":
                    self._run_gnss(device, cfg, monitor)
                else:
                    self._run_sweep(device, cfg)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"Ошибка измерения: {exc}")
        finally:
            self.stop_recording()
            try:
                device.close()
            except Exception:
                pass
            self._device = None
            self.status.emit("Остановлено")

    # ------------------------------------------------------------------
    def _apply_device_settings(self, device: USRPDevice, cfg: AcquisitionConfig) -> None:
        device.set_sample_rate(cfg.sample_rate)
        try:
            device.set_bandwidth(min(cfg.sample_rate * 0.9, 56e6))
        except Exception:
            pass
        device.set_gain(cfg.gain)
        device.set_antenna(cfg.antenna)

    def _run_sweep(self, device: USRPDevice, cfg: AcquisitionConfig) -> None:
        centers = cfg.sweep_frequencies()
        total = len(centers)
        ncap = cfg.samples_per_capture()
        rate = device.sample_rate or cfg.sample_rate

        line_f: list[np.ndarray] = []
        line_p: list[np.ndarray] = []
        last_freq = -np.inf
        skipped = 0

        for i, cf in enumerate(centers):
            if self._stop.is_set():
                return
            with self._lock:
                if self._dirty:
                    return

            device.set_center_freq(cf)
            if cfg.settle_time > 0:
                time.sleep(cfg.settle_time)

            samples = device.recv_samples(ncap, timeout=1.0)
            if samples.size < cfg.fft_size:
                skipped += 1
                continue
            self._record(samples)

            seg_f, seg_p = dsp.compute_psd(
                samples, rate, cfg.fft_size, cfg.window, cfg.overlap, cfg.averages
            )
            seg_f = seg_f + cf
            mask = (
                (seg_f >= cfg.start_freq)
                & (seg_f <= cfg.stop_freq)
                & (seg_f > last_freq)
            )
            if not np.any(mask):
                continue
            f = seg_f[mask]
            p = seg_p[mask]
            line_f.append(f)
            line_p.append(p)
            last_freq = float(f[-1])

            self.spectrum_ready.emit(np.concatenate(line_f), np.concatenate(line_p))
            self.progress.emit(i + 1, total)

        if line_f:
            full_f = np.concatenate(line_f)
            full_p = np.concatenate(line_p)
            self.row_ready.emit(full_f, full_p)
            message = (
                f"Свип: {len(full_f)} точек, "
                f"{dsp.format_freq(full_f[0])} .. {dsp.format_freq(full_f[-1])}"
            )
            if skipped:
                message += f" (пропущено шагов: {skipped})"
            self.status.emit(message)
        else:
            self.error.emit(
                "Не удалось получить данные ни на одной частоте.\n"
                "Проверьте антенну, усиление и аргументы устройства."
            )

    def _run_realtime(self, device: USRPDevice, cfg: AcquisitionConfig) -> None:
        n = cfg.samples_per_block()
        rate = device.sample_rate or cfg.sample_rate
        device.set_center_freq(cfg.center_freq)
        device.start_continuous()
        self.status.emit("Реальное время: поток запущен")
        empty = 0
        try:
            while not self._stop.is_set():
                with self._lock:
                    if self._dirty:
                        return
                samples = device.recv_block(n, timeout=0.25)
                if samples.size < cfg.fft_size:
                    empty += 1
                    if empty == 20:
                        self.status.emit(
                            "Нет данных от устройства: проверьте антенну и USB."
                        )
                    continue
                empty = 0
                self._record(samples)
                f, p = dsp.compute_psd(
                    samples, rate, cfg.fft_size, cfg.window, cfg.overlap, cfg.averages
                )
                f = f + cfg.center_freq
                self.spectrum_ready.emit(f, p)
                self.row_ready.emit(f, p)
        finally:
            try:
                device.stop_continuous()
            except Exception:
                pass

    def _run_gnss(self, device: USRPDevice, cfg: AcquisitionConfig,
                  monitor: GnssMonitor) -> None:
        n = cfg.samples_per_block()
        rate = device.sample_rate or cfg.sample_rate
        device.set_center_freq(cfg.center_freq)
        device.start_continuous()
        self.status.emit("Мониторинг ГНСС: поток запущен")
        empty = 0
        try:
            while not self._stop.is_set():
                with self._lock:
                    if self._dirty:
                        return
                samples = device.recv_block(n, timeout=0.25)
                if samples.size < cfg.fft_size:
                    empty += 1
                    if empty == 20:
                        self.status.emit(
                            "Нет данных от устройства: проверьте антенну и USB."
                        )
                    continue
                empty = 0
                self._record(samples)
                f, p = dsp.compute_psd(
                    samples, rate, cfg.fft_size, cfg.window, cfg.overlap, cfg.averages
                )
                f = f + cfg.center_freq
                self.spectrum_ready.emit(f, p)
                self.row_ready.emit(f, p)
                self.gnss_ready.emit(monitor.analyze(f, p))
        finally:
            try:
                device.stop_continuous()
            except Exception:
                pass


class OfflineWorker(QtCore.QThread):
    """Разбор записанного файла I/Q: спектр и (опционально) анализ помех."""

    spectrum_ready = QtCore.pyqtSignal(object, object)
    row_ready = QtCore.pyqtSignal(object, object)
    gnss_ready = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(int, int)
    status = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        path: str,
        config: AcquisitionConfig,
        sample_rate: float,
        center_freq: float,
        metadata: dict | None = None,
        gnss: bool = False,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.path = path
        self._cfg = config
        self.rate = float(sample_rate)
        self.center = float(center_freq)
        self.metadata = metadata
        self.gnss = bool(gnss)
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:  # noqa: C901
        try:
            total = offline.iq_sample_count(self.path, self.metadata)
            if total <= 0:
                self.error.emit("Файл пуст или недоступен.")
                return
            block = max(int(self._cfg.fft_size) * max(1, int(self._cfg.averages)),
                        int(self._cfg.fft_size))
            monitor = GnssMonitor(
                jam_delta_db=self._cfg.gnss_jam_delta_db,
                nb_prominence_db=self._cfg.gnss_nb_db,
                saturation_db=self._cfg.gnss_saturation_db,
            ) if self.gnss else None

            acc: np.ndarray | None = None
            count = 0
            last_f: np.ndarray | None = None
            pos = 0
            while pos < total and not self._stop.is_set():
                chunk = offline.read_iq(self.path, self.metadata, start=pos, count=block)
                if chunk.size < self._cfg.fft_size:
                    break
                pos += int(chunk.size)
                f, p = dsp.compute_psd(
                    chunk, self.rate, self._cfg.fft_size, self._cfg.window,
                    self._cfg.overlap, self._cfg.averages,
                )
                f = f + self.center
                last_f = f
                if acc is None:
                    acc = np.zeros_like(p)
                acc += np.power(10.0, p / 10.0)
                count += 1
                self.spectrum_ready.emit(f, p)
                self.row_ready.emit(f, p)
                if monitor is not None:
                    self.gnss_ready.emit(monitor.analyze(f, p))
                self.progress.emit(pos, total)

            if count and acc is not None and last_f is not None:
                avg = 10.0 * np.log10(acc / count + 1e-24)
                self.spectrum_ready.emit(last_f, avg)
                dur = total / self.rate if self.rate > 0 else 0.0
                self.status.emit(
                    f"Файл обработан: {total} отсчётов, {dur:.2f} с, "
                    f"{count} блоков"
                )
            else:
                self.error.emit("Не удалось обработать файл.")
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"Ошибка чтения I/Q: {exc}")
