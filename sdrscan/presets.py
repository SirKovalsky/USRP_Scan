"""Каталог предустановленных диапазонов: BLE, Wi-Fi, LTE, ГНСС и другие.

Диапазоны заданы парами «начало/конец» в герцах и предназначены для режима
свипа (sweep). Частота дискретизации (``rate_hz``) - рекомендация, чтобы одним
захватом покрыть выбранный канал.

Диапазон приёма USRP B210: 70 МГц .. 6 ГГц. Предустановки вне этих границ не
включаются в каталог.
"""

from __future__ import annotations

from dataclasses import dataclass

MHZ = 1e6
GHZ = 1e9


@dataclass(frozen=True)
class Preset:
    """Одна предустановка диапазона."""

    name: str          # подпись в списке
    start_hz: float    # начало полосы, Гц
    stop_hz: float     # конец полосы, Гц
    rate_hz: float | None = None   # рекомендуемая частота дискретизации, Гц
    note: str = ""     # пояснение

    @property
    def center_hz(self) -> float:
        return 0.5 * (self.start_hz + self.stop_hz)

    @property
    def span_hz(self) -> float:
        return self.stop_hz - self.start_hz


def _label(hz: float) -> str:
    if abs(hz) >= GHZ:
        return f"{hz / GHZ:.3f} ГГц"
    return f"{hz / MHZ:.3f} МГц"


# ----------------------------------------------------------------------
# BLE (Bluetooth Low Energy)
# ----------------------------------------------------------------------
def _ble() -> list[Preset]:
    out = [
        Preset("BLE: весь диапазон (2400-2483.5 МГц)", 2400 * MHZ, 2483.5 * MHZ,
               4 * MHZ, "40 каналов BLE с шагом 2 МГц"),
    ]
    for ch, fc in ((37, 2402 * MHZ), (38, 2426 * MHZ), (39, 2480 * MHZ)):
        out.append(
            Preset(f"BLE: рекламный канал {ch} ({fc / MHZ:.0f} МГц)",
                   fc - 1 * MHZ, fc + 1 * MHZ, 2 * MHZ,
                   f"Advertising-канал BLE {ch}")
        )
    out += [
        Preset("BLE: каналы 0-10 (2400-2422 МГц)", 2400 * MHZ, 2422 * MHZ, 4 * MHZ),
        Preset("BLE: каналы 11-22 (2422-2444 МГц)", 2422 * MHZ, 2444 * MHZ, 4 * MHZ),
        Preset("BLE: каналы 23-36 (2444-2472 МГц)", 2444 * MHZ, 2472 * MHZ, 4 * MHZ),
        Preset("BLE: канал 0 (2402 МГц)", 2401 * MHZ, 2403 * MHZ, 2 * MHZ),
    ]
    return out


# ----------------------------------------------------------------------
# Wi-Fi (по каналам)
# ----------------------------------------------------------------------
_WIFI_5_CHANNELS = [
    36, 40, 44, 48,             # UNII-1
    52, 56, 60, 64,             # UNII-2A
    100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144,  # UNII-2C
    149, 153, 157, 161, 165, 169, 173, 177,                     # UNII-3 / ISM
]


def _wifi() -> list[Preset]:
    out = [
        Preset("Wi-Fi 2.4 ГГц: весь диапазон (2401-2483 МГц)",
               2401 * MHZ, 2483.5 * MHZ, 4 * MHZ),
    ]
    for ch in range(1, 14):
        fc = 2412 * MHZ + 5 * MHZ * (ch - 1)
        out.append(
            Preset(f"Wi-Fi 2.4 ГГц, канал {ch} ({fc / MHZ:.0f} МГц)",
                   fc - 10 * MHZ, fc + 10 * MHZ, 20 * MHZ,
                   f"20 МГц, центр {_label(fc)}")
        )
    out.append(
        Preset("Wi-Fi 2.4 ГГц, канал 14 (2484 МГц)",
               2474 * MHZ, 2494 * MHZ, 20 * MHZ, "Япония, 20 МГц")
    )
    out.append(
        Preset("Wi-Fi 5 ГГц: весь диапазон (5150-5895 МГц)",
               5150 * MHZ, 5895 * MHZ, 4 * MHZ)
    )
    for ch in _WIFI_5_CHANNELS:
        fc = 5000 * MHZ + 5 * MHZ * ch
        out.append(
            Preset(f"Wi-Fi 5 ГГц, канал {ch} ({fc / MHZ:.0f} МГц)",
                   fc - 10 * MHZ, fc + 10 * MHZ, 20 * MHZ,
                   f"20 МГц, центр {_label(fc)}")
        )
    # 6 ГГц: B210 принимает только до 6 ГГц, поэтому даём доступную часть.
    out += [
        Preset("Wi-Fi 6E: доступная часть (5925-6000 МГц)",
               5925 * MHZ, 6000 * MHZ, 4 * MHZ,
               "6E диапазон 5925-7125 МГц; B210 принимает лишь до 6000 МГц"),
    ]
    for ch in (1, 5, 9, 13, 17):
        fc = 5950 * MHZ + 5 * MHZ * ch
        if fc + 10 * MHZ > 6000 * MHZ:
            continue
        out.append(
            Preset(f"Wi-Fi 6E, канал {ch} ({fc / MHZ:.0f} МГц)",
                   fc - 10 * MHZ, fc + 10 * MHZ, 20 * MHZ)
        )
    return out


# ----------------------------------------------------------------------
# LTE (E-UTRA, по бэндам)
# ----------------------------------------------------------------------
# (номер, подпись, DL (lo,hi) МГц | None, UL (lo,hi) МГц | None)
_LTE_FDD = [
    (1, "2100", (2110, 2170), (1920, 1980)),
    (2, "1900", (1930, 1990), (1850, 1910)),
    (3, "1800", (1805, 1880), (1710, 1785)),
    (4, "AWS-1", (2110, 2155), (1710, 1755)),
    (5, "850", (869, 894), (824, 849)),
    (7, "2600", (2620, 2690), (2500, 2570)),
    (8, "900", (925, 960), (880, 915)),
    (12, "700a", (729, 746), (699, 716)),
    (13, "700c", (746, 756), (777, 787)),
    (14, "700 PS", (758, 768), (788, 798)),
    (17, "700b", (734, 746), (704, 716)),
    (18, "850 JP", (860, 875), (815, 830)),
    (19, "850 JP+", (875, 890), (830, 845)),
    (20, "800 DD", (791, 821), (832, 862)),
    (21, "1500 JP", (1495, 1510), (1447, 1462)),
    (25, "1900+", (1930, 1995), (1850, 1915)),
    (26, "850+", (859, 894), (814, 849)),
    (28, "700 APT", (758, 803), (703, 748)),
    (30, "2300 WCS", (2350, 2360), (2305, 2315)),
    (32, "1500 L", (1452, 1496), None),
    (66, "AWS-3", (2110, 2200), (1710, 1780)),
    (71, "600", (617, 652), (663, 698)),
]

_LTE_TDD = [
    (33, "1900 TDD", (1900, 1920)),
    (34, "2000 TDD", (2010, 2025)),
    (35, "1900 TDD", (1850, 1910)),
    (38, "2600 TDD", (2570, 2620)),
    (39, "1900 TDD", (1880, 1920)),
    (40, "2300 TDD", (2300, 2400)),
    (41, "2500 TDD", (2496, 2690)),
    (42, "3500 TDD", (3400, 3600)),
    (43, "3700 TDD", (3600, 3800)),
    (46, "LAA 5 ГГц", (5150, 5925)),
    (47, "V2X 5.9 ГГц", (5855, 5925)),
    (48, "CBRS 3.5 ГГц", (3550, 3700)),
]


def _lte() -> list[Preset]:
    out: list[Preset] = []
    for n, label, dl, ul in _LTE_FDD:
        if dl:
            lo, hi = dl
            out.append(
                Preset(f"LTE B{n} ({label}): DL {lo}-{hi} МГц",
                       lo * MHZ, hi * MHZ, 20 * MHZ,
                       f"E-UTRA Band {n}, нисходящий (базовая станция)")
            )
        if ul:
            lo, hi = ul
            out.append(
                Preset(f"LTE B{n} ({label}): UL {lo}-{hi} МГц",
                       lo * MHZ, hi * MHZ, 20 * MHZ,
                       f"E-UTRA Band {n}, восходящий (абонент)")
            )
    for n, label, tdd in _LTE_TDD:
        lo, hi = tdd
        out.append(
            Preset(f"LTE B{n} ({label}): {lo}-{hi} МГц",
                   lo * MHZ, hi * MHZ, 20 * MHZ,
                   f"E-UTRA Band {n}, TDD (приём и передача в одной полосе)")
        )
    return out


# ----------------------------------------------------------------------
# ГНСС (по системам и частотам)
# ----------------------------------------------------------------------
# (система, сигнал, центральная частота МГц)
_GNSS_SIGNALS = [
    ("GPS", "L1 C/A", 1575.42),
    ("GPS", "L2", 1227.60),
    ("GPS", "L5", 1176.45),
    ("ГЛОНАСС", "L1", 1602.00),
    ("ГЛОНАСС", "L2", 1246.00),
    ("ГЛОНАСС", "L3 (CDMA)", 1202.025),
    ("Galileo", "E1", 1575.42),
    ("Galileo", "E5a", 1176.45),
    ("Galileo", "E5b", 1207.14),
    ("Galileo", "E5 AltBOC", 1191.795),
    ("Galileo", "E6", 1278.75),
    ("BeiDou", "B1I", 1561.098),
    ("BeiDou", "B1C", 1575.42),
    ("BeiDou", "B2a", 1176.45),
    ("BeiDou", "B2b", 1207.14),
    ("BeiDou", "B3I", 1268.52),
    ("QZSS", "L1", 1575.42),
    ("QZSS", "L2", 1227.60),
    ("QZSS", "L5", 1176.45),
    ("QZSS", "L6", 1278.75),
    ("NavIC (IRNSS)", "L5", 1176.45),
    ("NavIC (IRNSS)", "S", 2492.028),
    ("SBAS", "L1", 1575.42),
    ("SBAS", "L5", 1176.45),
]


def _gnss() -> list[Preset]:
    out = [
        Preset("ГНСС: L1/E1/B1 (1559-1610 МГц)", 1559 * MHZ, 1610 * MHZ, 4 * MHZ,
               "GPS L1, Galileo E1, BeiDou B1, ГЛОНАСС L1, QZSS L1, SBAS"),
        Preset("ГНСС: L2 (1215-1240 МГц)", 1215 * MHZ, 1240 * MHZ, 4 * MHZ,
               "GPS L2, ГЛОНАСС L2, QZSS L2"),
        Preset("ГНСС: L5/E5/B2a (1160-1215 МГц)", 1160 * MHZ, 1215 * MHZ, 4 * MHZ,
               "GPS L5, Galileo E5, BeiDou B2, QZSS L5, NavIC L5"),
        Preset("ГНСС: E6/B3/L6 (1260-1300 МГц)", 1260 * MHZ, 1300 * MHZ, 4 * MHZ,
               "Galileo E6, BeiDou B3I, QZSS L6"),
    ]
    for system, signal, fc_mhz in _GNSS_SIGNALS:
        fc = fc_mhz * MHZ
        out.append(
            Preset(f"ГНСС: {system} {signal} ({fc_mhz:.3f} МГц)",
                   fc - 5 * MHZ, fc + 5 * MHZ, 10 * MHZ,
                   f"{system}, сигнал {signal}, центр {_label(fc)}")
        )
    return out


# ----------------------------------------------------------------------
# Прочие популярные диапазоны
# ----------------------------------------------------------------------
def _other() -> list[Preset]:
    return [
        Preset("FM-вещание (87.5-108 МГц)", 87.5 * MHZ, 108 * MHZ, 4 * MHZ,
               "Радиовещание УКВ/FM"),
        Preset("DAB/DAB+ (174-240 МГц)", 174 * MHZ, 240 * MHZ, 4 * MHZ,
               "Цифровое радиовещание"),
        Preset("DVB-T (470-790 МГц)", 470 * MHZ, 790 * MHZ, 4 * MHZ,
               "Цифровое ТВ"),
        Preset("GSM/LTE 900: DL (925-960 МГц)", 925 * MHZ, 960 * MHZ, 4 * MHZ),
        Preset("GSM 900: UL (890-915 МГц)", 890 * MHZ, 915 * MHZ, 4 * MHZ),
        Preset("GSM/LTE 1800: DL (1805-1880 МГц)", 1805 * MHZ, 1880 * MHZ, 4 * MHZ),
        Preset("UMTS/LTE 2100: DL (2110-2170 МГц)", 2110 * MHZ, 2170 * MHZ, 4 * MHZ),
        Preset("TETRA (380-400 МГц)", 380 * MHZ, 400 * MHZ, 4 * MHZ,
               "Профессиональная радиосвязь"),
        Preset("TETRA (410-430 МГц)", 410 * MHZ, 430 * MHZ, 4 * MHZ),
        Preset("ISM 433 МГц (433.05-434.79 МГц)",
               433.05 * MHZ, 434.79 * MHZ, 2 * MHZ,
               "Пульты, датчики, LoRa (регион 1)"),
        Preset("ISM/LoRa 868 МГц (863-870 МГц)", 863 * MHZ, 870 * MHZ, 2 * MHZ,
               "LoRa, Sigfox, датчики (Европа)"),
        Preset("ISM 915 МГц (902-928 МГц)", 902 * MHZ, 928 * MHZ, 4 * MHZ,
               "ISM, LoRa (Америка)"),
        Preset("PMR446 (446.0-446.2 МГц)", 446.0 * MHZ, 446.2 * MHZ, 1 * MHZ,
               "Безлицензионные рации"),
        Preset("DECT (1880-1900 МГц)", 1880 * MHZ, 1900 * MHZ, 4 * MHZ,
               "Беспроводные телефоны"),
        Preset("ADS-B (1080-1100 МГц)", 1080 * MHZ, 1100 * MHZ, 4 * MHZ,
               "Ответчики воздушных судов, 1090 МГц"),
        Preset("AIS (161.5-162.5 МГц)", 161.5 * MHZ, 162.5 * MHZ, 2 * MHZ,
               "Морские суда, 161.975/162.025 МГц"),
    ]


# Порядок категорий сохраняется (Python 3.7+).
CATEGORIES: dict[str, list[Preset]] = {
    "BLE (Bluetooth LE)": _ble(),
    "Wi-Fi (по каналам)": _wifi(),
    "LTE (по бэндам)": _lte(),
    "ГНСС (GNSS)": _gnss(),
    "Прочие диапазоны": _other(),
}


def category_names() -> list[str]:
    return list(CATEGORIES.keys())


def presets_for(category: str) -> list[Preset]:
    return list(CATEGORIES.get(category, []))
